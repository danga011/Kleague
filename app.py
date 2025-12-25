
import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from openai import OpenAI
import base64
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- 페이지 설정 (가장 먼저 실행되어야 함) ---
st.set_page_config(
    page_title="K-Scout Adapt-Fit AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# WVS(도시) C-Fit 모듈 로드 (Streamlit 실행 환경에서도 안정적으로)
try:
    from src.wvs_city_fit import load_wvs_city_vectors_and_stats, compute_wvs_city_c_fit
except ImportError:
    # src/가 모듈 경로에 없을 수 있으니, 프로젝트 루트를 sys.path에 추가 후 재시도
    import sys

    project_root = str(Path(__file__).resolve().parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.wvs_city_fit import load_wvs_city_vectors_and_stats, compute_wvs_city_c_fit

# .env 파일 로드 (앱 시작 시)
load_dotenv()

def group_position(pos):
    if pos in ['CF', 'RW', 'LW', 'SS']:
        return 'FW'
    if pos in ['CM', 'DM', 'AM', 'LM', 'RM']:
        return 'MF'
    if pos in ['CB', 'LB', 'RB', 'LWB', 'RWB']:
        return 'DF'
    return 'ETC'

def get_score_grade(score):
    """점수에 따른 등급과 색상 반환 - 브랜드 아이덴티티 적용"""
    if score >= 95:
        return "S+", "#FE3D67", "완벽한 적합 (Excellent)"
    elif score >= 90:
        return "S", "#FF7031", "최상위 적합"
    elif score >= 85:
        return "A+", "#872B95", "매우 우수"
    elif score >= 80:
        return "A", "#22c55e", "우수"
    elif score >= 75:
        return "B+", "#3b82f6", "양호"
    elif score >= 70:
        return "B", "#FF7031", "평균 이상"
    elif score >= 65:
        return "C", "#f59e0b", "보통"
    else:
        return "D", "#ef4444", "부적합"

def get_position_korean(pos):
    """포지션을 한글로 변환"""
    pos_map = {
        'CF': '중앙 공격수', 'RW': '오른쪽 윙어', 'LW': '왼쪽 윙어', 'SS': '섀도 스트라이커',
        'CM': '중앙 미드필더', 'DM': '수비형 미드필더', 'AM': '공격형 미드필더',
        'LM': '왼쪽 미드필더', 'RM': '오른쪽 미드필더',
        'CB': '중앙 수비수', 'LB': '왼쪽 풀백', 'RB': '오른쪽 풀백',
        'LWB': '왼쪽 윙백', 'RWB': '오른쪽 윙백',
        'FW': '공격수', 'MF': '미드필더', 'DF': '수비수', 'ETC': '기타'
    }
    return pos_map.get(pos, pos)

# --- 데이터 로드 ---
@st.cache_data
def load_data():
    try:
        hsi_df = pd.read_csv('output/hsi_scores_2024.csv')
        foreigners_df = pd.read_csv('output/2024_foreigners_list.csv')
        with open('output/team_templates.json', 'r') as f:
            templates = json.load(f)
    except FileNotFoundError as e:
        st.error(f"필수 데이터 파일이 없습니다: {e}. 이전 단계를 먼저 실행해주세요.")
        return None, None, None
    return hsi_df, templates, foreigners_df


# ============================================================
# WVS(도시) 기반 C-Fit 유틸
# ============================================================
WVS_CITY_VECTORS_PATH = Path("data/processed/wvs_city_vectors.csv")
WVS_LOC_LABELS_PATH = Path("data/processed/wvs_loc_labels.csv")
PLAYER_CITY_MAP_PATH = Path("data/processed/player_upbringing_city_map.csv")
TEAM_CITY_MAP_PATH = Path("data/processed/kleague_team_city_map.csv")
PLAYER_PROFILE_PATH = Path("data/raw/player_profile.csv")


@st.cache_data
def load_wvs_city_vectors_cached():
    """
    Load pre-aggregated WVS city vectors.
    - Build file via: `python src/wvs_city_profiler.py`
    """
    if not WVS_CITY_VECTORS_PATH.exists():
        return None, None, None, None, None
    city_map, country_map, global_vector, var_trad, var_surv = load_wvs_city_vectors_and_stats(WVS_CITY_VECTORS_PATH)
    return city_map, country_map, global_vector, var_trad, var_surv


def ensure_player_city_map(player_names):
    """Create a mapping template file if missing. User fills (country_alpha, loc_code)."""
    PLAYER_CITY_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PLAYER_CITY_MAP_PATH.exists():
        df = pd.read_csv(PLAYER_CITY_MAP_PATH)
        if "player_name_ko" not in df.columns:
            df = pd.DataFrame(
                columns=[
                    "player_name_ko",
                    "club_name_ko",
                    "home_country_alpha",
                    "home_loc_code",
                    "home_city_label",
                    "english_full_name",
                    "nationality",
                    "source",
                ]
            )
    else:
        df = pd.DataFrame(
            {
                "player_name_ko": sorted(set(player_names)),
                "club_name_ko": "",
                "home_country_alpha": "",
                "home_loc_code": "",
                "home_city_label": "",
                "english_full_name": "",
                "nationality": "",
                "source": "",
            }
        )
        df.to_csv(PLAYER_CITY_MAP_PATH, index=False)
    return df


def ensure_team_city_map(team_names):
    """Create a mapping template file if missing. User fills (country_alpha, loc_code)."""
    TEAM_CITY_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    # WVS 7차 조사(KOR) 기준 지역 코드 매핑
    # 410001: 서울, 410004: 인천, 410006: 대전, 410003: 대구, 410005: 광주, 410007: 울산
    # 410008: 경기(수원), 410009: 강원(춘천), 410012: 전북(전주), 410014: 경북(포항/김천), 410016: 제주(서귀포)
    default_team_info = {
        "FC서울": {"label": "서울", "loc": 410001},
        "강원FC": {"label": "춘천", "loc": 410009},
        "광주FC": {"label": "광주", "loc": 410005},
        "김천 상무 프로축구단": {"label": "김천", "loc": 410014},
        "대구FC": {"label": "대구", "loc": 410003},
        "대전 하나 시티즌": {"label": "대전", "loc": 410006},
        "수원FC": {"label": "수원", "loc": 410008},
        "울산 HD FC": {"label": "울산", "loc": 410007},
        "인천 유나이티드": {"label": "인천", "loc": 410004},
        "전북 현대 모터스": {"label": "전주", "loc": 410012},
        "제주SK FC": {"label": "서귀포", "loc": 410016},
        "포항 스틸러스": {"label": "포항", "loc": 410014},
    }
    
    if TEAM_CITY_MAP_PATH.exists():
        df = pd.read_csv(TEAM_CITY_MAP_PATH)
        # 필수 컬럼이 없으면 재생성
        if not all(col in df.columns for col in ["team_name_ko", "host_country_alpha", "host_loc_code", "host_city_label"]):
             df = pd.DataFrame(columns=["team_name_ko", "host_country_alpha", "host_loc_code", "host_city_label"])
    else:
        df = pd.DataFrame(columns=["team_name_ko", "host_country_alpha", "host_loc_code", "host_city_label"])

    # 누락된 팀이나 코드가 없는 팀 자동 업데이트
    changed = False
    for team_name in sorted(set(team_names)):
        info = default_team_info.get(team_name, {"label": "", "loc": ""})
        mask = df["team_name_ko"] == team_name
        
        if not mask.any():
            new_row = {
                "team_name_ko": team_name,
                "host_country_alpha": "KOR",
                "host_loc_code": info["loc"],
                "host_city_label": info["label"]
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            changed = True
        else:
            # 기존 행이 있지만 loc_code가 비어있는 경우 채워줌
            if pd.isna(df.loc[mask, "host_loc_code"].iloc[0]) or str(df.loc[mask, "host_loc_code"].iloc[0]).strip() == "":
                df.loc[mask, "host_loc_code"] = info["loc"]
                df.loc[mask, "host_city_label"] = info["label"]
                df.loc[mask, "host_country_alpha"] = "KOR"
                changed = True
                
    if changed:
        df.to_csv(TEAM_CITY_MAP_PATH, index=False)
    return df


@st.cache_data
def load_wvs_city_vectors_df_cached():
    """
    Load WVS city vectors as DataFrame (latest-only per loc_code).
    This is used only to help users manually fill LOC_CODEs.
    """
    if not WVS_CITY_VECTORS_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(WVS_CITY_VECTORS_PATH)
        required = {"country_alpha", "loc_code", "year", "n", "tradagg_mean", "survsagg_mean"}
        if not required.issubset(set(df.columns)):
            return pd.DataFrame()
        df = df.copy()
        df["country_alpha"] = df["country_alpha"].astype(str).str.upper()
        df["loc_code"] = pd.to_numeric(df["loc_code"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df = df.dropna(subset=["country_alpha", "loc_code", "year", "n"]).copy()
        df["loc_code"] = df["loc_code"].astype(int)
        df["year"] = df["year"].astype(int)
        df["n"] = df["n"].astype(int)
        # latest-only
        df = df.sort_values(["country_alpha", "loc_code", "year"], ascending=[True, True, False])
        df = df.drop_duplicates(subset=["country_alpha", "loc_code"], keep="first")
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_wvs_loc_labels_df_cached():
    """
    Optional mapping: (country_alpha, loc_code) -> loc_label (human-readable).
    This file is not required for C-Fit computation; it only helps manual mapping UI.
    """
    if not WVS_LOC_LABELS_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(WVS_LOC_LABELS_PATH)
        required = {"country_alpha", "loc_code", "loc_label"}
        if not required.issubset(set(df.columns)):
            return pd.DataFrame()
        df = df.copy()
        df["country_alpha"] = df["country_alpha"].astype(str).str.upper()
        df["loc_code"] = pd.to_numeric(df["loc_code"], errors="coerce")
        df = df.dropna(subset=["country_alpha", "loc_code"]).copy()
        df["loc_code"] = df["loc_code"].astype(int)
        df["loc_label"] = df["loc_label"].astype(str)
        df = df.drop_duplicates(subset=["country_alpha", "loc_code"], keep="first")
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _safe_int(v):
    try:
        if pd.isna(v):
            return None
        s = str(v).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def compute_city_based_c_fit(
    player_name_ko: str, team_name_ko: str, player_club_name_ko: Optional[str] = None
) -> Tuple[Optional[float], str, str, str, Optional[Dict[str, Any]]]:
    """
    Return (c_fit, player_city_label, host_city_label, reason, meta).
    If mapping or vectors are missing, c_fit is None and reason explains why.
    """
    city_map, country_map, global_vector, var_trad, var_surv = load_wvs_city_vectors_cached()
    if city_map is None:
        return (
            None,
            "",
            "",
            f"WVS 벡터 파일이 없습니다: {WVS_CITY_VECTORS_PATH} (먼저 `python src/wvs_city_profiler.py` 실행)",
            None,
        )

    # Load mapping files (small; no caching so edits reflect)
    player_map_df = pd.read_csv(PLAYER_CITY_MAP_PATH) if PLAYER_CITY_MAP_PATH.exists() else pd.DataFrame()
    team_map_df = pd.read_csv(TEAM_CITY_MAP_PATH) if TEAM_CITY_MAP_PATH.exists() else pd.DataFrame()

    # Player city
    p_row = None
    if not player_map_df.empty and "player_name_ko" in player_map_df.columns:
        hit = player_map_df[player_map_df["player_name_ko"] == player_name_ko]
        # 동명이인 케이스: club_name_ko가 있으면 구단으로 우선 매칭
        if (
            player_club_name_ko
            and "club_name_ko" in player_map_df.columns
            and not hit.empty
        ):
            hit2 = hit[hit["club_name_ko"] == player_club_name_ko]
            if not hit2.empty:
                hit = hit2
        if not hit.empty:
            p_row = hit.iloc[0]
    if p_row is None:
        return None, "", "", f"선수 성장도시 매핑이 없습니다: {PLAYER_CITY_MAP_PATH}", None

    p_country = str(p_row.get("home_country_alpha", "")).strip().upper()
    p_loc = _safe_int(p_row.get("home_loc_code", ""))
    p_label = str(p_row.get("home_city_label", "")).strip()
    if not p_country:
        return None, p_label, "", "선수 성장국가(home_country_alpha)가 비어 있습니다", None
    # loc_code가 비어있으면 '도시 미입력'으로 간주하고 국가 단위로 fallback 계산(-1)
    if p_loc is None:
        p_loc = -1

    # Host city (analysis city for team)
    t_row = None
    if not team_map_df.empty and "team_name_ko" in team_map_df.columns:
        hit = team_map_df[team_map_df["team_name_ko"] == team_name_ko]
        if not hit.empty:
            t_row = hit.iloc[0]
    if t_row is None:
        return None, p_label, "", f"팀-도시 매핑이 없습니다: {TEAM_CITY_MAP_PATH}", None

    h_country = str(t_row.get("host_country_alpha", "")).strip().upper()
    h_loc = _safe_int(t_row.get("host_loc_code", ""))
    h_label = str(t_row.get("host_city_label", "")).strip()
    if not h_country:
        return None, p_label, h_label, "분석 국가(host_country_alpha)가 비어 있습니다", None
    # loc_code가 비어있으면 국가 단위로 fallback 계산(-1)
    if h_loc is None:
        h_loc = -1

    c_fit, meta = compute_wvs_city_c_fit(
        city_map=city_map,
        country_map=country_map,
        global_vector=global_vector,
        var_trad=var_trad,
        var_surv=var_surv,
        home_country_alpha=p_country,
        home_loc_code=p_loc,
        host_country_alpha=h_country,
        host_loc_code=h_loc,
        unknown_default=0.85,
        method="inv1p",
        reliability_k=200.0,
    )
    try:
        if isinstance(meta, dict):
            meta = dict(meta)
            meta["home_country_alpha"] = p_country
            meta["home_loc_code"] = int(p_loc)
            meta["host_country_alpha"] = h_country
            meta["host_loc_code"] = int(h_loc)
    except Exception:
        meta = None
    return float(c_fit), p_label, h_label, "", meta

def local_css():
    css = """
    <style>
    /* ===== ANYONE COMPANY 브랜드 테마 ===== */
    /* Primary: #FE3D67 (핑크) → #FF7031 (오렌지) */
    /* Secondary: #FF3B65 (핑크) → #872B95 (퍼플) */
    
    /* 메인 배경 - 다크 그라데이션 */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1a0a1e 0%, #0f0a1a 50%, #0a0f1a 100%) !important;
    }
    
    /* 모든 텍스트를 흰색으로 */
    .stApp, .stApp * {
        color: #ffffff !important;
    }
    
    /* 사이드바 배경 - 브랜드 컬러 힌트 */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1e0a24 0%, #0f0a1a 100%) !important;
        border-right: 1px solid rgba(254, 61, 103, 0.3) !important;
    }
    
    /* 사이드바 토글 버튼 - 브랜드 그라데이션 */
    [data-testid="collapsedControl"] {
        background: linear-gradient(135deg, #FE3D67 0%, #FF7031 100%) !important;
    }
    [data-testid="collapsedControl"] svg {
        stroke: #ffffff !important;
    }
    
    /* ===== 드롭다운 (검정 텍스트) ===== */
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #FE3D67 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] > div * {
        color: #000000 !important;
    }
    div[role="listbox"], div[role="listbox"] *,
    [data-baseweb="popover"], [data-baseweb="popover"] *,
    [data-baseweb="menu"], [data-baseweb="menu"] * {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* ===== 버튼 스타일 - 브랜드 그라데이션 ===== */
    .stButton > button {
        background: linear-gradient(135deg, #FE3D67 0%, #FF7031 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #FF7031 0%, #872B95 100%) !important;
        box-shadow: 0 4px 20px rgba(254, 61, 103, 0.4) !important;
    }
    
    /* ===== 테이블 스타일 ===== */
    .stTable, .stDataFrame {
        background-color: rgba(30, 10, 36, 0.8) !important;
    }
    .stTable th, .stDataFrame th {
        background: linear-gradient(135deg, #FE3D67 0%, #872B95 100%) !important;
        color: #ffffff !important;
    }
    .stTable td, .stDataFrame td {
        background-color: rgba(15, 10, 26, 0.9) !important;
        color: #ffffff !important;
        border-bottom: 1px solid rgba(254, 61, 103, 0.2) !important;
    }
    
    /* ===== 메트릭 카드 ===== */
    [data-testid="stMetricValue"] {
        color: #FE3D67 !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    
    /* ===== 탭 스타일 ===== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(30, 10, 36, 0.6) !important;
        border-radius: 10px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, #FE3D67 0%, #FF7031 100%) !important;
        border-radius: 8px !important;
    }
    
    /* ===== Expander ===== */
    .streamlit-expanderHeader {
        background-color: rgba(30, 10, 36, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid rgba(254, 61, 103, 0.3) !important;
    }
    .streamlit-expanderContent {
        background-color: rgba(15, 10, 26, 0.9) !important;
    }
    
    /* ===== 알림 박스 ===== */
    .stAlert {
        background-color: rgba(30, 10, 36, 0.8) !important;
        color: #ffffff !important;
        border-left: 4px solid #FE3D67 !important;
    }
    
    /* ===== 구분선 - 그라데이션 ===== */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, #FE3D67, #FF7031, transparent) !important;
    }
    
    /* ===== 캡션 ===== */
    .stCaption, small {
        color: #FF7031 !important;
    }
    
    /* ===== 링크 색상 ===== */
    a, a:visited {
        color: #FE3D67 !important;
    }
    a:hover {
        color: #FF7031 !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

local_css()

# --- 데이터 로드 ---
hsi_df, templates, foreigners_df = load_data()

if hsi_df is None:
    st.stop()

# WVS(도시) C-Fit 매핑 템플릿 파일 보장
try:
    ensure_player_city_map(foreigners_df["player_name_ko"].unique())
except Exception:
    pass
try:
    ensure_team_city_map(list(templates.keys()))
except Exception:
    pass

# --- 로직 함수 ---
def calculate_adapt_fit_score(player_hsi, team_template, pos_group):
    """실시간으로 선수와 팀 템플릿 간의 적합도 점수를 계산합니다."""
    def directional_similarity(player_score, team_score):
        dist = abs(player_score - team_score) / (team_score + 1e-6)
        if player_score >= team_score:
            return min(1.1, 1 / (1 + dist * 0.4) + 0.05)
        return 1 / (1 + dist)
    
    t_sim = directional_similarity(player_hsi['t_fit_score'], team_template['t_fit_score'])
    p_sim = directional_similarity(player_hsi['p_fit_score'], team_template['p_fit_score'])
    c_sim = directional_similarity(player_hsi['c_fit_score'], team_template['c_fit_score'])
    
    if pos_group == 'FW':
        score = (t_sim * 0.5) + (p_sim * 0.3) + (c_sim * 0.2)
    elif pos_group == 'MF':
        score = (t_sim * 0.4) + (p_sim * 0.4) + (c_sim * 0.2)
    elif pos_group == 'DF':
        score = (t_sim * 0.3) + (p_sim * 0.3) + (c_sim * 0.4)
    else:
        score = (t_sim * 0.33) + (p_sim * 0.33) + (c_sim * 0.34)
        
    return min(100, score * 100)

def get_all_player_scores(team_name, templates, hsi_df, foreigners_df):
    """모든 외국인 선수의 적합도 점수를 계산하여 반환"""
    scores = []
    team_templates_by_pos = templates[team_name]
    
    foreign_player_names = foreigners_df['player_name_ko'].unique()
    
    for _, player in hsi_df[hsi_df['player_name_ko'].isin(foreign_player_names)].iterrows():
        player_name = player['player_name_ko']
        try:
            player_pos = foreigners_df.loc[foreigners_df['player_name_ko'] == player_name, 'main_position'].iloc[0]
            pos_group = group_position(player_pos)
            team_template = team_templates_by_pos.get(pos_group, team_templates_by_pos.get('ETC'))
            
            if team_template:
                # WVS(도시) 기반 C-Fit을 팀(분석도시) 기준으로 동적 계산
                # 외국인 리스트에 club_name_ko가 있으면 동명이인 구분에 활용
                pclub = ""
                try:
                    if "club_name_ko" in foreigners_df.columns:
                        clubs = foreigners_df.loc[foreigners_df["player_name_ko"] == player_name, "club_name_ko"]
                        if not clubs.empty:
                            pclub = str(clubs.mode().iloc[0]).strip()
                except Exception:
                    pclub = ""
                c_fit_dyn, _, _, _, _ = compute_city_based_c_fit(player_name, team_name, pclub or None)
                player_for_score = player.copy()
                if c_fit_dyn is not None:
                    player_for_score['c_fit_score'] = float(c_fit_dyn)
                team_template_for_score = dict(team_template)
                team_template_for_score['c_fit_score'] = 1.0

                score = calculate_adapt_fit_score(player_for_score, team_template_for_score, pos_group)
                grade, color, desc = get_score_grade(score)
                scores.append({
                    'name': player_name,
                    'position': player_pos,
                    'pos_group': pos_group,
                    'score': score,
                    'grade': grade,
                    'color': color,
                    't_fit': player['t_fit_score'],
                    'p_fit': player['p_fit_score'],
                    'c_fit': player_for_score['c_fit_score']
                })
        except:
            continue
    
    return sorted(scores, key=lambda x: x['score'], reverse=True)

def get_ai_analysis_for_pdf(api_key, player_hsi, team_template, team_name, player_name, pos_group, adapt_fit_score):
    """Calls OpenAI GPT-4 to get a detailed analysis for the PDF report."""
    client = OpenAI(api_key=api_key)
    
    # 포지션별 가중치 설명
    if pos_group == 'FW':
        weight_desc = "공격수(FW)는 전술 실행(T-Fit 50%), 체력 유지(P-Fit 30%), 문화 적응(C-Fit 20%) 가중치를 적용"
    elif pos_group == 'MF':
        weight_desc = "미드필더(MF)는 전술 실행(T-Fit 40%), 체력 유지(P-Fit 40%), 문화 적응(C-Fit 20%) 가중치를 적용"
    elif pos_group == 'DF':
        weight_desc = "수비수(DF)는 전술 실행(T-Fit 30%), 체력 유지(P-Fit 30%), 문화 적응(C-Fit 40%) 가중치를 적용"
    else:
        weight_desc = "기타 포지션은 균등 가중치(각 33%)를 적용"
    
    # 차이 계산
    t_diff = player_hsi['t_fit_score'] - team_template['t_fit_score']
    p_diff = player_hsi['p_fit_score'] - team_template['p_fit_score']
    c_diff = player_hsi['c_fit_score'] - team_template['c_fit_score']

    scout_template = f"""
# K리그 선수 스카우팅 리포트

## 1. 선수 & 경기 정보
- 선수 이름: {player_name}
- 포지션: {pos_group}
- 소속 클럽: 정보 없음(추가 입력 필요)
- 주발: 정보 없음(추가 입력 필요)
- 신장 / 체중: 정보 없음(추가 입력 필요)
- 관찰 경기: 정보 없음(대회 / 라운드 / 상대 / 날짜 / 장소 / 날씨)
- 관찰 방식: 정보 없음(직관 / 영상, 풀매치/하이라이트)
- 출전 시간: 정보 없음

## 2. 전술 역할 요약
- 팀 포메이션: 정보 없음(추가 입력 필요)
- 팀 스타일(점유/역습/직선적 등): 데이터로 직접 관측 불가(추가 지표 필요)
- 선수 역할 요약 (공격/수비 시 역할, 주요 존): 아래 HSI/시너지 분석 기반으로 요약

## 3. 인 포제션(공격 시)
- 포지셔닝: 데이터로 직접 관측 불가(추가 이벤트/트래킹 지표 필요)
- 기술(터치, 패스, 드리블, 슈팅): 데이터로 직접 관측 불가(추가 지표 필요)
- 의사결정: 데이터로 직접 관측 불가(추가 지표 필요)
- 관련 수치(가능 시): 현재 리포트는 HSI(T/P/C) 중심

## 4. 아웃 오브 포제션(수비 시)
- 포지셔닝 & 커버: T-Fit(전술 실행) 기반으로 해석
- 1대1 수비: 데이터로 직접 관측 불가(추가 지표 필요)
- 전환(볼 소실 후 리액션 등): T-Fit 기반으로 해석(압박/전방 수비 참여)
- 관련 수치(가능 시): T-Fit, 팀 대비 차이

## 5. 피지컬 & 멘탈
- 피지컬(스피드, 지구력, 체격, 공중볼 등): P-Fit(혹서기 유지) 중심으로 제한적 해석
- 멘탈(집중력, 압박 대처, 리더십, 태도 등): 데이터로 직접 관측 불가(추가 정보 필요)
- 인성/훈련 태도(정보 있을 시): 정보 없음(추가 입력 필요)

## 6. 강점 / 보완점 / 발전 가능성
- 강점:
- 보완점:
- 발전 가능성:

## 7. 종합 평가 & 추천
- 종합 평가:
- 추천 사항: (즉시 영입 / 장기 모니터링 / 조건부 추천 등)
""".strip()

    prompt = f"""
    당신은 K-리그 이적 시장을 분석하는 전문 데이터 분석가입니다. 아래 데이터와 알고리즘 설명을 바탕으로, 특정 선수에 대한 상세한 스카우팅 리포트를 작성해주세요.
    
    리포트는 전문적이고 신뢰감 있는 어조로 작성해야 하며, 이모티콘 사용을 최소화하고 문장 형식의 분석을 선호합니다. 단순히 점수만 나열하지 말고, 점수의 산출 근거와 실제 경기에서의 전술적 의미를 구체적으로 서술하세요.

    [HSI 알고리즘 설명]
    1. T-Fit (Tactical): 전방 수비 강도와 파울/카드 리스크를 결합한 지표. 고강도 압박과 경기 규율 유지 능력을 동시에 평가합니다.
    2. P-Fit (Physical): 여름철 활동량 유지 비율. 기후 변화에 따른 신체 능력 유지 성능을 의미합니다.
    3. C-Fit (Cultural): 출생/성장 도시와 구단 연고지 간의 문화적 거리. 적응 기간 및 소통 리스크를 수치화한 지표입니다.
    
    [분석 대상 데이터]
    - 선수: {player_name} ({pos_group})
    - 종합 점수: {adapt_fit_score:.1f} / 100
    - 세부 지표 (선수 vs 팀 평균): 
      T-Fit: {player_hsi['t_fit_score']:.1f} (팀 평균 {team_template['t_fit_score']:.1f})
      P-Fit: {player_hsi['p_fit_score']:.2f} (팀 평균 {team_template['p_fit_score']:.2f})
      C-Fit: {player_hsi['c_fit_score']:.3f} (팀 기준 1.000)

    [리포트 작성 지침]
    1. 전문 용어를 적절히 사용하여 분석의 깊이를 더하세요.
    2. 불필요한 장식 기호나 이모티콘은 지양하십시오.
    3. 아래 템플릿의 섹션 구분을 준수하되, 각 항목은 서술형으로 작성하세요.

    {scout_template}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 K-리그 축구 전문 데이터 분석가입니다. 전문적이고 객관적인 스카우팅 리포트를 작성합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        # AI 실패 시 상세 자동 분석 생성
        # T-Fit 분석
        t_score = player_hsi['t_fit_score']
        if t_score >= 80:
            t_analysis = f"T-Fit {t_score:.1f}점은 상위권 수준의 전방 수비/압박 실행 지표입니다. 전방 지역에서의 Duel/Tackle/Interception 참여가 적극적이며, 전술적으로 높은 압박 강도를 유지할 수 있습니다. 또한 T-Fit에는 파울/카드(징계) 리스크(클린플레이)가 함께 반영되므로, 단순 활동량이 아니라 '전술 실행 품질' 관점에서 해석해야 합니다."
        elif t_score >= 50:
            t_analysis = f"T-Fit {t_score:.1f}점은 중상위권 수준의 전술 실행 지표입니다. 전방 수비 참여가 일정 수준 이상이며, 팀 전술에 따라 압박 강도를 조절할 수 있습니다. 다만 팀이 극단적인 하이프레싱을 요구하는 경우에는 역할 정의(압박 트리거/커버 범위) 조정이 필요할 수 있습니다."
        else:
            t_analysis = f"T-Fit {t_score:.1f}점은 평균 이하의 전술 실행 지표입니다. 전방 지역에서의 수비 행동 참여가 낮아, 하이프레싱 성향이 강한 팀에서는 압박 트리거/전환 속도에 대한 적응이 필요합니다. 반대로 블록 수비/역습 중심 팀에서는 역할에 따라 효율적으로 활용될 수 있습니다."
        
        # P-Fit 분석
        p_score = player_hsi['p_fit_score']
        if p_score >= 1.0:
            p_analysis = f"P-Fit {p_score:.2f}점은 뛰어난 여름 적응력을 나타냅니다. 여름철 활동량이 일반 시즌 대비 {(p_score-1)*100:.1f}% 증가하여, K리그 혹서기(6-8월)에도 안정적인 활동량을 유지합니다. 주전 경쟁에서 유리한 체력 조건입니다."
        elif p_score >= 0.9:
            p_analysis = f"P-Fit {p_score:.2f}점은 양호한 수준입니다. 여름철 활동량이 {(1-p_score)*100:.1f}% 감소하지만 경기력에 큰 영향은 없습니다. 다만 8월 밀집 일정 시 로테이션 운영이 권장됩니다."
        else:
            p_analysis = f"P-Fit {p_score:.2f}점은 주의가 필요합니다. 여름철 활동량이 {(1-p_score)*100:.1f}% 감소하여 혹서기 체력 관리에 신경 써야 합니다. 7-8월 원정 경기 시 컨디션 저하 가능성이 있습니다."
        
        # C-Fit 분석
        c_score = player_hsi['c_fit_score']
        if c_score >= 0.95:
            c_analysis = f"C-Fit {c_score:.3f}점은 선수 성장도시 ↔ 분석도시(구단 연고지) 문화적 거리가 낮은 편으로 해석됩니다. 생활/의사소통/조직문화 적응 비용이 상대적으로 낮아, 초기 적응 과정에서 전술 이해와 팀 커뮤니케이션에 빠르게 녹아들 가능성이 큽니다."
        elif c_score >= 0.90:
            c_analysis = f"C-Fit {c_score:.3f}점은 무난한 문화 적합도 수준입니다. 기본 적응은 가능하나, 언어/생활 루틴/커뮤니케이션 지원이 있으면 적응 속도가 빨라질 수 있습니다."
        elif c_score >= 0.85:
            c_analysis = f"C-Fit {c_score:.3f}점은 중간 수준의 문화 적합도입니다. 초반 4~8주 동안 생활/훈련 루틴 적응과 커뮤니케이션(용어/전술 약속) 지원이 성과에 영향을 줄 수 있습니다."
        else:
            c_analysis = f"C-Fit {c_score:.3f}점은 문화적 거리 관점에서 적응 리스크가 큰 편입니다. 초기 정착(주거/식습관/언어) 및 팀 내 커뮤니케이션 지원 없이는 전술 수행이 지연될 수 있으므로, 구단 차원의 온보딩/멘토링 체계를 병행하는 것이 바람직합니다."
        
        # 등급 및 추천
        if adapt_fit_score >= 90:
            grade_text = "S등급 (최상위 적합)"
            recommendation = f"{team_name}의 전술 시스템에 매우 적합한 선수입니다. 즉시 주전 경쟁이 가능하며, 팀 전술 구현에 핵심 역할을 수행할 수 있습니다. 영입을 적극 추천합니다."
        elif adapt_fit_score >= 80:
            grade_text = "A등급 (우수)"
            recommendation = f"{team_name}에 좋은 보강이 될 수 있습니다. 약간의 적응 기간 후 주전 경쟁이 가능하며, 팀 전술에 기여할 수 있습니다. 영입 추천합니다."
        elif adapt_fit_score >= 70:
            grade_text = "B등급 (양호)"
            recommendation = f"{team_name}에서 역할을 수행할 수 있으나, 전술적 보완이 필요합니다. 백업 또는 로테이션 자원으로 적합합니다. 조건부 영입을 권장합니다."
        else:
            grade_text = "C등급 이하 (보통)"
            recommendation = f"{team_name}의 현재 전술 시스템과 다소 맞지 않습니다. 전술 변화 또는 선수 역할 재정의가 필요합니다. 신중한 검토를 권장합니다."
        
        fallback = f"""
A. 선수 프로필 및 HSI 분석

{player_name} 선수의 HSI(Harmonic Synergy Index) 지표를 분석한 결과입니다.

[T-Fit: 전술 적합도]
{t_analysis}

[P-Fit: 환경 적합도]
{p_analysis}

        [C-Fit: 문화 적합도]
{c_analysis}

B. {team_name}과의 전술 시너지 분석

{team_name} 팀의 {pos_group} 포지션 선수들과 비교 분석 결과:

- T-Fit 비교: 선수 {player_hsi['t_fit_score']:.1f} vs 팀 평균 {team_template['t_fit_score']:.1f} (차이: {t_diff:+.1f})
  → {'팀 평균을 상회하는 압박력으로 전술 적합도가 높습니다.' if t_diff > 0 else '팀 평균보다 낮은 압박 참여로 적응 훈련이 필요합니다.'}

- P-Fit 비교: 선수 {player_hsi['p_fit_score']:.2f} vs 팀 평균 {team_template['p_fit_score']:.2f} (차이: {p_diff:+.2f})
  → {'여름 시즌 체력 면에서 팀에 기여할 수 있습니다.' if p_diff >= 0 else '여름 일정 소화에 로테이션 운영이 권장됩니다.'}

        - C-Fit 비교: 선수 {player_hsi['c_fit_score']:.3f} vs 팀 평균 {team_template['c_fit_score']:.3f} (차이: {c_diff:+.3f})
          → {'리그/팀 문화 적응 리스크가 낮은 편입니다.' if c_diff >= 0 else '초기 적응(언어/생활/전술 커뮤니케이션) 지원이 필요합니다.'}

C. 스카우팅 의견 및 영입 제언

[종합 전술 적합도]
{adapt_fit_score:.1f}점 - {grade_text}

[영입 추천 의견]
{recommendation}

---
본 분석은 2024 시즌 K리그 공식 데이터를 기반으로, HSI(Harmonic Synergy Index) 알고리즘을 통해 산출되었습니다.
분석 모델: K-Scout Adapt-Fit AI v1.0 | ANYONE COMPANY
        """
        return fallback

def generate_analysis_summary(player_hsi, team_template):
    """선수의 HSI 점수와 팀 템플릿을 비교하여 분석 코멘트를 생성합니다."""
    strengths, weaknesses = [], []
    
    player_t_fit = player_hsi['t_fit_score']
    team_t_fit = team_template['t_fit_score']
    if player_t_fit > team_t_fit * 1.15:
        strengths.append(f"팀 평균({team_t_fit:.1f})을 상회하는 전방 압박 능력(선수: {player_t_fit:.1f})")
    elif player_t_fit < team_t_fit * 0.85:
        weaknesses.append(f"팀 평균({team_t_fit:.1f})보다 낮은 압박 가담률(선수: {player_t_fit:.1f})")

    player_p_fit = player_hsi['p_fit_score']
    if player_p_fit < 0.9:
        weaknesses.append(f"여름철 활동량 유지율({player_p_fit:.2f})이 다소 낮음")
    elif abs(player_p_fit - 1.0) < 0.1:
        strengths.append(f"혹서기에도 안정적인 활동량 유지({player_p_fit:.2f})")

    player_c_fit = player_hsi['c_fit_score']
    if player_c_fit < 0.85:
        weaknesses.append(f"문화 적합도(C-Fit {player_c_fit:.3f})가 낮아 초기 적응 리스크가 있을 수 있음")
    elif player_c_fit >= 0.93:
        strengths.append(f"문화 적응(C-Fit {player_c_fit:.3f}) 측면의 리스크가 낮은 편")

    return strengths, weaknesses

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import navy, deepskyblue
from reportlab.lib.units import inch

def create_pdf(file_path, player_hsi, team_template, chart_path, ai_text, team_name, player_name, player_pos):
    """Creates a PDF report with the analysis."""
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    story = []
    
    font_candidates = [
        ('AppleGothic', '/System/Library/Fonts/AppleGothic.ttf'),
        ('AppleGothic', '/System/Library/Fonts/Supplemental/AppleGothic.ttf'),
        ('NotoSansGothic', '/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf'),
    ]
    korean_font = 'Helvetica'
    for font_name, font_path in font_candidates:
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                korean_font = font_name
                break
            except Exception:
                continue
    if korean_font == 'Helvetica':
        st.warning("한글 폰트를 찾지 못해 기본 폰트로 생성합니다.")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleStyle', fontName=korean_font, fontSize=24, leading=28, textColor=navy))
    styles.add(ParagraphStyle(name='HeaderStyle', fontName=korean_font, fontSize=16, leading=20, textColor=deepskyblue))
    styles.add(ParagraphStyle(name='BodyStyle', fontName=korean_font, fontSize=10, leading=14))

    story.append(Paragraph(f"K-Scout AI: {player_name} 선수 분석 보고서", styles['TitleStyle']))
    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph(f"분석 대상 팀: {team_name}", styles['BodyStyle']))
    story.append(Spacer(1, 0.5*inch))

    story.append(Paragraph("선수 프로필 및 HSI 레이더 차트", styles['HeaderStyle']))
    story.append(Spacer(1, 0.1*inch))
    
    if chart_path and Path(chart_path).exists():
        try:
            img = Image(chart_path, width=4*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.25*inch))
        except Exception as e:
            story.append(Paragraph(f"(차트 이미지 로드 실패: {e})", styles['BodyStyle']))
            story.append(Spacer(1, 0.25*inch))
    else:
        story.append(Paragraph("(차트 데이터 시각화 준비 중 또는 생성 실패)", styles['BodyStyle']))
        story.append(Spacer(1, 0.25*inch))
    
    story.append(Paragraph("AI 기반 심층 분석", styles['HeaderStyle']))
    story.append(Spacer(1, 0.1*inch))
    
    # Markdown(헤딩) 일부를 PDF에서 보기 좋게 변환
    import re
    from xml.sax.saxutils import escape

    safe_text = escape(ai_text or "")
    # Convert markdown headings (#, ##, ###...) to bold lines
    safe_text = re.sub(r'(?m)^#{1,6}\s*(.+)$', r'<b>\g<1></b>', safe_text)
    # Convert markdown bold **text** to <b>text</b>
    safe_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\g<1></b>', safe_text)
    formatted_ai_text = safe_text.replace('\n', '<br/>')
    story.append(Paragraph(formatted_ai_text, styles['BodyStyle']))
    
    doc.build(story)

# ============================================================
# 사이드바
# ============================================================
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1 style="font-size: 1.5rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #FE3D67, #FF7031); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        K-Scout
    </h1>
    <p style="color: #FF7031; font-size: 0.9rem; margin: 0.2rem 0 0 0; font-weight: 500;">Adapt-Fit AI System</p>
    <p style="color: #872B95; font-size: 0.7rem; margin: 0.1rem 0 0 0;">ANYONE COMPANY</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

# 팀 선택
team_list = list(templates.keys())
client_team = st.sidebar.selectbox("분석 대상 구단", team_list)

# 선수 선택
foreign_player_names = foreigners_df['player_name_ko'].unique()
player_list = hsi_df[hsi_df['player_name_ko'].isin(foreign_player_names)]['player_name_ko'].unique()
selected_player_name = st.sidebar.selectbox("대상 외국인 선수", player_list)

# 선택 선수의 구단(동명이인 구분/매핑 UI용)
selected_player_club = ""
try:
    if "club_name_ko" in foreigners_df.columns:
        clubs = foreigners_df.loc[foreigners_df["player_name_ko"] == selected_player_name, "club_name_ko"]
        if not clubs.empty:
            selected_player_club = str(clubs.mode().iloc[0]).strip()
except Exception:
    selected_player_club = ""

st.sidebar.markdown("---")

# 지표 설명
with st.sidebar.expander("HSI 지표 가이드"):
    st.markdown("""
    점수 기반 전술 적합도 분석을 위해 다음 세 가지 지표를 사용합니다.
    
    1. T-Fit (Tactical): 전방 수비 및 압박 실행 능력과 카드/파울 리스크를 결합한 지표입니다.
    2. P-Fit (Physical): 혹서기(여름) 활동량 유지 능력을 나타내는 지표입니다.
    3. C-Fit (Cultural): 출생/성장 배경과 구단 연고지 간의 문화적 거리를 산출한 적응 지표입니다.
    """)
    st.caption("데이터 자동 매핑 상태: 정상")

# 고급 매핑 설정 (기본적으로 숨김)
with st.sidebar.expander("데이터 매핑 상세 설정"):
    st.caption("필요 시 선수와 구단의 지역 코드를 수동으로 조정할 수 있습니다.")
    
    # --- 선수 매핑 UI ---
    st.markdown("---")
    st.markdown("선수 성장 도시 정보")
    try:
        pmap = pd.read_csv(PLAYER_CITY_MAP_PATH)
        prow_df = pmap[pmap["player_name_ko"] == selected_player_name]
        if "club_name_ko" in pmap.columns and selected_player_club:
            hit2 = prow_df[prow_df["club_name_ko"] == selected_player_club]
            if not hit2.empty:
                prow_df = hit2
        prow_val = prow_df.head(1).iloc[0] if not prow_df.empty else {}
    except Exception:
        pmap = pd.DataFrame(columns=["player_name_ko", "club_name_ko", "home_country_alpha", "home_loc_code", "home_city_label"])
        prow_val = {}

    p_country_in = st.text_input(
        "COUNTRY_ALPHA (선수)",
        value=str(prow_val.get("home_country_alpha", "")).strip(),
        key="p_home_country_alpha_in",
    )
    p_loc_in = st.text_input(
        "LOC_CODE (선수)",
        value=str(prow_val.get("home_loc_code", "")).strip(),
        key="p_home_loc_code_in",
    )
    p_label_in = st.text_input(
        "도시 라벨 (선수)",
        value=str(prow_val.get("home_city_label", "")).strip(),
        key="p_home_city_label_in",
    )
    if st.button("선수 성장도시 저장", key="save_player_city_btn"):
        pmap = pmap.copy()
        mask = (pmap["player_name_ko"] == selected_player_name)
        if "club_name_ko" in pmap.columns and selected_player_club:
            mask = mask & (pmap["club_name_ko"] == selected_player_club)
        
        if mask.any():
            pmap.loc[mask, ["home_country_alpha", "home_loc_code", "home_city_label"]] = [
                p_country_in.strip().upper(), p_loc_in.strip(), p_label_in.strip()
            ]
        else:
            new_row = {
                "player_name_ko": selected_player_name,
                "club_name_ko": selected_player_club,
                "home_country_alpha": p_country_in.strip().upper(),
                "home_loc_code": p_loc_in.strip(),
                "home_city_label": p_label_in.strip()
            }
            pmap = pd.concat([pmap, pd.DataFrame([new_row])], ignore_index=True)
        pmap.to_csv(PLAYER_CITY_MAP_PATH, index=False)
        st.success("저장 완료!")
        st.experimental_rerun()

    # --- 팀 매핑 UI ---
    st.markdown("---")
    st.markdown("분석 도시 정보 (구단 연고지)")
    try:
        tmap = pd.read_csv(TEAM_CITY_MAP_PATH)
        trow_df = tmap[tmap["team_name_ko"] == client_team]
        trow_val = trow_df.head(1).iloc[0] if not trow_df.empty else {}
    except Exception:
        tmap = pd.DataFrame(columns=["team_name_ko", "host_country_alpha", "host_loc_code", "host_city_label"])
        trow_val = {}

    t_country_in = st.text_input(
        "COUNTRY_ALPHA (팀)",
        value=str(trow_val.get("host_country_alpha", "KOR")).strip().upper(),
        key="t_host_country_alpha_in",
    )
    t_loc_in = st.text_input(
        "LOC_CODE (팀)",
        value=str(trow_val.get("host_loc_code", "")).strip(),
        key="t_host_loc_code_in",
    )
    t_label_in = st.text_input(
        "도시 라벨 (팀)",
        value=str(trow_val.get("host_city_label", "")).strip(),
        key="t_host_city_label_in",
    )
    if st.button("분석도시 저장", key="save_team_city_btn"):
        tmap = tmap.copy()
        mask = tmap["team_name_ko"] == client_team
        if mask.any():
            tmap.loc[mask, ["host_country_alpha", "host_loc_code", "host_city_label"]] = [
                t_country_in.strip().upper(), t_loc_in.strip(), t_label_in.strip()
            ]
        else:
            new_row = {"team_name_ko": client_team, "host_country_alpha": t_country_in.strip().upper(), "host_loc_code": t_loc_in.strip(), "host_city_label": t_label_in.strip()}
            tmap = pd.concat([tmap, pd.DataFrame([new_row])], ignore_index=True)
        tmap.to_csv(TEAM_CITY_MAP_PATH, index=False)
        st.success("저장 완료!")
        st.experimental_rerun()

# 매핑 진행률 체크 (고급 설정 하단)
with st.sidebar.expander("데이터 매핑 현황"):
    try:
        p_all = pd.read_csv(PLAYER_CITY_MAP_PATH)
        t_all = pd.read_csv(TEAM_CITY_MAP_PATH)
        def _f(x): return bool(str(x).strip()) and str(x).lower() != "nan"
        st.write(f"선수 데이터: {p_all['home_loc_code'].apply(_f).sum()}/{len(p_all)}")
        st.write(f"구단 데이터: {t_all['host_loc_code'].apply(_f).sum()}/{len(t_all)}")
    except: pass

# ============================================================
# 메인 컨텐츠
# ============================================================

# PDF 버튼
pdf_button = st.sidebar.button("PDF 분석 보고서 생성")

st.sidebar.markdown("---")

# 헤더
st.title("K-Scout Adapt-Fit AI")
st.caption("2024 K리그 데이터 기반 전술 적합도 정량 분석 시스템")
st.markdown("---")

# 선택된 선수 데이터
selected_player_hsi = hsi_df[hsi_df['player_name_ko'] == selected_player_name].iloc[0]
player_pos = foreigners_df.loc[foreigners_df['player_name_ko'] == selected_player_name, 'main_position'].iloc[0]
pos_group = group_position(player_pos)

team_templates_by_pos = templates[client_team]
team_template_data = team_templates_by_pos.get(pos_group, team_templates_by_pos.get('ETC'))

if not team_template_data:
    st.error(f"{client_team}에는 {pos_group} 포지션 그룹의 데이터가 부족합니다.")
    st.stop()

# --- WVS(도시) 기반 C-Fit 계산 (선수 성장도시 ↔ 분석도시) ---
c_fit_dynamic, player_city_label, host_city_label, cfit_reason, cfit_meta = compute_city_based_c_fit(
    selected_player_name, client_team, selected_player_club or None
)

player_hsi_for_score = selected_player_hsi.copy()
if c_fit_dynamic is not None:
    player_hsi_for_score["c_fit_score"] = float(c_fit_dynamic)

team_template_for_score = dict(team_template_data)
# C-Fit은 '분석도시(팀 연고지)' 기준으로 계산되므로, 팀의 목표값은 1.0(자기 도시와의 거리=0)으로 둡니다.
team_template_for_score["c_fit_score"] = 1.0

adapt_fit_score = calculate_adapt_fit_score(player_hsi_for_score, team_template_for_score, pos_group)
grade, grade_color, grade_desc = get_score_grade(adapt_fit_score)
strengths, weaknesses = generate_analysis_summary(player_hsi_for_score, team_template_for_score)

# 탭 구성
tab1, tab2, tab3 = st.tabs(["선수 분석", "팀 추천 랭킹", "상세 비교"])

with tab1:
    col1, col2 = st.columns([1, 1.5])

    with col1:
        # 선수 카드
        st.subheader(f"{selected_player_name}")
        st.caption(f"{get_position_korean(player_pos)} ({pos_group})")

        # 선수 프로필
        try:
            if PLAYER_PROFILE_PATH.exists():
                prof = pd.read_csv(PLAYER_PROFILE_PATH)
                if not prof.empty:
                    prof["player_id"] = pd.to_numeric(prof.get("player_id"), errors="coerce")
                    pid = float(selected_player_hsi.get("player_id")) if "player_id" in selected_player_hsi else None
                    hit = pd.DataFrame()
                    if pid is not None and "player_id" in prof.columns:
                        hit = prof[prof["player_id"] == pid]
                    if hit.empty and "player_name_ko" in prof.columns:
                        hit = prof[prof["player_name_ko"] == selected_player_name]
                    if not hit.empty:
                        prow = hit.iloc[0]
                        en_name = str(prow.get("player_name_en_full", "")).strip()
                        if en_name:
                            st.caption(f"영문 성명: {en_name}")

                        nat = str(prow.get("nationality", "")).strip()
                        if nat and nat.lower() != "foreign":
                            st.caption(f"국적: {nat}")
        except Exception:
            pass

        st.markdown("---")

        # 적합도 점수 표시
        st.metric(
            label="전술 적합도 종합 점수",
            value=f"{adapt_fit_score:.1f}점",
            delta=f"등급: {grade}",
        )
        st.info(f"{grade_desc}")

        st.markdown("---")

        # HSI 세부 점수
        st.markdown("### HSI 세부 지표")

        st.write(
            f"T-Fit (Tactical): {player_hsi_for_score['t_fit_score']:.1f} "
            f"(팀 평균: {team_template_for_score['t_fit_score']:.1f})"
        )
        st.write(
            f"P-Fit (Physical): {player_hsi_for_score['p_fit_score']:.2f} "
            f"(팀 평균: {team_template_for_score['p_fit_score']:.2f})"
        )
        st.write(
            f"C-Fit (Cultural): {player_hsi_for_score['c_fit_score']:.3f} "
            f"(팀 기준: {team_template_for_score['c_fit_score']:.3f})"
        )

        if c_fit_dynamic is None and cfit_reason:
            st.caption(f"C-Fit(도시) 계산 불가: {cfit_reason} (임시값 사용)")
        elif c_fit_dynamic is not None:
            from_city = player_city_label if player_city_label else "선수 성장도시(미상)"
            to_city = host_city_label if host_city_label else "분석도시(미상)"
            st.caption(f"도시 기반 C-Fit: {from_city} → {to_city}")
            try:
                if isinstance(cfit_meta, dict):
                    def _src_ko(v):
                        m = {"city": "도시", "country": "국가", "global": "글로벌"}
                        return m.get(str(v), str(v))
                    hs = _src_ko(cfit_meta.get("home_source"))
                    ts = _src_ko(cfit_meta.get("host_source"))
                    st.caption(f"C-Fit 계산 단위: 선수={hs} / 분석={ts}")
            except Exception:
                pass

    with col2:
        # 레이더 차트
        st.markdown(f"### {selected_player_name} vs {client_team} 비교")

        categories = ["T-Fit (전술)", "P-Fit (환경)", "C-Fit (문화)"]
        player_r = [
            player_hsi_for_score["t_fit_score"],
            player_hsi_for_score["p_fit_score"] * 100,
            player_hsi_for_score["c_fit_score"] * 100,
        ]
        team_r = [
            team_template_for_score["t_fit_score"],
            team_template_for_score["p_fit_score"] * 100,
            team_template_for_score["c_fit_score"] * 100,
        ]
        max_r = max(player_r + team_r + [120])

        fig = go.Figure()
        # 선수 프로필 - 브랜드 핑크-오렌지
        fig.add_trace(
            go.Scatterpolar(
                r=player_r + [player_r[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=f"{selected_player_name}",
                line=dict(color="#FE3D67", width=3),
                fillcolor="rgba(254, 61, 103, 0.3)",
            )
        )
        # 팀 템플릿 - 브랜드 퍼플
        fig.add_trace(
            go.Scatterpolar(
                r=team_r + [team_r[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=f"{client_team} 템플릿",
                line=dict(color="#872B95", width=2),
                fillcolor="rgba(135, 43, 149, 0.2)",
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_r],
                    tickfont=dict(color="#FF7031", size=10),  # 브랜드 오렌지
                    gridcolor="rgba(254, 61, 103, 0.2)",  # 브랜드 핑크
                ),
                angularaxis=dict(
                    tickfont=dict(color="#ffffff", size=12),
                    gridcolor="rgba(254, 61, 103, 0.2)",  # 브랜드 핑크
                ),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5,
                font=dict(color="#ffffff"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=80, b=40, l=40, r=40),
            height=400,
        )
        st.plotly_chart(fig)

        # 강점/약점 분석
        st.markdown("### AI 분석 요약")

        st.markdown("전술적 강점")
        if strengths:
            for s in strengths:
                st.success(f"• {s}")
        else:
            st.markdown("*특이 강점 항목 없음*")

        st.markdown("보완 및 검토 사항")
        if weaknesses:
            for w in weaknesses:
                st.warning(f"• {w}")
        else:
            st.markdown("*특이 보완 사항 없음*")

with tab2:
    st.markdown(f"### {client_team} 추천 외국인 선수 Top 10")
    st.markdown(f"*{client_team}의 전술 스타일에 적합한 외국인 선수 순위입니다.*")
    
    all_scores = get_all_player_scores(client_team, templates, hsi_df, foreigners_df)
    
    # 랭킹 테이블로 표시
    ranking_data = []
    for i, player in enumerate(all_scores[:10]):
        rank_label = f"{i+1}위"
        is_selected = " (분석 대상)" if player['name'] == selected_player_name else ""
        ranking_data.append({
            '순위': rank_label,
            '선수명': f"{player['name']}{is_selected}",
            '포지션': get_position_korean(player['position']),
            '적합도': f"{player['score']:.1f}점",
            '등급': player['grade']
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    st.table(ranking_df)

with tab3:
    st.markdown("### 전체 선수 상세 데이터")
    
    all_scores = get_all_player_scores(client_team, templates, hsi_df, foreigners_df)
    
    df_display = pd.DataFrame([{
        '순위': i+1,
        '선수명': p['name'],
        '포지션': get_position_korean(p['position']),
        '적합도': f"{p['score']:.1f}",
        '등급': p['grade'],
        'T-Fit': f"{p['t_fit']:.1f}",
        'P-Fit': f"{p['p_fit']:.2f}",
        'C-Fit': f"{p['c_fit']:.3f}"
    } for i, p in enumerate(all_scores)])
    
    st.dataframe(df_display)

# ============================================================
# PDF 생성
# ============================================================
if pdf_button:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요.")
    else:
        with st.spinner("AI 분석 보고서를 생성하는 중..."):
            output_dir = Path("output")
            reports_dir = output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            chart_path = reports_dir / f"report_{selected_player_name}_{client_team}.png"
            
            # PDF용 차트 생성 (흰 배경용 - 검정 글씨)
            pdf_fig = go.Figure()
            pdf_fig.add_trace(go.Scatterpolar(
                r=player_r + [player_r[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=f'{selected_player_name}',
                line=dict(color='#FE3D67', width=3),
                fillcolor='rgba(254, 61, 103, 0.3)'
            ))
            pdf_fig.add_trace(go.Scatterpolar(
                r=team_r + [team_r[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=f'{client_team} 템플릿',
                line=dict(color='#872B95', width=2),
                fillcolor='rgba(135, 43, 149, 0.2)'
            ))
            pdf_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max_r],
                        tickfont=dict(color='#333333', size=11),  # 검정 글씨
                        gridcolor='rgba(100, 100, 100, 0.3)'
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='#000000', size=13, family='Arial Black'),  # 굵은 검정 글씨
                        gridcolor='rgba(100, 100, 100, 0.3)'
                    ),
                    bgcolor='rgba(250, 250, 250, 1)'  # 연한 회색 배경
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.1,
                    xanchor="center",
                    x=0.5,
                    font=dict(color='#000000', size=12)  # 검정 글씨
                ),
                paper_bgcolor='#ffffff',  # 흰 배경
                plot_bgcolor='#ffffff',
                margin=dict(t=80, b=40, l=60, r=60),
                height=400,
                title=dict(
                    text=f'{selected_player_name} vs {client_team} HSI 비교',
                    font=dict(color='#000000', size=16),
                    x=0.5
                )
            )
            
            # 이미지 저장 시도 (kaleido 충돌 대비)
            try:
                import kaleido
                pdf_fig.write_image(str(chart_path), engine="kaleido")
            except Exception as e:
                st.warning(f"차트 이미지 생성 중 오류가 발생했습니다: {e}. 이미지 없이 보고서를 생성합니다.")
                if chart_path.exists():
                    chart_path.unlink() # 실패한 파일이 있으면 삭제

            ai_analysis_text = get_ai_analysis_for_pdf(
                api_key, 
                player_hsi_for_score, 
                team_template_for_score, 
                client_team, 
                selected_player_name, 
                pos_group, 
                adapt_fit_score
            )

            # ============================================================
            # C-Fit 산출 주의사항(공모전 제출 버전) - PDF에 명시
            # ============================================================
            try:
                def _src_ko(v):
                    m = {"city": "도시", "country": "국가", "global": "글로벌"}
                    return m.get(str(v), str(v))

                if isinstance(cfit_meta, dict):
                    hs = _src_ko(cfit_meta.get("home_source"))
                    ts = _src_ko(cfit_meta.get("host_source"))
                    unit_line = f"- C-Fit 계산 단위: 선수={hs} / 분석={ts}"
                else:
                    unit_line = "- C-Fit 계산 단위: 확인 불가"

                if c_fit_dynamic is None and cfit_reason:
                    status_line = f"- 상태: C-Fit(도시) 계산 불가 → 임시값 사용 ({cfit_reason})"
                else:
                    status_line = "- 상태: C-Fit 계산 성공"

                cfit_disclosure = (
                    "## C-Fit 산출 관련 안내\n"
                    f"{unit_line}\n"
                    f"{status_line}\n"
                    "- WVS 데이터 매핑 정책에 따라, 도시 코드가 미비한 경우 국가 단위 데이터로 계산(fallback)되었습니다.\n"
                    "- 일부 국가는 데이터 가용성에 따라 글로벌 평균 벡터를 사용하였습니다.\n"
                )
                ai_analysis_text = f"{cfit_disclosure}\n\n{ai_analysis_text}"
            except Exception:
                pass

            pdf_path = reports_dir / f"K-Scout_Report_{selected_player_name}_{client_team}.pdf"
            create_pdf(
                file_path=str(pdf_path), 
                player_hsi=player_hsi_for_score, 
                team_template=team_template_for_score, 
                chart_path=str(chart_path), 
                ai_text=ai_analysis_text, 
                team_name=client_team, 
                player_name=selected_player_name, 
                player_pos=player_pos
            )

            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            
            st.success("보고서 생성이 완료되었습니다.")
            st.markdown(f"""
            <a href="data:application/octet-stream;base64,{base64_pdf}" 
               download="{pdf_path.name}"
               style="display: inline-block; background: linear-gradient(135deg, #6366f1, #8b5cf6); 
                      color: white; padding: 0.75rem 1.5rem; border-radius: 10px; 
                      text-decoration: none; font-weight: 600; margin-top: 1rem;">
                PDF 보고서 다운로드
            </a>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1.5rem; margin-top: 2rem;">
    <p style="color: #FE3D67; font-weight: 600; margin: 0;">K-Scout Adapt-Fit AI • MVP Version</p>
    <p style="color: #872B95; font-size: 0.75rem; margin: 0.5rem 0;">2024 K리그 데이터 기반 전술 적합도 분석 시스템</p>
    <p style="color: #FF7031; font-size: 0.625rem; margin: 0;">© 2024 ANYONE COMPANY. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
