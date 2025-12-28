import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
import base64
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Gemini AI 설정
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

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

# Gemini API 설정
if GENAI_AVAILABLE:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)

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
FOREIGN_PLAYERS_EXTENDED_PATH = Path("data/raw/foreign_birth_city_2024.csv")


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

def setup_design_system():
    # 1. Theme State
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'Dark'
    
    theme_mode = st.session_state.theme_mode

    # 2. Color Palette (Pro Scout Dashboard)
    if theme_mode == 'Dark':
        bg_color = "#09090B"             # Main Background
        sidebar_bg = "#121214"           # Sidebar Background
        card_bg = "#18181B"              # Component Background
        
        text_primary = "#FFFFFF"         # Pure White
        text_secondary = "#A1A1AA"       # Zinc-400
        text_tertiary = "#71717A"        # Zinc-500
        
        border_color = "#27272A"         # Zinc-800
        input_bg = "#27272A"
        
        accent_color = "#FE3D67"         # K-League/Brand Red
        hover_bg = "#27272A"
        
        metric_color = "#FFFFFF"
        tick_color = "#71717A"
        grid_color = "#27272A"
        polar_bgcolor = "rgba(0,0,0,0)"
    else:
        bg_color = "#F4F4F5"             # Zinc-100
        sidebar_bg = "#FFFFFF"           # White
        card_bg = "#FFFFFF"
        
        text_primary = "#09090B"
        text_secondary = "#71717A"
        text_tertiary = "#A1A1AA"
        
        border_color = "#E4E4E7"         # Zinc-200
        input_bg = "#F4F4F5"
        
        accent_color = "#FE3D67"
        hover_bg = "#F4F4F5"
        
        metric_color = "#09090B"
        tick_color = "#71717A"
        grid_color = "#E4E4E7"
        polar_bgcolor = "rgba(255,255,255,0.8)"

    # 3. CSS Injection
    css = f"""
    <style>
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");
    
    html, body, [class*="css"] {{
        font-family: 'Pretendard', sans-serif !important;
    }}
    
    /* --- LAYOUT: FIXED 390px SIDEBAR --- */
    [data-testid="stSidebar"] {{
        min-width: 390px !important;
        max-width: 390px !important;
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color} !important;
    }}

    /* Sidebar 내부 여백(좌우) 최적화: 그리드가 "꽉 차" 보이도록 */
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        padding-left: 14px !important;
        padding-right: 14px !important;
    }}
    
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_primary} !important;
    }}
    
    /* --- SIDEBAR: TEAM SELECTOR (GRID CARD v9 - true full width + perfect centering) --- */
    /* 위젯 라벨(예: 'Select Club')은 숨기고, 옵션 카드(label)만 스타일링 */
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {{
        display: none !important;
    }}

    /* 라디오/컨테이너 자체가 폭 100%를 가지도록 강제 (가로폭 꽉 차 보이게) */
    [data-testid="stSidebar"] .element-container:has(.stRadio),
    [data-testid="stSidebar"] .stRadio {{
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }}

    /* 라디오 그룹 컨테이너 */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px !important;
        padding: 0 !important;
        margin-top: 4px !important;
        width: 100% !important;
        max-width: 100% !important;
        justify-items: stretch !important;
        box-sizing: border-box !important;
    }}

    /* 홀수 개일 때 마지막 카드가 반쪽으로 남지 않도록 마지막 옵션(label)을 2칸(span) */
    /* last-child가 아닌 last-of-type 기반으로 더 안전하게 매칭 */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:nth-last-of-type(1):nth-child(odd) {{
        /* 마지막(홀수) 항목은 왼쪽 1열에 고정 (2열은 비워둠) */
        grid-column: 1 / 2 !important;
        justify-self: stretch !important;
    }}
    /* 라디오 전체 래퍼도 폭 100% */
    [data-testid="stSidebar"] .stRadio {{
        width: 100% !important;
    }}
    
    /* 옵션 카드(label): radiogroup 내부만 타겟 */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
        padding: 0 8px !important;
        margin: 0 !important;
        height: 50px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        box-shadow: none !important;
        position: relative !important;
        overflow: hidden !important;
        text-align: center !important;
        box-sizing: border-box !important;
        min-width: 0 !important;
    }}
    
    /* 내부 라디오 컨트롤(동그라미) 숨기기 */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label > div:first-child {{
        display: none !important;
        width: 0 !important;
        margin: 0 !important;
    }}
    
    /* 텍스트 컨테이너 */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label > div:last-child {{
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        height: 100% !important;
    }}

    /* Streamlit 마크다운 래퍼가 기본 패딩/정렬을 갖는 경우가 있어 완전 리셋 */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label div[data-testid="stMarkdownContainer"] {{
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
    }}
    
    /* 텍스트 */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label p {{
        font-size: 0.92rem !important;
        font-weight: 400 !important;  /* 기본은 가볍게 */
        color: {text_secondary} !important;
        margin: 0 !important;
        text-align: center !important;
        line-height: 1.15 !important;
        padding: 0 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        width: 100% !important;
        letter-spacing: -0.02em !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 100% !important;
        -webkit-font-smoothing: antialiased !important;
        text-rendering: optimizeLegibility !important;
    }}
    
    /* Hover */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover {{
        border-color: {text_secondary} !important;
        background-color: {hover_bg} !important;
    }}
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover p {{
        color: {text_primary} !important;
    }}
    
    /* Selected */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:has(input:checked) {{
        background-color: {input_bg} !important;
        border-color: {text_primary} !important;
        box-shadow: inset 0 0 0 1.5px {text_primary} !important;
    }}
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:has(input:checked) p {{
        color: {text_primary} !important;
        font-weight: 500 !important;  /* 선택 시만 살짝 */
    }}
    
    /* --- COMPONENT: BUTTONS --- */
    .stButton > button {{
        width: 100% !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
        border: 1px solid {border_color} !important;
        background-color: {input_bg} !important;
        color: {text_primary} !important;
        transition: all 0.2s !important;
    }}
    .stButton > button:hover {{
        border-color: {text_secondary} !important;
        background-color: {border_color} !important;
    }}
    
    /* Primary Action (PDF) - Minimal Style */
    button[kind="primary"] {{
        background-color: {text_primary} !important;
        border: 1px solid {text_primary} !important;
        color: {bg_color} !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }}
    button[kind="primary"]:hover {{
        background-color: {text_secondary} !important;
        border-color: {text_secondary} !important;
        color: {bg_color} !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }}
    button[kind="primary"] p {{
        color: {bg_color} !important;
    }}
    button[kind="primary"]:active {{
        transform: scale(0.98) !important;
    }}
    
    /* --- COMPONENT: INPUTS --- */
    [data-baseweb="select"] > div, .stTextInput > div > div {{
        background-color: {input_bg} !important;
        border-color: {border_color} !important;
        color: {text_primary} !important;
        border-radius: 8px !important;
    }}
    
    /* --- TABS & OTHERS --- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: none !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent; 
        border: 1px solid {border_color}; 
        border-radius: 99px;
        padding: 6px 16px;
        color: {text_secondary};
        height: auto !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {text_primary} !important;
        color: {bg_color} !important;
        border-color: {text_primary} !important;
    }}
    /* Remove Highlight Bar */
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
    }}
    
    /* Utils */
    hr {{ border-color: {border_color} !important; margin: 1.5rem 0 !important; }}
    .metric-card, .stPlotlyChart {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin-bottom: 24px !important;
    }}

    /* --- TAB2: TEAM RANKING (Scout-friendly UI) --- */
    .rank-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0 18px 0;
    }}
    @media (max-width: 1200px) {{
        .rank-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 820px) {{
        .rank-grid {{ grid-template-columns: 1fr; }}
    }}
    .rank-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 16px 16px 14px 16px;
    }}
    .rank-card.is-selected {{
        border-color: {text_primary};
        box-shadow: inset 0 0 0 1px {text_primary};
    }}
    .rank-head {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }}
    .rank-badge {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {text_primary};
        letter-spacing: -0.02em;
    }}
    .grade-chip {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid {border_color};
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 1;
        color: {text_secondary};
        background: {input_bg};
        white-space: nowrap;
    }}
    .rank-name {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {text_primary};
        letter-spacing: -0.02em;
        margin: 0;
    }}
    .rank-meta {{
        font-size: 0.82rem;
        color: {text_secondary};
        margin-top: 2px;
    }}
    .score-line {{
        margin-top: 12px;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 10px;
    }}
    .score-value {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {text_primary};
        letter-spacing: -0.03em;
        line-height: 1;
    }}
    .score-unit {{
        font-size: 0.85rem;
        font-weight: 500;
        color: {text_tertiary};
        margin-left: 4px;
    }}
    .score-bar {{
        margin-top: 10px;
        height: 6px;
        width: 100%;
        border-radius: 999px;
        background: {border_color};
        overflow: hidden;
    }}
    .score-bar > span {{
        display: block;
        height: 100%;
        border-radius: 999px;
    }}
    .rank-list {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 8px 0;
        margin-top: 10px;
    }}
    .rank-row {{
        display: grid;
        grid-template-columns: 52px 1fr 120px 72px;
        gap: 10px;
        align-items: center;
        padding: 10px 14px;
    }}
    .rank-row + .rank-row {{
        border-top: 1px solid {border_color};
    }}
    .rank-row:hover {{
        background: {hover_bg};
    }}
    .rank-rank {{
        font-size: 0.85rem;
        font-weight: 700;
        color: {text_primary};
        text-align: center;
    }}
    .rank-player {{
        display: flex;
        flex-direction: column;
        min-width: 0;
    }}
    .rank-player .nm {{
        font-size: 0.92rem;
        font-weight: 600;
        color: {text_primary};
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .rank-player .pos {{
        font-size: 0.8rem;
        color: {text_secondary};
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .rank-score {{
        font-size: 0.9rem;
        font-weight: 700;
        color: {text_primary};
        text-align: right;
        white-space: nowrap;
    }}
    .rank-grade {{
        text-align: right;
    }}
    .rank-row.is-selected {{
        background: {input_bg};
        box-shadow: inset 3px 0 0 0 {accent_color};
    }}
    .rank-row.is-selected .rank-player .nm {{
        color: {accent_color} !important;
    }}

    /* --- TAB3: DETAIL COMPARE (Scout-friendly list) --- */
    .compare-list {{
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 10px;
    }}
    .compare-row {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 14px 14px 12px 14px;
    }}
    .compare-row:hover {{
        background: {hover_bg};
    }}
    .compare-row.is-selected {{
        background: {input_bg};
        box-shadow: inset 3px 0 0 0 {accent_color};
        border-color: {border_color};
    }}
    .compare-row.is-selected .compare-center .nm {{
        color: {accent_color};
    }}
    .compare-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }}
    .compare-badge {{
        width: 52px;
        height: 28px;
        border-radius: 999px;
        border: 1px solid {border_color};
        background: {card_bg};
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.82rem;
        font-weight: 800;
        color: {text_primary};
        flex: 0 0 auto;
    }}
    .compare-center {{
        flex: 1;
        min-width: 0;
    }}
    .compare-center .nm {{
        font-size: 0.95rem;
        font-weight: 700;
        color: {text_primary};
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        letter-spacing: -0.02em;
    }}
    .compare-center .pos {{
        font-size: 0.8rem;
        color: {text_secondary};
        margin-top: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .compare-right {{
        text-align: right;
        white-space: nowrap;
        flex: 0 0 auto;
    }}
    .compare-right .score {{
        font-size: 1.2rem;
        font-weight: 800;
        color: {text_primary};
        letter-spacing: -0.03em;
        line-height: 1;
    }}
    .compare-right .score span {{
        font-size: 0.85rem;
        font-weight: 500;
        color: {text_tertiary};
        margin-left: 4px;
    }}
    .compare-metrics {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-top: 12px;
    }}
    @media (max-width: 900px) {{
        .compare-metrics {{ grid-template-columns: 1fr; }}
    }}
    .metric-mini .lbl {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 10px;
        font-size: 0.75rem;
        color: {text_secondary};
        margin-bottom: 6px;
    }}
    .metric-mini .lbl strong {{
        color: {text_primary};
        font-weight: 600;
    }}
    .metric-mini .bar {{
        height: 6px;
        width: 100%;
        border-radius: 999px;
        background: {border_color};
        overflow: hidden;
    }}
    .metric-mini .bar > span {{
        display: block;
        height: 100%;
        border-radius: 999px;
    }}

    header, footer {{ display: none !important; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return tick_color, grid_color, polar_bgcolor, text_primary, text_secondary, text_tertiary, bg_color, card_bg, border_color, input_bg, metric_color, accent_color

tick_color, grid_color, polar_bgcolor, text_primary, text_secondary, text_tertiary, bg_color, card_bg, border_color, input_bg, metric_color, accent_color = setup_design_system()
text_color = text_primary

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
    
    # 이름 중복 제거: 같은 한글명(동명이인/데이터 중복)으로 여러 번 등장하는 경우,
    # 스카우터 UI에서는 "이름 1개 = 1개 엔트리"가 되도록 최고 점수 1개만 남깁니다.
    best_by_name = {}
    for s in scores:
        nm = str(s.get("name", "")).strip()
        if not nm:
            continue
        if nm not in best_by_name or float(s.get("score", 0)) > float(best_by_name[nm].get("score", 0)):
            best_by_name[nm] = s

    return sorted(best_by_name.values(), key=lambda x: x['score'], reverse=True)

def get_gemini_model():
    """Gemini 무료 모델 자동 선택"""
    if not GENAI_AVAILABLE or not genai:
        return None
    try:
        available_models = [m.name for m in genai.list_models() 
                          if 'generateContent' in m.supported_generation_methods]
        
        # 우선순위: gemini-1.5-flash (무료, 빠름) -> gemini-1.5-pro -> gemini-pro
        for model_name in ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]:
            if model_name in available_models:
                return genai.GenerativeModel(model_name)
        
        # 위에 없으면 첫 번째 사용 가능한 모델
        if available_models:
            return genai.GenerativeModel(available_models[0])
    except Exception as e:
        print(f"Gemini 모델 로드 실패: {e}")
    return None

def load_player_insights():
    """player_insights.json 로드"""
    try:
        insights_path = Path("output/player_insights.json")
        if insights_path.exists():
            with open(insights_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"인사이트 파일 로드 실패: {e}")
    return {}

def get_ai_analysis_for_pdf(player_hsi, team_template, team_name, player_name, pos_group, adapt_fit_score):
    """Gemini AI - 유럽 빅리그 스카우터 수준 보고서 생성"""
    model = get_gemini_model()
    
    if not model:
        print("WARNING: Gemini 모델 로드 실패, Fallback 사용")
        return _generate_fallback_analysis(player_hsi, team_template, team_name, player_name, pos_group, adapt_fit_score)
    
    # 선수 상세 인사이트 로드
    insights = load_player_insights()
    player_insight = insights.get(player_name, {})
    
    # 인사이트가 없으면 기본값 사용
    if not player_insight:
        print(f"WARNING: {player_name} 선수 인사이트 없음, 기본값 사용")
        player_insight = {
            "defensive_style": "데이터 부족",
            "pressing_style": "데이터 부족",
            "pressing_intensity_pct": "N/A",
            "discipline_level": "정보 부족",
            "fouls_per_game": "N/A",
            "summer_profile": "데이터 부족",
            "summer_retention_pct": "N/A",
            "experience_level": "정보 부족",
            "total_games": 0
        }
    
    # 포지션별 가중치 및 역할
    if pos_group == 'FW':
        weight_desc = "공격수(FW)는 전술 실행(T-Fit 50%), 체력 유지(P-Fit 30%), 문화 적응(C-Fit 20%) 가중치 적용"
        tactical_focus = "전방 압박 강도, 수비 전환 참여도, 공격 공간 창출 능력"
        benchmark = "리버풀 피르미누(압박형 FW), 맨체스터 시티 홀란드(결정력형 FW)"
    elif pos_group == 'MF':
        weight_desc = "미드필더(MF)는 전술 실행(T-Fit 40%), 체력 유지(P-Fit 40%), 문화 적응(C-Fit 20%) 가중치 적용"
        tactical_focus = "중원 압박 및 인터셉트, 공수 전환 연결, 활동량 지속력"
        benchmark = "첼시 캉테(압박형 MF), 바르셀로나 페드리(빌드업형 MF)"
    elif pos_group == 'DF':
        weight_desc = "수비수(DF)는 전술 실행(T-Fit 30%), 체력 유지(P-Fit 30%), 문화 적응(C-Fit 40%) 가중치 적용"
        tactical_focus = "최종 방어선 관리, 빌드업 참여, 라인 컨트롤"
        benchmark = "맨시티 디아스(빌드업형 CB), 리버풀 판 다이크(지배형 CB)"
    else:
        weight_desc = "기타 포지션은 균등 가중치(각 33%) 적용"
        tactical_focus = "팀 전술 시스템 내 특수 역할"
        benchmark = "포지션별 벤치마크 적용"
    
    # 통계적 맥락
    t_diff = player_hsi['t_fit_score'] - team_template['t_fit_score']
    p_diff = player_hsi['p_fit_score'] - team_template['p_fit_score']
    c_diff = player_hsi['c_fit_score'] - team_template['c_fit_score']
    
    # 퍼센타일 해석
    t_pctl = player_hsi['t_fit_score']
    p_pctl = player_hsi['p_fit_score']
    
    if t_pctl >= 90:
        t_league_rank = "상위 10% (엘리트 수준)"
    elif t_pctl >= 75:
        t_league_rank = "상위 25% (우수)"
    elif t_pctl >= 50:
        t_league_rank = "중위권 (평균 이상)"
    else:
        t_league_rank = "하위권 (개선 필요)"
    
    if p_pctl >= 90:
        p_league_rank = "상위 10% (여름철 강점)"
    elif p_pctl >= 75:
        p_league_rank = "상위 25% (안정적)"
    elif p_pctl >= 50:
        p_league_rank = "중위권 (무난)"
    else:
        p_league_rank = "하위권 (주의 필요)"
    
    # 현재 날짜 가져오기
    from datetime import datetime
    current_date = datetime.now().strftime("%Y년 %m월 %d일")
    
    # 유럽 빅리그 스카우터 수준 프롬프트
    prompt = f"""
당신은 유럽 5대 리그(프리미어리그, 라리가, 분데스리가, 세리에A, 리그1) 소속 구단의 스카우팅 디렉터입니다.
20년 경력의 데이터 분석 기반 스카우팅 전문가로서, {team_name}의 {pos_group} 포지션 영입을 위한 
**{player_name} 선수에 대한 전술 분석 보고서**를 작성합니다.

보고서 헤더:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{team_name} {pos_group} 포지션 영입을 위한 {player_name} 선수 전술 분석 보고서
작성일: {current_date}
작성자: 스카우팅 디렉터
수신: {team_name} 단장 및 감독
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

주의사항: 
- 기술적인 계산 과정이나 데이터 매핑 정책은 언급하지 마세요.
- C-Fit 계산 방법, WVS 데이터 출처, fallback 정책 등 내부 기술 정보를 보고서에 포함하지 마세요.
- 오직 스카우팅 분석 결과와 실무적 제안만 작성하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【EXECUTIVE SUMMARY】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▸ 선수: {player_name} ({pos_group})
▸ 분석 대상 구단: {team_name}
▸ 종합 적합도: {adapt_fit_score:.1f}/100 (HSI 알고리즘 기반)

핵심 지표 (K리그 2024 시즌, 베이지안 보정 + 퍼센타일 변환)
  → T-Fit (전술 실행력): {t_pctl:.1f}%ile | 리그 내 {t_league_rank}
     · 팀 평균 대비: {t_diff:+.1f}p ({("ABOVE: 팀 기준치 초과" if t_diff > 5 else "FAIR: 팀 평균 수준" if t_diff > -5 else "BELOW: 팀 기준치 미달")})
     · 수비 스타일: {player_insight.get('defensive_style', '데이터 부족')}
     · 압박 스타일: {player_insight.get('pressing_style', '데이터 부족')}
     · 압박 강도: {player_insight.get('pressing_intensity_pct', 'N/A')}
  
  → P-Fit (피지컬 지속력): {p_pctl:.1f}%ile | 리그 내 {p_league_rank}
     · 팀 평균 대비: {p_diff:+.1f}p ({("STRONG: 여름철 강점" if p_diff > 5 else "MODERATE: 무난한 수준" if p_diff > -5 else "WEAK: 여름철 주의")})
     · 여름철 프로파일: {player_insight.get('summer_profile', '데이터 부족')}
     · 혹서기 유지율: {player_insight.get('summer_retention_pct', 'N/A')}
  
  → C-Fit (문화 적응): {player_hsi['c_fit_score']:.2f} (WVS 도시 벡터 기반)
     · 초기 적응 예상 기간: {("2-4주 (빠른 정착)" if c_diff > 0 else "6-8주 (적응 지원 필요)")}

▸ 출전 경험: {player_insight.get('total_games', 0)}경기 ({player_insight.get('experience_level', '정보 부족')})
▸ 경기 규율: {player_insight.get('discipline_level', '정보 부족')} (경기당 파울 {player_insight.get('fouls_per_game', 'N/A')})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【SECTION 1: TACTICAL PROFILE & SYNERGY ANALYSIS】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.1 포지션별 전술 요구사항 (Position-Specific Requirements)
  • {pos_group} 핵심 역할: {tactical_focus}
  • 유럽 빅리그 벤치마크: {benchmark}
  • {weight_desc}

1.2 전술 실행 능력 평가 (Tactical Execution Assessment)
  [T-Fit {t_pctl:.1f}%ile 심층 분석]
  
  당신의 임무: 다음 관점에서 전술 분석을 수행하세요.
  
  ① 리그 내 상대 평가
     - T-Fit {t_pctl:.1f}%ile은 K리그 전체 선수 중 어느 수준인가?
     - 이 수치가 유럽 리그로 환산 시 어느 등급에 해당하는가?
     - {pos_group} 포지션 특성상 이 압박 강도가 충분한가, 부족한가?
  
  ② {team_name}과의 전술 궁합
     - 팀 평균 대비 {t_diff:+.1f}p 차이의 실전 의미는 무엇인가?
     - {team_name}의 플레이 스타일(압박 vs 블록, 점유 vs 역습)에 적합한가?
     - 포지션 내 역할 분담(압박 트리거, 커버 범위) 최적화 방안은?
  
  ③ 압박 스타일 & 수비 성향 분석
     - 수비 스타일: {player_insight.get('defensive_style', '분석 불가')}
     - 압박 스타일: {player_insight.get('pressing_style', '분석 불가')}
     - 이 스타일이 {team_name} {pos_group} 포지션에 유리한가, 불리한가?
     - 경쟁 포지션 선수들과 비교 시 차별점은?
  
  ④ 경기 규율 & 클린플레이
     - {player_insight.get('discipline_level', '정보 부족')}: 경기당 {player_insight.get('fouls_per_game', 'N/A')} 파울
     - 압박 강도 대비 징계 리스크 수준은 적절한가?
     - 주요 경쟁 리그(ACL, 컵 대회) 심판 기준 적응 가능성은?

1.3 전술 시너지 종합 판단
  • 즉시 전력 가능 여부
  • 적응 훈련 필요 항목
  • 포지션 내 최적 역할 제안

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【SECTION 2: PHYSICAL & ENVIRONMENTAL ADAPTATION】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2.1 K리그 특수 환경 (혹서기 6-8월) 적응력
  [P-Fit {p_pctl:.1f}%ile 심층 분석]
  
  당신의 임무: K리그만의 특수 환경에 대한 분석을 수행하세요.
  
  ① 여름철 퍼포먼스 패턴
     - P-Fit {p_pctl:.1f}%ile의 의미: {p_league_rank}
     - 여름철 성능 유지율: {player_insight.get('summer_retention_pct', 'N/A')}
     - 이것이 주전 경쟁력에 미치는 영향은?
  
  ② 일정 밀집도 대응력
     - 주중-주말 연속 경기(주 2경기) 소화 능력 예측
     - 로테이션 운영 필요성 vs 풀타임 주전 가능성
     - 체력 저하 시점 예측 (시즌 초반/중반/후반)
  
  ③ 팀 전력 분석
     - 팀 평균 대비 {p_diff:+.2f}p 차이 해석
     - {("여름철 경쟁력 우위" if p_diff > 0 else "로테이션 운영 권장")}
     - 백업 선수 vs 주전 선수 운용 전략 제안
  
  ④ 피지컬 리스크 관리
     - 부상 리스크 요인 (체력 저하, 과부하)
     - 예방 트레이닝 권장 사항
     - 의료진 모니터링 포인트

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【SECTION 3: CULTURAL INTEGRATION & OFF-FIELD FACTORS】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3.1 초기 적응 예측 (Onboarding Timeline)
  • C-Fit: {player_hsi['c_fit_score']:.2f} (WVS 문화 거리 지수)
  • 예상 정착 기간: {("2-4주" if c_diff > 0 else "6-8주")}
  • 주요 장벽: 언어, 생활 루틴, 전술 커뮤니케이션

3.2 구단 지원 체계 제안
  • 통역 배치 필요성
  • 멘토링 시스템 (선배 외국인 선수 배정)
  • 주거/가족 정착 지원

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【SECTION 4: TRANSFER RECOMMENDATION & CONTRACT STRATEGY】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4.1 종합 영입 의견 (Final Verdict)
  당신의 최종 판단을 명확히 제시하세요:
  
  [RECOMMEND] 즉시 영입 추천 (Immediate Sign)
  [CONDITIONAL] 조건부 영입 (Conditional Sign)
  [PASS] 영입 보류 (Pass)

4.2 계약 조건 제안
  • 연봉 범위 추정
  • 계약 기간 (단기 vs 장기)
  • 성과 조항 (출전 시간, 득점, 어시스트)
  • 바이아웃 조항 검토

4.3 리스크 관리 전략
  • 주요 리스크 3가지
  • 각 리스크별 완화 방안
  • 모니터링 KPI 설정

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【작성 지침 (STRICT GUIDELINES)】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[작성 가이드라인]
- 어조: 단호하고 전문적 ("~입니다", "~해야 합니다" 사용)
- 근거: 모든 판단에 구체적 수치와 백분위 언급
- 깊이: 단순 나열 금지, 인과관계와 전술적 맥락 제시
- 실용성: 추상적 평가 대신 실행 가능한 액션 아이템 제시
- 균형: 강점과 약점을 모두 명시하되, 해결책 제시
- 분량: 최소 1200자 이상 (유럽 스카우팅 보고서 표준)

이제 위 프레임워크에 따라 **유럽 빅리그 수준의 전술 분석 보고서**를 작성하세요.
"""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=8192,  # 최대 토큰 수로 증가 (하루 약 140개 보고서 생성 가능)
            )
        )
        
        # 응답 확인
        if hasattr(response, 'text') and response.text:
            result_text = response.text.strip()
            print(f"INFO: Gemini AI 분석 완료 ({len(result_text)} 자)")
            
            # 너무 짧은 응답은 오류로 간주
            if len(result_text) < 500:
                print(f"WARNING: Gemini 응답이 너무 짧음 ({len(result_text)}자). Fallback 사용")
                return _generate_fallback_analysis(player_hsi, team_template, team_name, player_name, pos_group, adapt_fit_score)
            
            return result_text
        else:
            print(f"✗ Gemini 응답이 비어있음. Response: {response}")
            return _generate_fallback_analysis(player_hsi, team_template, team_name, player_name, pos_group, adapt_fit_score)
            
    except Exception as e:
        print(f"✗ Gemini API 호출 실패: {e}")
        import traceback
        traceback.print_exc()
        return _generate_fallback_analysis(player_hsi, team_template, team_name, player_name, pos_group, adapt_fit_score)

def _generate_fallback_analysis(player_hsi, team_template, team_name, player_name, pos_group, adapt_fit_score):
    """AI 호출 실패 시 자동 생성되는 상세 분석"""
    from datetime import datetime
    current_date = datetime.now().strftime("%Y년 %m월 %d일")
    
    # T-Fit 분석 (퍼센타일 기준)
    t_score = player_hsi['t_fit_score']
    t_diff = player_hsi['t_fit_score'] - team_template['t_fit_score']
    p_diff = player_hsi['p_fit_score'] - team_template['p_fit_score']
    c_diff = player_hsi['c_fit_score'] - team_template['c_fit_score']
    
    if t_score >= 80:
        t_analysis = f"T-Fit {t_score:.1f}%ile (리그 상위 20%)는 우수한 전방 수비/압박 실행 능력을 나타냅니다. 전방 지역에서의 Duel/Tackle/Interception 참여가 리그 평균을 크게 상회하며, 하이프레싱 전술에 즉시 투입 가능합니다."
    elif t_score >= 60:
        t_analysis = f"T-Fit {t_score:.1f}%ile (리그 중상위권)는 양호한 전술 실행 지표입니다. 팀 전술에 따라 압박 강도를 조절할 수 있으며, 적응 훈련 후 주전 경쟁이 가능합니다."
    elif t_score >= 40:
        t_analysis = f"T-Fit {t_score:.1f}%ile (리그 중위권)는 평균 수준의 전술 실행력입니다. 특정 역할에 특화된 훈련이 필요하며, 로테이션 자원으로 활용 가능합니다."
    else:
        t_analysis = f"T-Fit {t_score:.1f}%ile (리그 하위권)는 전방 압박 참여가 제한적입니다. 블록 수비 중심 팀에서 특정 역할로 활용 가능하나, 하이프레싱 팀에는 부적합합니다."
    
    # P-Fit 분석 (퍼센타일 기준)
    p_score = player_hsi['p_fit_score']
    if p_score >= 80:
        p_analysis = f"P-Fit {p_score:.1f}%ile (리그 상위 20%)는 뛰어난 혹서기 적응력을 보입니다. K리그 여름철(6-8월)에도 안정적인 활동량을 유지하여 주전 경쟁에서 유리합니다."
    elif p_score >= 60:
        p_analysis = f"P-Fit {p_score:.1f}%ile (리그 중상위권)는 양호한 여름철 체력 유지 능력입니다. 여름 일정 소화에 큰 문제가 없으나, 밀집 일정 시 로테이션 고려가 권장됩니다."
    elif p_score >= 40:
        p_analysis = f"P-Fit {p_score:.1f}%ile (리그 중위권)는 평균적인 여름철 대응력입니다. 혹서기 컨디션 관리가 필요하며, 주중-주말 연속 경기 시 체력 저하 가능성이 있습니다."
    else:
        p_analysis = f"P-Fit {p_score:.1f}%ile (리그 하위권)는 여름철 성능 저하 리스크가 있습니다. 7-8월 로테이션 운영이 필수이며, 체력 관리 프로그램이 필요합니다."
    
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{team_name} {pos_group} 포지션 영입을 위한 {player_name} 선수 전술 분석 보고서
작성일: {current_date}
작성자: 스카우팅 디렉터
수신: {team_name} 단장 및 감독
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A. 선수 프로필 및 HSI 분석

{player_name} 선수의 HSI(Harmonic Synergy Index) 지표를 분석한 결과입니다.

[T-Fit: 전술 적합도]
{t_analysis}

[P-Fit: 환경 적합도]
{p_analysis}

[C-Fit: 문화 적응]
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
    
    # 한글 폰트 설정 (배포 환경 호환)
    font_candidates = [
        # 1순위: 배포 환경 (Streamlit Cloud - apt로 설치된 나눔폰트)
        ('NanumGothic', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'),
        ('NanumBarunGothic', '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'),
        # 2순위: macOS 시스템 폰트
        ('AppleGothic', '/System/Library/Fonts/AppleGothic.ttf'),
        ('AppleGothic', '/System/Library/Fonts/Supplemental/AppleGothic.ttf'),
        ('NotoSansGothic', '/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf'),
        # 3순위: Windows 시스템 폰트
        ('Malgun', 'C:\\Windows\\Fonts\\malgun.ttf'),
        ('Gulim', 'C:\\Windows\\Fonts\\gulim.ttc'),
    ]
    korean_font = 'Helvetica'
    for font_name, font_path in font_candidates:
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                korean_font = font_name
                print(f"INFO: 한글 폰트 로드 성공: {font_name} ({font_path})")
                break
            except Exception as e:
                print(f"✗ 폰트 로드 실패: {font_name} - {e}")
                continue
    
    if korean_font == 'Helvetica':
        st.warning("한글 폰트를 찾지 못해 기본 폰트로 생성합니다. 한글이 깨질 수 있습니다.")

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
# 사이드바 (New Scouter Layout 390px Optimized)
# ============================================================

# 1. Branding (Clean)
st.sidebar.markdown(f"""
<div style="padding: 1rem 0 0.5rem 0;">
    <h1 style="font-size: 1.2rem; font-weight: 800; margin: 0; color: {text_primary}; letter-spacing: -0.01em;">K-SCOUT PRO</h1>
    <p style="font-size: 0.75rem; color: {text_tertiary}; margin: 2px 0 0 0;">Advanced Player Adaptation Analysis</p>
</div>
""", unsafe_allow_html=True)

if GENAI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
    st.sidebar.caption("🟢 AI Engine Active")
else:
    st.sidebar.caption("🔴 AI Engine Offline")

st.sidebar.markdown(f"<hr style='margin: 1rem 0; border-color: {border_color};'>", unsafe_allow_html=True)

# 2. PDF Action (Top Priority)
st.sidebar.markdown(f"<div style='font-size: 0.75rem; font-weight: 600; color: {text_secondary}; margin-bottom: 8px; letter-spacing: 0.05em;'>ACTIONS</div>", unsafe_allow_html=True)
pdf_button = st.sidebar.button("📄 Generate PDF Report", key="generate_pdf_primary", type="primary", use_container_width=True)

st.sidebar.markdown(f"<div style='height: 24px'></div>", unsafe_allow_html=True)

# 3. Scouting Target (Player)
st.sidebar.markdown(f"<div style='font-size: 0.75rem; font-weight: 600; color: {text_secondary}; margin-bottom: 8px; letter-spacing: 0.05em;'>SCOUTING TARGET</div>", unsafe_allow_html=True)
foreign_player_names = sorted(foreigners_df['player_name_ko'].unique())
selected_player_name = st.sidebar.selectbox(
    "Select Player",
    foreign_player_names,
    index=0 if len(foreign_player_names) > 0 else None,
    placeholder="Search player...",
    label_visibility="collapsed"
)

# 선택 선수의 구단(동명이인 구분/매핑 UI용)
selected_player_club = ""
try:
    if "club_name_ko" in foreigners_df.columns:
        clubs = foreigners_df.loc[foreigners_df["player_name_ko"] == selected_player_name, "club_name_ko"]
        if not clubs.empty:
            selected_player_club = str(clubs.mode().iloc[0]).strip()
except Exception:
    selected_player_club = ""

if selected_player_club:
    st.sidebar.caption(f"Current Club: {selected_player_club}")

st.sidebar.markdown(f"<div style='height: 24px'></div>", unsafe_allow_html=True)

# 4. Context (Team Selector - Nav List Style)
st.sidebar.markdown(f"<div style='font-size: 0.75rem; font-weight: 600; color: {text_secondary}; margin-bottom: 8px; letter-spacing: 0.05em;'>ANALYSIS CLUB</div>", unsafe_allow_html=True)

# 2024 Ranking Order
priority_order = [
    "울산 HD FC", "강원FC", "FC서울", "수원FC", 
    "포항 스틸러스", "제주SK FC", "대전 하나 시티즌", "광주FC", 
    "전북 현대 모터스", "대구FC", "인천 유나이티드"
]
existing_teams = list(templates.keys())
# 김천 상무 제외 필터링
team_list = [t for t in existing_teams if "김천" not in t]
team_list = sorted(team_list, key=lambda x: priority_order.index(x) if x in priority_order else 999)

team_display_map = {
    "제주SK FC": "제주 SK FC",
    "강원FC": "강원 FC",
    "광주FC": "광주 FC",
    "대구FC": "대구 FC",
    "수원FC": "수원 FC",
    "FC서울": "FC 서울"
}

# [NEW] 선택 상태 표시(즉시 반영): placeholder를 위에 만들고, 라디오 선택값으로 채웁니다.
target_box = st.sidebar.empty()

# 이전 선택값이 옵션에 없으면(예: 김천 제거 등) 안전하게 첫 항목으로 보정
if team_list and st.session_state.get("client_team_selector") not in team_list:
    st.session_state["client_team_selector"] = team_list[0]

# Navigation Style Grid (styled by CSS)
client_team = st.sidebar.radio(
    "Select Club",
    team_list,
    format_func=lambda x: team_display_map.get(x, x),
    label_visibility="collapsed",
    key="client_team_selector",
)

current_team_display = team_display_map.get(client_team, client_team)
target_box.markdown(
    f"""
<div style="
    margin-bottom: 12px; 
    padding: 10px 14px; 
    background: {input_bg}; 
    border: 1px solid {accent_color}60; 
    border-radius: 8px; 
    display: flex; 
    justify-content: space-between; 
    align-items: center;
">
    <span style="font-size: 0.75rem; color: {text_secondary}; font-weight: 500;">TARGET</span>
    <span style="font-size: 0.9rem; font-weight: 700; color: {text_primary};">{current_team_display}</span>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

# 5. Advanced Settings & Theme
with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
    st.caption("HSI Metrics Guide")
    st.markdown("""
    - **T-Fit**: Tactical Execution
    - **P-Fit**: Physical Adaptation
    - **C-Fit**: Cultural Fit (WVS)
    """)
    st.markdown("---")
    
    # --- 선수 매핑 UI ---
    st.markdown("**Player Origin**")
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

    p_country_in = st.text_input("Country (Alpha-3)", value=str(prow_val.get("home_country_alpha", "")).strip(), key="p_home_country_alpha_in")
    p_loc_in = st.text_input("Loc Code (WVS)", value=str(prow_val.get("home_loc_code", "")).strip(), key="p_home_loc_code_in")
    
    if st.button("Save Player Origin", key="save_player_city_btn"):
        pmap = pmap.copy()
        mask = (pmap["player_name_ko"] == selected_player_name)
        if "club_name_ko" in pmap.columns and selected_player_club:
            mask = mask & (pmap["club_name_ko"] == selected_player_club)
        
        if mask.any():
            pmap.loc[mask, ["home_country_alpha", "home_loc_code", "home_city_label"]] = [
                p_country_in.strip().upper(), p_loc_in.strip(), ""
            ]
        else:
            new_row = {
                "player_name_ko": selected_player_name,
                "club_name_ko": selected_player_club,
                "home_country_alpha": p_country_in.strip().upper(),
                "home_loc_code": p_loc_in.strip(),
                "home_city_label": ""
            }
            pmap = pd.concat([pmap, pd.DataFrame([new_row])], ignore_index=True)
        pmap.to_csv(PLAYER_CITY_MAP_PATH, index=False)
        st.success("Saved!")
        st.rerun()

    # --- 팀 매핑 UI ---
    st.markdown("**Club Location**")
    try:
        tmap = pd.read_csv(TEAM_CITY_MAP_PATH)
        trow_df = tmap[tmap["team_name_ko"] == client_team]
        trow_val = trow_df.head(1).iloc[0] if not trow_df.empty else {}
    except Exception:
        tmap = pd.DataFrame(columns=["team_name_ko", "host_country_alpha", "host_loc_code", "host_city_label"])
        trow_val = {}

    t_loc_in = st.text_input("Loc Code (Team)", value=str(trow_val.get("host_loc_code", "")).strip(), key="t_host_loc_code_in")
    
    if st.button("Save Club Location", key="save_team_city_btn"):
        tmap = tmap.copy()
        mask = tmap["team_name_ko"] == client_team
        if mask.any():
            tmap.loc[mask, ["host_country_alpha", "host_loc_code"]] = ["KOR", t_loc_in.strip()]
        else:
            new_row = {"team_name_ko": client_team, "host_country_alpha": "KOR", "host_loc_code": t_loc_in.strip(), "host_city_label": ""}
            tmap = pd.concat([tmap, pd.DataFrame([new_row])], ignore_index=True)
        tmap.to_csv(TEAM_CITY_MAP_PATH, index=False)
        st.success("Saved!")
        st.rerun()

st.sidebar.markdown('<div style="flex-grow: 1;"></div>', unsafe_allow_html=True)

# Theme Toggle
current_theme = st.session_state.theme_mode
toggle_icon = "☀️ Light Mode" if current_theme == 'Dark' else "🌙 Dark Mode"
if st.sidebar.button(toggle_icon, key="theme_toggle_btn"):
    st.session_state.theme_mode = 'Light' if current_theme == 'Dark' else 'Dark'
    st.rerun()

# ============================================================
# 메인 컨텐츠
# ============================================================

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
    # --- 레이아웃 구조 개선 (Bento Grid Style) ---
    # 상단: 프로필 카드 + 종합 점수 카드
    col_profile, col_score = st.columns([1.2, 0.8])
    
    with col_profile:
        # 프로필 정보 구성 (예외 처리 포함)
        en_name = ""
        nationality = ""
        try:
            # 외국인 선수 상세 정보
            if FOREIGN_PLAYERS_EXTENDED_PATH.exists():
                foreign_prof = pd.read_csv(FOREIGN_PLAYERS_EXTENDED_PATH)
                if not foreign_prof.empty and "player_name_ko" in foreign_prof.columns:
                    fhit = foreign_prof[foreign_prof["player_name_ko"] == selected_player_name]
                    if not fhit.empty:
                        frow = fhit.iloc[0]
                        en_name = str(frow.get("english_full_name", "")).strip()
                        nationality = str(frow.get("nationality", "")).strip()
            
            # 기본 프로필 파일
            if (not en_name or not nationality) and PLAYER_PROFILE_PATH.exists():
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
                        if not en_name:
                            en_name = str(prow.get("player_name_en_full", "")).strip()
                        if not nationality:
                            nationality = str(prow.get("nationality", "")).strip()
        except:
            pass

        # HTML 카드 렌더링
        st.markdown(f"""
        <div class="metric-card" style="display: flex; align-items: center; gap: 24px; height: 100%;">
            <div style="
                width: 80px; height: 80px; 
                background-color: {input_bg}; 
                border-radius: 50%; 
                display: flex; align-items: center; justify-content: center;
                font-size: 2rem; font-weight: 700; color: {text_secondary};
                border: 1px solid {border_color};
                flex-shrink: 0;
            ">
                {selected_player_name[0] if selected_player_name else "?"}
            </div>
            <div>
                <div style="font-size: 0.85rem; color: {text_secondary}; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em;">{get_position_korean(player_pos)} • {pos_group}</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {text_primary}; margin-bottom: 8px; letter-spacing: -0.02em;">{selected_player_name}</div>
                <div style="display: flex; gap: 12px; font-size: 0.85rem; color: {text_tertiary};">
                    {'<span>' + en_name + '</span>' if 'en_name' in locals() and en_name and en_name != 'nan' else ''}
                    {'<span style="opacity: 0.5;">|</span>' if 'en_name' in locals() and en_name and 'nationality' in locals() and nationality else ''}
                    {'<span>' + nationality + '</span>' if 'nationality' in locals() and nationality and nationality != 'nan' else ''}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_score:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 20px; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 0.85rem; color: {text_secondary}; margin-bottom: 4px;">ADAPT-FIT SCORE</div>
            <div style="font-size: 3rem; font-weight: 800; color: {text_primary}; line-height: 1; letter-spacing: -0.03em;">
                {adapt_fit_score:.0f}
                <span style="font-size: 1rem; color: {text_tertiary}; font-weight: 500; margin-left: 2px;">/ 100</span>
            </div>
            <div style="font-size: 0.9rem; color: {grade_color if 'grade_color' in locals() else text_primary}; font-weight: 600; margin-top: 8px;">
                {grade} Grade
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # 하단: 2열 구조 (Metrics & Radar)
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        # ---------------------------------------------------------
        # 1. HSI Metrics 카드
        # ---------------------------------------------------------
        def get_metric_row_html(label, value, avg_val, color_hex):
            pct = min(100, max(0, value if value > 1 else value * 100))
            val_str = f"{value:.1f}" if value > 1 else f"{value:.2f}"
            avg_str = f"{avg_val:.1f}" if avg_val > 1 else f"{avg_val:.2f}"
            # 들여쓰기 제거된 HTML 문자열
            return f"""<div style="margin-bottom: 20px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items: baseline;">
<span style="font-weight: 500; color: {text_primary}; font-size: 0.9rem;">{label}</span>
<span style="color: {text_secondary}; font-size: 0.85rem;">
<strong style="color: {text_primary};">{val_str}</strong> 
<span style="opacity: 0.7;">vs {avg_str}</span>
</span>
</div>
<div style="width: 100%; height: 6px; background-color: {input_bg}; border-radius: 3px; overflow: hidden;">
<div style="width: {pct}%; height: 100%; background-color: {color_hex}; border-radius: 3px;"></div>
</div>
</div>"""

        hsi_content = ""
        hsi_content += get_metric_row_html("Tactical Fit", player_hsi_for_score['t_fit_score'], team_template_for_score['t_fit_score'], "#3B82F6")
        hsi_content += get_metric_row_html("Physical Fit", player_hsi_for_score['p_fit_score'], team_template_for_score['p_fit_score'], "#10B981")
        hsi_content += get_metric_row_html("Cultural Fit", player_hsi_for_score['c_fit_score'], team_template_for_score['c_fit_score'], "#F59E0B")

        # 전체 카드 HTML (들여쓰기 제거)
        st.markdown(f"""<div class="metric-card">
<h4 style="margin: 0 0 24px 0; font-size: 1rem; color: {text_primary}; font-weight: 600;">HSI BREAKDOWN</h4>
{hsi_content}
</div>""", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 2. AI Analysis 카드
        # ---------------------------------------------------------
        ai_content = ""
        if strengths:
            for s in strengths:
                ai_content += f"<div style='margin-bottom: 8px; font-size: 0.9rem; color: {text_secondary}; display: flex; gap: 8px;'><span style='color: #10B981;'>✓</span> <span>{s}</span></div>"
        
        if weaknesses:
            ai_content += "<div style='height: 8px;'></div>"
            for w in weaknesses:
                ai_content += f"<div style='margin-bottom: 8px; font-size: 0.9rem; color: {text_secondary}; display: flex; gap: 8px;'><span style='color: #F59E0B;'>!</span> <span>{w}</span></div>"
        
        if not strengths and not weaknesses:
             ai_content += f"<div style='color: {text_secondary}; font-size: 0.9rem;'>특이 사항 없음</div>"

        st.markdown(f"""<div class="metric-card">
<h4 style="margin: 0 0 16px 0; font-size: 1rem; color: {text_primary}; font-weight: 600;">AI INSIGHTS</h4>
{ai_content}
</div>""", unsafe_allow_html=True)

    with c2:
        # Radar Chart (별도 박스 없이 차트 자체를 CSS로 꾸밈)
        categories = ["Tactical", "Physical", "Cultural"]
        player_r = [
            player_hsi_for_score["t_fit_score"],
            player_hsi_for_score["p_fit_score"],
            player_hsi_for_score["c_fit_score"] * 100,
        ]
        team_r = [
            team_template_for_score["t_fit_score"],
            team_template_for_score["p_fit_score"],
            team_template_for_score["c_fit_score"] * 100,
        ]

        fig = go.Figure()
        # Player Area
        fig.add_trace(go.Scatterpolar(
                r=player_r + [player_r[0]],
                theta=categories + [categories[0]],
                fill="toself",
            name=selected_player_name,
            line=dict(color=text_primary, width=2),
            fillcolor=f"rgba(128, 128, 128, 0.2)"
        ))
        # Team Line
        fig.add_trace(go.Scatterpolar(
                r=team_r + [team_r[0]],
                theta=categories + [categories[0]],
            fill="none",
            name="Team Avg",
            line=dict(color=text_secondary, width=1, dash='dot'),
        ))

        fig.update_layout(
            # 차트 제목을 내부에서 처리
            title=dict(
                text="SKILL RADAR",
                x=0.05,
                y=0.98,
                xanchor='left',
                yanchor='top',
                font=dict(size=14, color=text_primary, family="Pretendard")
            ),
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color=tick_color, size=9), gridcolor=grid_color, linecolor=grid_color),
                angularaxis=dict(tickfont=dict(color=text_primary, size=11, weight="bold"), gridcolor=grid_color, linecolor=grid_color),
                bgcolor=polar_bgcolor,
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5, font=dict(color=text_primary)),
            paper_bgcolor="rgba(0,0,0,0)", # 투명 배경 (CSS 카드가 뒤에 보임)
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=60, b=40, l=40, r=40), # 제목 공간 확보
            height=500, # 높이 증가
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown(f"### {client_team} 추천 외국인 선수 Top 10")
    st.markdown(f"*{client_team}의 전술 스타일에 적합한 외국인 선수 순위입니다.*")
    
    all_scores = get_all_player_scores(client_team, templates, hsi_df, foreigners_df)
    top10 = all_scores[:10]

    # Top 3 Highlight Cards (한 번에 판단)
    top3 = top10[:3]
    if top3:
        cards_html = "<div class='rank-grid'>"
        for i, p in enumerate(top3):
            rank = i + 1
            nm = p.get("name", "")
            pos_ko = get_position_korean(p.get("position", ""))
            score = float(p.get("score", 0))
            grade = str(p.get("grade", ""))
            gcolor = str(p.get("color", accent_color))
            is_sel = (nm == selected_player_name)
            sel_cls = " is-selected" if is_sel else ""
            bar_w = max(0, min(100, score))

            # grade-chip은 선수 grade 색상 기반으로 테두리/텍스트만 포인트
            cards_html += f"""
<div class="rank-card{sel_cls}">
  <div class="rank-head">
    <div class="rank-badge">{rank}위</div>
    <div class="grade-chip" style="border-color:{gcolor}55;color:{gcolor};">{grade}</div>
  </div>
  <div class="rank-name">{nm}</div>
  <div class="rank-meta">{pos_ko}</div>
  <div class="score-line">
    <div class="score-value">{score:.1f}<span class="score-unit">점</span></div>
  </div>
  <div class="score-bar"><span style="width:{bar_w:.1f}%;background:{gcolor};"></span></div>
</div>
"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

    # Full Top 10 List (스캔하기 쉬운 리스트)
    if top10:
        list_html = "<div class='rank-list'>"
        for i, p in enumerate(top10):
            rank = i + 1
            nm = str(p.get("name", ""))
            pos_ko = get_position_korean(p.get("position", ""))
            score = float(p.get("score", 0))
            grade = str(p.get("grade", ""))
            gcolor = str(p.get("color", accent_color))
            is_sel = (nm == selected_player_name)
            row_cls = "rank-row is-selected" if is_sel else "rank-row"

            list_html += f"""
<div class="{row_cls}">
  <div class="rank-rank">{rank}위</div>
  <div class="rank-player">
    <div class="nm">{nm}</div>
    <div class="pos">{pos_ko}</div>
  </div>
  <div class="rank-score">{score:.1f}점</div>
  <div class="rank-grade"><span class="grade-chip" style="border-color:{gcolor}55;color:{gcolor};">{grade}</span></div>
</div>
"""
        list_html += "</div>"
        st.markdown(list_html, unsafe_allow_html=True)

with tab3:
    st.markdown("### 전체 선수 상세 데이터")
    
    all_scores = get_all_player_scores(client_team, templates, hsi_df, foreigners_df)

    def _to_pct(v: Any) -> float:
        try:
            fv = float(v)
        except Exception:
            return 0.0
        pct = fv if fv > 1 else fv * 100.0
        return max(0.0, min(100.0, pct))

    if not all_scores:
        st.info("표시할 데이터가 없습니다.")
    else:
        rows_html = "<div class='compare-list'>"
        for i, p in enumerate(all_scores):
            rank = i + 1
            nm = str(p.get("name", ""))
            pos_ko = get_position_korean(p.get("position", ""))
            score = float(p.get("score", 0.0))
            grade = str(p.get("grade", ""))
            gcolor = str(p.get("color", accent_color))

            t_val = float(p.get("t_fit", 0.0))
            p_val = float(p.get("p_fit", 0.0))
            c_val = float(p.get("c_fit", 0.0))

            t_pct = _to_pct(t_val)
            p_pct = _to_pct(p_val)
            c_pct = _to_pct(c_val)

            sel_cls = " is-selected" if nm == selected_player_name else ""
            rows_html += f"""
<div class="compare-row{sel_cls}">
  <div class="compare-top">
    <div class="compare-badge">{rank}위</div>
    <div class="compare-center">
      <div class="nm">{nm}</div>
      <div class="pos">{pos_ko}</div>
    </div>
    <div class="compare-right">
      <div class="score">{score:.1f}<span>점</span></div>
      <div class="grade-chip" style="border-color:{gcolor}55;color:{gcolor};">{grade}</div>
    </div>
  </div>
  <div class="compare-metrics">
    <div class="metric-mini">
      <div class="lbl"><span>T-Fit</span><strong>{t_val:.1f}</strong></div>
      <div class="bar"><span style="width:{t_pct:.1f}%;background:#3B82F6;"></span></div>
    </div>
    <div class="metric-mini">
      <div class="lbl"><span>P-Fit</span><strong>{p_val:.2f}</strong></div>
      <div class="bar"><span style="width:{p_pct:.1f}%;background:#10B981;"></span></div>
    </div>
    <div class="metric-mini">
      <div class="lbl"><span>C-Fit</span><strong>{c_val:.3f}</strong></div>
      <div class="bar"><span style="width:{c_pct:.1f}%;background:#F59E0B;"></span></div>
    </div>
  </div>
</div>
"""
        rows_html += "</div>"
        st.markdown(rows_html, unsafe_allow_html=True)

# ============================================================
# PDF 생성
# ============================================================
if pdf_button:
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not GENAI_AVAILABLE:
        st.warning("google-generativeai 라이브러리가 설치되지 않았습니다. 'pip install google-generativeai'를 실행해주세요.")
    
    if not gemini_key and GENAI_AVAILABLE:
        st.info("💡 Gemini API 키가 설정되지 않았습니다. AI 분석 없이 기본 통계 분석만 제공됩니다.")
    
    with st.spinner("AI 기반 스카우팅 보고서를 생성하는 중..."):
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
                        range=[0, 100],
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
                    yanchor="top",
                    y=-0.15,  # 그래프 하단으로 이동
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
                player_hsi_for_score, 
                team_template_for_score, 
                client_team, 
                selected_player_name, 
                pos_group, 
                adapt_fit_score
            )
            
            # AI 분석 결과 확인
            if not ai_analysis_text or len(ai_analysis_text.strip()) < 100:
                st.warning("AI 분석 생성에 실패했거나 내용이 부족합니다. Fallback 분석을 사용합니다.")
                ai_analysis_text = _generate_fallback_analysis(
                    player_hsi_for_score,
                    team_template_for_score,
                    client_team,
                    selected_player_name,
                    pos_group,
                    adapt_fit_score
                )
            
            # 디버깅: AI 분석 길이 확인
            print(f"AI 분석 텍스트 길이: {len(ai_analysis_text)} 자")
            
            # C-Fit 기술 정보는 PDF에 포함하지 않음 (AI 프롬프트에서 이미 제외 지시)

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
    <p style="color: #9CA3AF; font-weight: 600; margin: 0;">K-Scout Adapt-Fit AI • MVP Version</p>
    <p style="color: #A1A1AA; font-size: 0.75rem; margin: 0.5rem 0;">2024 K리그 데이터 기반 전술 적합도 분석 시스템</p>
    <p style="color: #6B7280; font-size: 0.625rem; margin: 0;">© 2024 ANYONE COMPANY. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
