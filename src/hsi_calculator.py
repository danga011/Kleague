import os
import json
import logging
import numpy as np
import pandas as pd

# ============================================================
# 로깅 시스템 설정
# ============================================================
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/hsi_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def validate_raw_data(df: pd.DataFrame) -> bool:
    """입력 데이터 검증"""
    required_cols = ['player_id', 'player_name_ko', 'game_id', 'type_name', 'start_x']
    missing = set(required_cols) - set(df.columns)
    
    if missing:
        logger.error(f"필수 컬럼 누락: {missing}")
        raise ValueError(f"필수 컬럼 누락: {missing}")
    
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        logger.warning(f"결측치 발견:\n{null_counts[null_counts > 0]}")
    
    logger.info("데이터 검증 통과")
    return True


def bayesian_smoothing(raw_values, n_samples, global_mean, k=5.0):
    """베이지안 스무딩으로 소표본 선수 안정화"""
    return (raw_values * n_samples + global_mean * k) / (n_samples + k)


def to_percentile(series):
    """원본 점수를 백분위 점수(0-100)로 변환"""
    return series.rank(pct=True) * 100.0


def calculate_hsi_metrics(raw_data_path, match_info_path, output_dir):
    """
    통계적으로 고도화된 HSI 지표 계산
    - 베이지안 스무딩: 소표본 선수 안정화
    - 퍼센타일 변환: 리그 내 상대평가
    - 상세 인사이트: AI가 활용할 정성적 데이터 생성
    """
    logger.info("=" * 60)
    logger.info("HSI 고도화 파이프라인 시작")
    logger.info("=" * 60)
    
    # 1. 데이터 로딩 및 검증
    try:
        logger.info("데이터 로딩 중...")
        raw_df = pd.read_csv(raw_data_path)
        match_df = pd.read_csv(match_info_path)
        validate_raw_data(raw_df)
        logger.info(f"데이터 로드 완료: {len(raw_df)} 이벤트, {raw_df['player_id'].nunique()} 선수")
    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        return
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        return
    
    # 선수 목록 및 경기수
    players = raw_df[['player_id', 'player_name_ko']].drop_duplicates()
    games_played = raw_df.groupby("player_id")["game_id"].nunique().rename("games_played")
    
    # 2. T-Fit 계산 (전술 적합도 + 베이지안 스무딩)
    logger.info("T-Fit 계산 중 (베이지안 스무딩 적용)...")
    defensive_actions = ['Duel', 'Tackle', 'Interception']
    def_events = raw_df[raw_df['type_name'].isin(defensive_actions)]
    
    # 전방 압박 비중 (x > 60)
    high_press_counts = def_events[def_events['start_x'] > 60].groupby("player_id").size()
    total_def_counts = def_events.groupby("player_id").size()
    press_ratio = (high_press_counts / total_def_counts).fillna(0)
    
    # 수비 스타일 (가로채기 vs 태클 비율)
    interceptions = raw_df[raw_df['type_name'] == 'Interception'].groupby("player_id").size()
    tackles = raw_df[raw_df['type_name'] == 'Tackle'].groupby("player_id").size()
    style_ratio = (interceptions / (tackles + 1e-6)).fillna(0)
    
    # 베이지안 스무딩 적용
    league_avg_t = high_press_counts.sum() / games_played.sum()
    k_t = 5.0  # Prior strength
    
    t_fit_raw = bayesian_smoothing(
        high_press_counts.reindex(games_played.index).fillna(0),
        games_played,
        league_avg_t,
        k=k_t
    )
    
    # 클린플레이 반영
    fouls = raw_df[raw_df['type_name'] == 'Foul'].groupby('player_id').size()
    cards = raw_df[raw_df['type_name'].isin(['Yellow Card', 'Red Card', 'Yellow/Red Card'])].groupby('player_id').size()
    penalty_per_game = ((fouls.reindex(games_played.index).fillna(0) + 
                        cards.reindex(games_played.index).fillna(0) * 2) / games_played).fillna(0)
    
    cp_scale = penalty_per_game.quantile(0.75) if penalty_per_game.quantile(0.75) > 0 else 1.0
    clean_play_score = 1.0 / (1.0 + (penalty_per_game / cp_scale))
    
    t_fit_final_raw = t_fit_raw * (0.7 + 0.3 * clean_play_score)
    
    logger.info(f"  리그 평균 T-Fit: {league_avg_t:.2f}, 베이지안 k={k_t}")
    
    # 3. P-Fit 계산 (혹서기 적응 + 베이지안 스무딩)
    logger.info("P-Fit 계산 중 (여름철 활동량 분석)...")
    merged_df = pd.merge(raw_df, match_df[['game_id', 'game_date']], on='game_id')
    merged_df['game_date'] = pd.to_datetime(merged_df['game_date'])
    merged_df['is_summer'] = merged_df['game_date'].dt.month.isin([6, 7, 8])
    
    # 여름/평시 경기당 이벤트
    summer_games = merged_df[merged_df['is_summer']].groupby('player_id')['game_id'].nunique()
    rest_games = merged_df[~merged_df['is_summer']].groupby('player_id')['game_id'].nunique()
    
    summer_events = merged_df[merged_df['is_summer']].groupby('player_id').size()
    rest_events = merged_df[~merged_df['is_summer']].groupby('player_id').size()
    
    summer_per_game = (summer_events / summer_games).fillna(0)
    rest_per_game = (rest_events / rest_games).fillna(0)
    
    # 여름철 유지율
    p_retention = (summer_per_game / (rest_per_game + 1e-6)).fillna(1.0)
    
    # 베이지안 스무딩 (여름철 경기수 기반)
    league_avg_p = 1.0  # 중립 = 유지율 100%
    k_p = 3.0
    
    p_fit_raw = bayesian_smoothing(
        p_retention.reindex(games_played.index).fillna(1.0),
        summer_games.reindex(games_played.index).fillna(0),
        league_avg_p,
        k=k_p
    )
    
    logger.info(f"  평균 여름철 유지율: {p_retention.mean():.2%}, 베이지안 k={k_p}")
    
    # 4. 퍼센타일 변환 (상대평가)
    logger.info("퍼센타일 변환 중 (리그 내 상대평가)...")
    
    hsi_results = players.copy()
    hsi_results = hsi_results.merge(games_played, on="player_id", how="left")
    
    hsi_results['t_fit_score'] = to_percentile(t_fit_final_raw).reindex(hsi_results['player_id']).fillna(50.0).values
    hsi_results['p_fit_score'] = to_percentile(p_fit_raw).reindex(hsi_results['player_id']).fillna(50.0).values
    hsi_results['clean_play_score'] = clean_play_score.reindex(hsi_results['player_id']).fillna(0.8).values
    hsi_results['c_fit_score'] = 50.0  # Placeholder (앱에서 동적 계산)
    
    logger.info(f"  T-Fit 범위: {hsi_results['t_fit_score'].min():.1f} ~ {hsi_results['t_fit_score'].max():.1f}")
    logger.info(f"  P-Fit 범위: {hsi_results['p_fit_score'].min():.1f} ~ {hsi_results['p_fit_score'].max():.1f}")
    
    # 5. AI용 상세 인사이트 생성
    logger.info("AI용 상세 인사이트 생성 중...")
    insights = {}
    
    for _, row in hsi_results.iterrows():
        pid = row['player_id']
        p_name = row['player_name_ko']
        
        # 수비 스타일
        def_style = "예측 및 길목 차단형 (Anticipatory)" if style_ratio.get(pid, 0) > 1.2 else "저돌적 경합 및 태클형 (Aggressive)"
        
        # 압박 스타일
        press_intensity = press_ratio.get(pid, 0)
        if press_intensity > 0.5:
            press_style = "강력한 전방 압박 수행 (High-Presser)"
        elif press_intensity > 0.3:
            press_style = "적극적 중원 압박 (Medium-Presser)"
        else:
            press_style = "지역 방어 및 블록 형성 (Positional)"
        
        # 규율 수준
        discipline = "우수 (Clean)" if clean_play_score.get(pid, 0) > 0.8 else "주의 필요 (Risky)"
        
        # 여름철 성능
        summer_ret = p_retention.get(pid, 1.0)
        if summer_ret >= 1.05:
            summer_profile = f"여름철 성능 향상 ({summer_ret*100:.1f}%, 혹서기 강점)"
        elif summer_ret >= 0.95:
            summer_profile = f"여름철 안정적 유지 ({summer_ret*100:.1f}%)"
        else:
            summer_profile = f"여름철 성능 저하 ({summer_ret*100:.1f}%, 로테이션 고려)"
        
        # 경험 수준
        total_games = int(games_played.get(pid, 0))
        if total_games >= 25:
            experience = "풍부한 경험 (주전급)"
        elif total_games >= 15:
            experience = "적정 경험 (로테이션급)"
        else:
            experience = "제한적 출전 (백업)"
        
        insights[p_name] = {
            "defensive_style": def_style,
            "pressing_style": press_style,
            "pressing_intensity_pct": f"{press_intensity*100:.1f}%",
            "discipline_level": discipline,
            "fouls_per_game": f"{penalty_per_game.get(pid, 0):.2f}",
            "summer_profile": summer_profile,
            "summer_retention_pct": f"{summer_ret*100:.1f}%",
            "experience_level": experience,
            "total_games": total_games,
            "t_fit_percentile": f"{row['t_fit_score']:.1f}",
            "p_fit_percentile": f"{row['p_fit_score']:.1f}"
        }
    
    # 6. 파일 저장
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'hsi_scores_2024.csv')
    hsi_results.to_csv(output_path, index=False)
    logger.info(f"HSI 점수 저장: {output_path}")
    
    insights_path = os.path.join(output_dir, 'player_insights.json')
    with open(insights_path, 'w', encoding='utf-8') as f:
        json.dump(insights, f, ensure_ascii=False, indent=2)
    logger.info(f"선수 인사이트 저장: {insights_path}")
    
    logger.info("=" * 60)
    logger.info(f"파이프라인 완료: {len(players)} 명 선수 처리")
    logger.info("=" * 60)


if __name__ == '__main__':
    RAW_DATA_PATH = 'data/raw/raw_data.csv'
    MATCH_INFO_PATH = 'data/raw/match_info.csv'
    OUTPUT_DIR = 'output'
    calculate_hsi_metrics(RAW_DATA_PATH, MATCH_INFO_PATH, OUTPUT_DIR)
