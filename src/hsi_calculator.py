
import os

import numpy as np
import pandas as pd


def calculate_hsi_metrics(
    raw_data_path,
    match_info_path,
    output_dir,
):
    """
    HSI 지표를 계산하여 CSV 파일로 저장합니다.

    - T-fit: 전술 적합도(전방 수비액션) + 파울/카드 기반 클린플레이(반칙 리스크) 정보를 반영해 고도화
    - P-fit: 환경 적합도(혹서기 유지율)
    - C-fit: 문화 적합도(도시 기반)는 팀(분석도시)에 따라 달라지는 값이므로,
             앱(app.py)에서 WVS 도시 벡터/매핑을 통해 **동적으로 계산**합니다.
             이 파일에서는 placeholder(중립값)를 저장합니다.
    """
    try:
        print("Reading raw data and match info...")
        raw_df = pd.read_csv(raw_data_path)
        match_df = pd.read_csv(match_info_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
        return

    # 선수 목록 사전 생성
    players = raw_df[['player_id', 'player_name_ko']].drop_duplicates()

    # 경기 수(출전량 프록시) - minutes 데이터가 없으므로 최소한 경기수로 정규화
    games_played_total = raw_df.groupby("player_id")["game_id"].nunique().rename("games_played_total")
    
    # --- 1. T-fit (K-Pressure) 계산 ---
    print("Calculating T-fit (K-Pressure + CleanPlay)...")
    defensive_actions = ['Duel', 'Tackle', 'Interception']
    pressure_events = raw_df[
        (raw_df['type_name'].isin(defensive_actions)) & 
        (raw_df['start_x'] > 60)
    ]
    # 누적 카운트는 출전시간(경기수)에 강하게 종속 → 경기당 평균으로 정규화
    t_fit_count = pressure_events.groupby("player_id").size().rename("t_fit_count")
    with np.errstate(divide="ignore", invalid="ignore"):
        t_fit_per_game = (t_fit_count / games_played_total).replace([np.inf, -np.inf], 0).fillna(0).rename("t_fit_raw")
    t_fit_raw = t_fit_per_game.reset_index()

    # --- 2. P-fit (Summer Retention) 계산 ---
    print("Calculating P-fit (Summer Retention)...")
    merged_df = pd.merge(raw_df, match_df[['game_id', 'game_date']], on='game_id')
    merged_df['game_date'] = pd.to_datetime(merged_df['game_date'])
    merged_df['month'] = merged_df['game_date'].dt.month
    
    merged_df['season'] = np.where(merged_df['month'].isin([6, 7, 8]), 'Summer', 'Rest')
    
    # 시즌별 경기 수 계산
    games_per_season = merged_df.groupby(['player_id', 'season'])['game_id'].nunique().unstack(fill_value=0)
    # 시즌별 이벤트 수 계산
    events_per_season = merged_df.groupby(['player_id', 'season']).size().unstack(fill_value=0)
    
    # 0으로 나누는 것을 방지 + 소표본(경기수 적음) 안정화
    # - raw ratio = (여름 경기당 이벤트) / (비여름 경기당 이벤트)
    # - reliability = sqrt(g_summer * g_rest) / (sqrt(...) + k_games)
    g_s = games_per_season.get("Summer", pd.Series(dtype=float))
    g_r = games_per_season.get("Rest", pd.Series(dtype=float))
    e_s = events_per_season.get("Summer", pd.Series(dtype=float))
    e_r = events_per_season.get("Rest", pd.Series(dtype=float))

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_s = (e_s / g_s).replace([np.inf, -np.inf], np.nan)
        avg_r = (e_r / g_r).replace([np.inf, -np.inf], np.nan)
        raw_ratio = (avg_s / avg_r).replace([np.inf, -np.inf], np.nan)

    # 결측(한쪽 시즌 경기 0 등)은 중립 1.0으로
    raw_ratio = raw_ratio.fillna(1.0)

    # 소표본 보정(둘 중 하나 시즌이 0이면 n_eff=0 → 완전 중립)
    k_games = 6.0
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.sqrt(g_s.astype(float) * g_r.astype(float))
        reliability = (n_eff / (n_eff + k_games)).replace([np.inf, -np.inf], 0).fillna(0)

    p_fit = (reliability * raw_ratio) + ((1.0 - reliability) * 1.0)
    # 과도한 이상치는 캡(공모전용 안정화)
    p_fit = p_fit.clip(lower=0.5, upper=2.0)

    p_fit_scores = pd.DataFrame(
        {
            "player_id": p_fit.index,
            "p_fit_score": p_fit.values,
            "games_summer": g_s.reindex(p_fit.index).fillna(0).astype(int).values,
            "games_rest": g_r.reindex(p_fit.index).fillna(0).astype(int).values,
        }
    )

    # --- 3. 파울/카드 기반 클린플레이(반칙 리스크) 점수 계산 -> T-fit에 흡수 ---
    print("Calculating clean-play factor (foul/card) to be merged into T-fit...")
    fouls = raw_df[raw_df['type_name'] == 'Foul'].groupby('player_id').size()
    cards = raw_df[raw_df['type_name'].isin(['Yellow Card', 'Red Card', 'Yellow/Red Card'])].groupby('player_id').size()
    
    clean_play_df = pd.DataFrame({
        'fouls': fouls,
        'cards': cards
    }).fillna(0)

    # 경기당 파울/카드 레이트 기반(변별력↑): penalty_per_game = (fouls + 2*cards) / games
    clean_play_df = clean_play_df.join(games_played_total, how="outer").fillna(0)
    clean_play_df["penalty_events"] = clean_play_df["fouls"] + clean_play_df["cards"] * 2
    
    with np.errstate(divide="ignore", invalid="ignore"):
        penalty_per_game = (clean_play_df["penalty_events"] / clean_play_df["games_played_total"]).replace([np.inf, -np.inf], np.nan)
    penalty_per_game = penalty_per_game.fillna(0)

    # 스케일은 분포 기반(상위 25% 지점)으로 자동 설정 → 리그마다 튜닝 없이 동작
    try:
        scale = float(np.nanpercentile(penalty_per_game.values, 75))
    except Exception:
        scale = 2.0
    if not scale or scale <= 0:
        scale = 2.0

    # 1에 가까울수록 클린플레이(파울/카드 리스크 낮음) (0~1)
    clean_play_df["clean_play_score"] = (1.0 / (1.0 + (penalty_per_game / scale))).astype(float)
    clean_play_df["clean_play_score"] = clean_play_df["clean_play_score"].clip(lower=0.0, upper=1.0).fillna(0.0)
    clean_play_scores = clean_play_df[["clean_play_score"]].reset_index()

    # T-fit 고도화: 전방 압박(원래 T-fit)에 클린플레이 점수 반영
    # - 클린플레이 점수가 낮을수록(파울/카드 리스크 높음) 동일 압박량이라도 전술 실행 품질이 낮다고 가정
    t_fit_enhanced = pd.merge(t_fit_raw, clean_play_scores, on="player_id", how="right").fillna({"t_fit_raw": 0})
    t_fit_enhanced["t_fit_score"] = t_fit_enhanced["t_fit_raw"] * (0.7 + 0.3 * t_fit_enhanced["clean_play_score"])
    t_fit_scores = t_fit_enhanced[["player_id", "t_fit_score", "clean_play_score"]]

    # --- 4. C-fit placeholder ---
    print("Setting placeholder C-fit score (computed dynamically in app)...")
    c_fit_scores = players[["player_id"]].copy()
    c_fit_scores["c_fit_score"] = 0.85

    # --- 5. 모든 지표 병합 ---
    print("Merging all HSI metrics...")
    hsi_df = pd.merge(players, t_fit_scores, on='player_id', how='left')
    hsi_df = pd.merge(hsi_df, p_fit_scores, on='player_id', how='left')
    hsi_df = pd.merge(hsi_df, c_fit_scores, on='player_id', how='left')
    # p_fit_score 결측치는 1(중립)로, 나머지는 0으로
    hsi_df['p_fit_score'] = hsi_df['p_fit_score'].fillna(1.0)
    hsi_df = hsi_df.fillna(0)

    # --- 6. 파일 저장 ---
    output_path = os.path.join(output_dir, 'hsi_scores_2024.csv')
    hsi_df.to_csv(output_path, index=False)
    print(f"HSI scores for all players saved to {output_path}")


if __name__ == '__main__':
    RAW_DATA_PATH = 'data/raw/raw_data.csv'
    MATCH_INFO_PATH = 'data/raw/match_info.csv'
    OUTPUT_DIR = 'output'
    calculate_hsi_metrics(RAW_DATA_PATH, MATCH_INFO_PATH, OUTPUT_DIR)
