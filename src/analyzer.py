
import pandas as pd
import os

# 웹 검색을 통해 얻은 2024년 K리그 주요 외국인 선수 이름 리스트
# 이 리스트는 완벽하지 않으며, 데이터에 있는 선수와 매칭하기 위한 기반으로 사용됩니다.
# 실제 국적 정보는 이 단계에서는 알 수 없으므로, 'Foreign'으로 표기하고 추후 수동 보강이 필요합니다.
FOREIGN_PLAYER_NAMES_KO = [
    '제시 린가드', '켈빈', '마틴 아담', '루빅손', '보야니치', '아타루', 
    '비니시우스', '보아텡', '티아고', '완델손', '오베르단', '제카', '그랜트', 
    '탈레스', '이탈로', '유리', '헤이스', '아사니', '베카', '가브리엘', 
    '빅톨', '브루노', '포포비치', '아르한', '잭슨', '무고사', '제르소', 
    '에르난데스', '델브리지', '음포쿠', '세징야', '에드가', '바셀루스', 
    '벨톨라', '안톤', '구텍', '강투지', '오두', '미유키', '유키야', '일류첸코'
]

def analyze_and_select_performers(players_path, stats_path, output_dir):
    """
    선수 스탯을 분석하여 외국인 선수 리스트와 Top Performer 리스트를 생성합니다.
    """
    try:
        players_df = pd.read_csv(players_path)
        stats_df = pd.read_csv(stats_path)
        raw_df = pd.read_csv('data/raw/raw_data.csv', usecols=['player_id', 'main_position'])
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
        return

    # --- 1. 외국인 선수 식별 ---
    # 검색된 이름 리스트를 기반으로 데이터에 있는 외국인 선수 필터링
    foreigners_df = players_df[players_df['player_name_ko'].isin(FOREIGN_PLAYER_NAMES_KO)].copy()
    foreigners_df['nationality'] = 'Foreign' # 국적은 'Foreign'으로 우선 표기

    # --- 2. 선수별 주 포지션 추출 ---
    # 가장 많이 등장하는 포지션을 주 포지션으로 설정
    main_pos_df = raw_df.groupby('player_id')['main_position'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown').reset_index()
    
    # 외국인 선수 정보에 포지션 병합
    foreigners_df = pd.merge(foreigners_df, main_pos_df, on='player_id')
    
    # --- 3. Top Performer 선정 ---
    # 분석을 위해 스탯 데이터와 포지션 정보 병합
    stats_with_pos = pd.merge(stats_df, main_pos_df, on='player_id')
    
    # 외국인 선수 스탯만 필터링
    foreign_stats_df = stats_with_pos[stats_with_pos['player_id'].isin(foreigners_df['player_id'])]

    # 포지션별 Performance Score 계산
    # FW: 이벤트 수에 가중치
    # MF: 이벤트와 경합 수에 가중치
    # DF: 경합 수에 가중치, 파울/카드에 페널티
    
    scores = []
    for _, row in foreign_stats_df.iterrows():
        score = 0
        events = row['total_events']
        duels = row['total_duels']
        fouls = row['total_fouls']
        cards = row['total_cards']
        
        # 이벤트가 0인 경우 점수 계산에서 제외 (활동이 거의 없는 선수)
        if events == 0:
            continue

        if row['main_position'] == 'Forward':
            score = (events * 1.2 + duels * 0.8) / (1 + fouls * 0.1 + cards * 0.2)
        elif row['main_position'] == 'Midfielder':
            score = (events * 1.0 + duels * 1.0) / (1 + fouls * 0.1 + cards * 0.2)
        elif row['main_position'] == 'Defender':
            score = (events * 0.8 + duels * 1.2) / (1 + fouls * 0.1 + cards * 0.2)
        else:
            score = (events * 1.0 + duels * 1.0) / (1 + fouls * 0.1 + cards * 0.2)
            
        scores.append({'player_id': row['player_id'], 'player_name_ko': row['player_name_ko'], 'position': row['main_position'], 'performance_score': score})

    performance_df = pd.DataFrame(scores)
    
    # 전체 Top 10 선정
    top_performers_df = performance_df.sort_values(by='performance_score', ascending=False).head(10)

    # --- 4. 파일 저장 ---
    foreigners_output_path = os.path.join(output_dir, '2024_foreigners_list.csv')
    top_performers_output_path = os.path.join(output_dir, '2024_top_performers.csv')
    
    # 필요한 컬럼만 선택하여 저장
    foreigners_to_save = foreigners_df[['player_name_ko', 'nationality', 'main_position']]
    top_performers_to_save = top_performers_df[['player_name_ko', 'position', 'performance_score']]

    foreigners_to_save.to_csv(foreigners_output_path, index=False)
    print(f"Foreign players list saved to {foreigners_output_path}")

    top_performers_to_save.to_csv(top_performers_output_path, index=False)
    print(f"Top 10 performers list saved to {top_performers_output_path}")


if __name__ == '__main__':
    PLAYERS_PATH = 'output/players.csv'
    STATS_PATH = 'output/player_summary_stats.csv'
    OUTPUT_DIR = 'output'
    analyze_and_select_performers(PLAYERS_PATH, STATS_PATH, OUTPUT_DIR)
