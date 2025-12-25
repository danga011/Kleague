
import pandas as pd
import os

def extract_player_data(raw_data_path, output_dir):
    """
    raw_data.csv에서 선수 명단과 기본 스탯을 추출하여
    두 개의 CSV 파일로 저장합니다.
    """
    print(f"Reading data from {raw_data_path}...")
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"Error: The file {raw_data_path} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    # --- 1. 선수 전체 명단 추출 ---
    players_df = df[['player_id', 'player_name_ko']].drop_duplicates().sort_values(by='player_id')
    
    # --- 2. 선수별 기본 스탯 요약 ---
    print("Calculating summary statistics...")
    
    # 기본 집계
    stats_summary = df.groupby(['player_id', 'player_name_ko']).agg(
        total_events=pd.NamedAgg(column='action_id', aggfunc='count')
    )
    
    # 유형별 집계
    def count_type(df, type_name):
        return df[df['type_name'] == type_name].groupby(['player_id', 'player_name_ko']).size()

    def count_types(df, type_names):
        return df[df['type_name'].isin(type_names)].groupby(['player_id', 'player_name_ko']).size()

    stats_summary['total_duels'] = count_type(df, 'Duel')
    stats_summary['total_fouls'] = count_type(df, 'Foul')
    stats_summary['total_cards'] = count_types(df, ['Red Card', 'Yellow Card', 'Yellow/Red Card'])

    # 집계 결과 병합 및 결측치 처리
    stats_summary = stats_summary.fillna(0).astype(int).reset_index()

    # --- 3. 파일 저장 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    players_output_path = os.path.join(output_dir, 'players.csv')
    stats_output_path = os.path.join(output_dir, 'player_summary_stats.csv')

    players_df.to_csv(players_output_path, index=False)
    print(f"Player list saved to {players_output_path}")

    stats_summary.to_csv(stats_output_path, index=False)
    print(f"Player summary stats saved to {stats_output_path}")


if __name__ == '__main__':
    # 현재 kleague 디렉토리 구조에 맞게 경로 설정
    RAW_DATA_PATH = 'data/raw/raw_data.csv'
    OUTPUT_DIR = 'output'
    extract_player_data(RAW_DATA_PATH, OUTPUT_DIR)

