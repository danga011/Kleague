import pandas as pd
import json
import os

def group_position(pos):
    """Helper function to group detailed positions into FW, MF, DF, ETC."""
    if pos in ['CF', 'RW', 'LW', 'SS']:
        return 'FW'
    if pos in ['CM', 'DM', 'AM', 'LM', 'RM']:
        return 'MF'
    if pos in ['CB', 'LB', 'RB', 'LWB', 'RWB']:
        return 'DF'
    return 'ETC'

def create_team_profiles_by_position(raw_data_path, hsi_scores_path, output_dir):
    """
    K리그 모든 팀에 대해, 포지션 그룹(FW, MF, DF)별 평균 HSI 점수를 계산하여
    '팀별/포지션별 전술 템플릿'을 생성합니다.
    """
    try:
        raw_df = pd.read_csv(raw_data_path, usecols=['player_id', 'team_name_ko', 'main_position'])
        hsi_df = pd.read_csv(hsi_scores_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
        return

    print("Profiling each team's tactical style by position group...")

    # 1. 각 선수의 주 소속팀과 주 포지션 그룹 할당
    player_main_team = raw_df.groupby('player_id')['team_name_ko'].agg(lambda x: x.mode().iloc[0]).reset_index()
    player_main_pos = raw_df.groupby('player_id')['main_position'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown').reset_index()
    
    # HSI 점수와 팀/포지션 정보 병합
    hsi_with_details = pd.merge(hsi_df, player_main_team, on='player_id')
    hsi_with_details = pd.merge(hsi_with_details, player_main_pos, on='player_id')
    hsi_with_details['pos_group'] = hsi_with_details['main_position'].apply(group_position)

    # 2. 팀별, 포지션 그룹별 평균 HSI 점수 계산
    team_pos_templates = hsi_with_details.groupby(['team_name_ko', 'pos_group'])[['t_fit_score', 'p_fit_score', 'c_fit_score']].mean()
    
    # 3. 최종 JSON 구조로 변환
    final_templates = {}
    for (team, pos), scores in team_pos_templates.iterrows():
        if team not in final_templates:
            final_templates[team] = {}
        final_templates[team][pos] = scores.to_dict()

    # 4. 파일 저장
    output_path = os.path.join(output_dir, 'team_templates.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_templates, f, ensure_ascii=False, indent=4)
        
    print(f"Team tactical templates by position group saved to {output_path}")

if __name__ == '__main__':
    RAW_DATA_PATH = 'data/raw/raw_data.csv'
    HSI_SCORES_PATH = 'output/hsi_scores_2024.csv'
    OUTPUT_DIR = 'output'
    create_team_profiles_by_position(RAW_DATA_PATH, HSI_SCORES_PATH, OUTPUT_DIR)