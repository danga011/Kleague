# 데이터 수집 가이드 (Model Dev 시작 전)

> 이 문서는 “논문 기반 축구 분석 모델” 개발을 시작하기 위해, **어떤 데이터가 필요하고(스키마)**, **어디에 저장하고**, **어떻게 누락을 점검할지**를 정리합니다.

---

## 1) 지금 repo에 이미 있는 데이터 (현재 MVP 기준)

### **필수(이미 존재; MVP 동작)**
- **이벤트 데이터**: `data/raw/raw_data.csv`
  - 기대 컬럼(예): `game_id, action_id, period_id, time_seconds, team_id, player_id, type_name, start_x, start_y, end_x, end_y, player_name_ko, team_name_ko, main_position`
- **경기 메타 데이터**: `data/raw/match_info.csv`
  - 기대 컬럼(예): `game_id, game_date, home_team_name_ko, away_team_name_ko, venue, competition_name`
- **WVS 도시 벡터(집계 결과; C-Fit 사용 시 필요)**: `data/processed/wvs_city_vectors.csv`
  - 생성 스크립트: `python src/wvs_city_profiler.py`

### **권장(재현/재생성용)**
- **WVS 원본(도시/로케이션 기반 문화지표)**: `data/raw/WVS_Time_Series_1981-2022_csv_v5_0.csv`
  - 용량/라이선스 이슈로 **제출물에 포함하지 않을 수 있음** → 대신 `wvs_city_vectors.csv`(가공 결과) + 출처/라이선스 표기를 권장
  - 현재 repo에서는 용량 경량화를 위해 원본 파일을 제외할 수 있으며, 필요 시 WVS 공식 출처에서 다운로드 후 해당 경로에 배치

### **매핑(필요시 채워야 함)**
- **선수 성장도시 매핑**: `data/processed/player_upbringing_city_map.csv`
  - `player_name_ko, home_country_alpha, home_loc_code(E179_WVS7LOC), home_city_label`
- **구단(분석도시/연고지) 매핑**: `data/processed/kleague_team_city_map.csv`
  - `team_name_ko, host_country_alpha, host_loc_code(E179_WVS7LOC), host_city_label`

---

## 2) 모델 개발을 위해 “추가로” 수집이 필요한 데이터(추천)

모델 목표에 따라 달라집니다. 아래는 논문 기반 분석에서 자주 요구되는 최소 세트입니다.

### A. 선수 바이오/프로필 (리포트 템플릿 1번 채우기 필수)
- 선수 ID ↔ 이름 매핑(이미 있음) 외에:
  - **소속 클럽(시즌별)**, **주발**, **신장/체중**, **출생년도/나이**, **국적**
- 권장 저장:
  - `data/raw/player_profile.csv`  ✅ (템플릿 생성 가능)
  - 최소 컬럼: `player_id, player_name_ko, club_name_ko, nationality, preferred_foot, height_cm, weight_kg, birth_year`

템플릿 생성(현재 raw_data.csv 기반으로 선수 목록 자동 채움):

```bash
python src/init_data_templates.py
```

### B. 출전시간/라인업/교체 (모든 “경기당/90분당” 정규화에 필요)
- 권장 저장:
  - `data/raw/player_minutes_by_match.csv` ✅ (템플릿 생성 가능)
  - 최소 컬럼: `game_id, player_id, team_name_ko, minutes_played, is_starter, position_name, sub_in_minute, sub_out_minute`

### C. 슛/득점 기대값(xG)용 이벤트 필드(있으면 강력)
목표가 xG/공격력 중심이면 필요:
- 슛 위치, 바디파트, 상황(오픈플레이/세트피스), 어시스트 유형 등
- 현재 `raw_data.csv`에 해당 필드가 없다면 별도 공급원 필요

### D. 팀 스타일/컨텍스트(전술 역할 요약 강화)
- PPDA, 점유율, 라인 높이, 전환 빈도, 롱볼 비중 등
- 없으면 논문 지표 구현에 제약 → 단계적으로 확장 권장

---

## 3) 데이터 수집 소스(선택)

K리그 이벤트/메타 데이터는 보통 유료 데이터 공급원이 많습니다. 현재 repo에 2024 시즌 이벤트가 이미 있으므로, 아래는 “추가 보강” 관점입니다.

- **선수 프로필/바이오**: K리그 공식, Transfermarkt, Wikipedia, 구단 프로필 등(라이선스/크롤링 정책 확인 필요)
- **출전시간/라인업**: 리그 공식/데이터 공급원
- **날씨/경기장 정보**: 기상청/오픈 API(날짜/장소 기반)
- **과거 시즌(학습 데이터 확대)**: 2020~2024 같은 멀티시즌 확보 권장

---

## 4) “수집 완료” 체크(자동 점검)

수집 누락을 빠르게 파악하려면 다음을 사용하세요:

```bash
python src/data_audit.py
```

이 스크립트는:
- 필수 파일 존재 여부
- 핵심 컬럼 존재 여부
- (선택) 매핑 파일이 비어있는지 여부
 - (선택) 선수 프로필/출전시간 파일 존재 여부
를 요약 출력합니다.


