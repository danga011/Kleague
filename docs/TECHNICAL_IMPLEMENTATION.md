# K-Scout Adapt-Fit AI 기술 구현 상세 문서

## 목차
1. [시스템 개요](#시스템-개요)
2. [HSI (Harmonic Synergy Index) 지표 체계](#hsi-harmonic-synergy-index-지표-체계)
3. [T-Fit (전술 적합도) 계산](#t-fit-전술-적합도-계산)
4. [P-Fit (환경 적합도) 계산](#p-fit-환경-적합도-계산)
5. [C-Fit (문화 적합도) 계산](#c-fit-문화-적합도-계산)
6. [통계적 안정화 기법](#통계적-안정화-기법)
7. [Adapt-Fit Score 계산](#adapt-fit-score-계산)
8. [AI 분석 시스템](#ai-분석-시스템)
9. [데이터 파이프라인](#데이터-파이프라인)
10. [성능 최적화](#성능-최적화)

---

## 시스템 개요

K-Scout Adapt-Fit AI는 K리그 외국인 선수의 적합도를 다차원적으로 평가하는 시스템입니다. 세 가지 핵심 지표(T-Fit, P-Fit, C-Fit)를 통해 선수의 전술적, 환경적, 문화적 적응 가능성을 정량화합니다.

### 핵심 철학

1. **데이터 기반 의사결정**: 주관적 판단을 최소화하고 객관적 지표를 활용
2. **통계적 견고성**: 베이지안 스무딩과 퍼센타일 변환을 통한 소표본 안정화
3. **맥락적 적합도**: 팀별 전술 템플릿과의 시너지 평가
4. **다차원 분석**: 전술, 환경, 문화의 3개 축을 독립적으로 평가

---

## HSI (Harmonic Synergy Index) 지표 체계

HSI는 선수의 종합 적합도를 나타내는 복합 지표로, 세 개의 하위 지표로 구성됩니다.

### 구성 요소

```
HSI = { T-Fit, P-Fit, C-Fit }

T-Fit: Tactical Fit (전술 적합도)
P-Fit: Physical/Environmental Fit (환경 적합도)
C-Fit: Cultural Fit (문화 적합도)
```

### 척도 체계

- **T-Fit**: 0-100 퍼센타일 (리그 내 상대적 순위)
- **P-Fit**: 0-100 퍼센타일 (리그 내 상대적 순위)
- **C-Fit**: 0-1 유사도 점수 (코사인 유사도 기반)

---

## T-Fit (전술 적합도) 계산

T-Fit은 선수의 전방 압박 능력과 클린플레이 수준을 평가합니다.

### 1단계: 전방 압박 비중 계산

#### 수식
```
high_press_count = Σ(defensive_actions where start_x > 60)
total_def_count = Σ(defensive_actions)

press_ratio = high_press_count / total_def_count

defensive_actions ∈ {Duel, Tackle, Interception}
```

#### 구현 상세
```python
# src/hsi_calculator.py Line 76-80

defensive_actions = ['Duel', 'Tackle', 'Interception']
def_events = raw_df[raw_df['type_name'].isin(defensive_actions)]

high_press_counts = def_events[def_events['start_x'] > 60].groupby("player_id").size()
total_def_counts = def_events.groupby("player_id").size()
press_ratio = high_press_counts / total_def_counts
```

#### 좌표 체계
- 피치 X축: 0-120 (상대 골대 방향)
- X > 60: 상대 진영 전방
- 전방 압박: 상대 진영에서 이루어진 수비 액션

### 2단계: 베이지안 스무딩 적용

소표본 선수(출전 경기 수가 적은 선수)의 점수를 안정화하기 위해 베이지안 추정을 사용합니다.

#### 수식
```
t_fit_smoothed = (x_i * n_i + μ_global * k) / (n_i + k)

여기서:
x_i = 선수 i의 전방 압박 카운트
n_i = 선수 i의 출전 경기 수
μ_global = 리그 전체 평균 (경기당 전방 압박)
k = prior strength (사전 강도) = 5.0
```

#### 이론적 배경
베이지안 스무딩은 개별 선수의 관측값과 전체 평균을 가중 평균하여, 데이터가 적은 선수의 추정치를 전체 평균 쪽으로 회귀시킵니다(shrinkage). 이는 과적합을 방지하고 추정의 분산을 줄입니다.

#### 하이퍼파라미터 선정
```
k = 5.0
```
- k가 클수록: 전체 평균에 더 가까워짐 (보수적)
- k가 작을수록: 개별 관측값에 더 의존 (공격적)
- k=5는 약 5경기의 데이터와 전체 평균을 동등하게 취급

#### 구현 상세
```python
# src/hsi_calculator.py Line 38-40

def bayesian_smoothing(raw_values, n_samples, global_mean, k=5.0):
    return (raw_values * n_samples + global_mean * k) / (n_samples + k)

# Line 90-96
league_avg_t = high_press_counts.sum() / games_played.sum()
k_t = 5.0

t_fit_raw = bayesian_smoothing(
    high_press_counts.reindex(games_played.index).fillna(0),
    games_played,
    league_avg_t,
    k=k_t
)
```

### 3단계: 클린플레이 보정

반칙 리스크가 높은 선수는 동일한 압박량이라도 전술 실행의 질이 낮다고 가정합니다.

#### 수식
```
penalty_per_game = (fouls + cards * 2) / games_played

cp_scale = Q_75(penalty_per_game)  // 75 퍼센타일

clean_play_score = 1 / (1 + penalty_per_game / cp_scale)

t_fit_final = t_fit_raw * (0.7 + 0.3 * clean_play_score)
```

#### 가중치 설명
- 기본 전방 압박: 70%
- 클린플레이 보너스: 최대 30%
- 반칙이 많은 선수는 최종 점수가 70%까지 감소
- 반칙이 적은 선수는 최종 점수 유지

#### 스케일링 방법
75 퍼센타일을 기준으로 정규화하여 이상치의 영향을 줄입니다. 이는 중앙값보다 높은 기준을 사용하여 클린플레이를 강조합니다.

#### 구현 상세
```python
# src/hsi_calculator.py Line 99-104

fouls = raw_df[raw_df['type_name'] == 'Foul'].groupby('player_id').size()
cards = raw_df[raw_df['type_name'].isin(['Yellow Card', 'Red Card', 'Yellow/Red Card'])].groupby('player_id').size()
penalty_per_game = ((fouls.reindex(games_played.index).fillna(0) + 
                    cards.reindex(games_played.index).fillna(0) * 2) / games_played).fillna(0)

cp_scale = penalty_per_game.quantile(0.75) if penalty_per_game.quantile(0.75) > 0 else 1.0
clean_play_score = 1.0 / (1.0 + (penalty_per_game / cp_scale))

t_fit_final_raw = t_fit_raw * (0.7 + 0.3 * clean_play_score)
```

### 4단계: 퍼센타일 변환

리그 내 상대적 순위로 변환하여 절대값의 의미를 명확히 합니다.

#### 수식
```
T-Fit_percentile = rank(t_fit_final) / N * 100

여기서:
rank() = 오름차순 순위 함수
N = 전체 선수 수
```

#### 해석
- T-Fit = 90: 상위 10% (리그 내 매우 뛰어난 전방 압박)
- T-Fit = 50: 중위권 (리그 평균 수준)
- T-Fit = 10: 하위 10% (전방 압박이 약함)

#### 구현 상세
```python
# src/hsi_calculator.py Line 43-45

def to_percentile(series):
    """원본 점수를 백분위 점수(0-100)로 변환"""
    return series.rank(pct=True) * 100.0

# Line 142
hsi_results['t_fit_score'] = to_percentile(t_fit_final_raw).reindex(hsi_results['player_id']).fillna(50.0).values
```

### 수비 스타일 분류 (보조 인사이트)

전방 압박 비중에 따른 스타일 분류:

```python
# src/hsi_calculator.py Line 161-167

if press_intensity > 0.5:
    press_style = "강력한 전방 압박 수행 (High-Presser)"
elif press_intensity > 0.3:
    press_style = "적극적 중원 압박 (Medium-Presser)"
else:
    press_style = "지역 방어 및 블록 형성 (Positional)"
```

---

## P-Fit (환경 적합도) 계산

P-Fit은 K리그 혹서기(6-8월)에 대한 선수의 활동량 유지 능력을 평가합니다.

### 1단계: 시즌 구분

#### 수식
```
is_summer = month ∈ {6, 7, 8}

summer_season = {games played in June, July, August}
rest_season = {games played in other months}
```

#### 구현 상세
```python
# src/hsi_calculator.py Line 114-116

merged_df = pd.merge(raw_df, match_df[['game_id', 'game_date']], on='game_id')
merged_df['game_date'] = pd.to_datetime(merged_df['game_date'])
merged_df['is_summer'] = merged_df['game_date'].dt.month.isin([6, 7, 8])
```

### 2단계: 경기당 이벤트 계산

#### 수식
```
events_per_game_summer = Σ(events_summer) / n_games_summer
events_per_game_rest = Σ(events_rest) / n_games_rest

retention_ratio = events_per_game_summer / events_per_game_rest
```

#### 이벤트 정의
모든 기록된 플레이 액션:
- 패스, 슈팅, 드리블
- 수비 액션 (태클, 인터셉트, 듀얼)
- 파울, 오프사이드 등

경기당 이벤트 수는 선수의 활동량을 나타내는 대리 지표(proxy)입니다.

#### 구현 상세
```python
# src/hsi_calculator.py Line 119-128

summer_games = merged_df[merged_df['is_summer']].groupby('player_id')['game_id'].nunique()
rest_games = merged_df[~merged_df['is_summer']].groupby('player_id')['game_id'].nunique()

summer_events = merged_df[merged_df['is_summer']].groupby('player_id').size()
rest_events = merged_df[~merged_df['is_summer']].groupby('player_id').size()

summer_per_game = (summer_events / summer_games).fillna(0)
rest_per_game = (rest_events / rest_games).fillna(0)

p_retention = (summer_per_game / (rest_per_game + 1e-6)).fillna(1.0)
```

### 3단계: 베이지안 스무딩

여름 경기 수를 기반으로 스무딩을 적용합니다.

#### 수식
```
p_fit_smoothed = (retention_i * n_summer_i + 1.0 * k) / (n_summer_i + k)

여기서:
retention_i = 선수 i의 여름철 유지율
n_summer_i = 선수 i의 여름 경기 수
k = 3.0 (T-Fit보다 작은 값, 여름 데이터가 희소하므로)
1.0 = 중립 유지율 (100%)
```

#### 하이퍼파라미터 선정
```
k = 3.0
```
- T-Fit의 k=5.0보다 작음
- 여름 경기가 전체의 1/4 정도이므로 더 적은 prior를 사용
- 약 3경기의 여름 데이터와 중립 가정을 동등하게 취급

#### 구현 상세
```python
# src/hsi_calculator.py Line 131-139

league_avg_p = 1.0  # 중립 = 유지율 100%
k_p = 3.0

p_fit_raw = bayesian_smoothing(
    p_retention.reindex(games_played.index).fillna(1.0),
    summer_games.reindex(games_played.index).fillna(0),
    league_avg_p,
    k=k_p
)
```

### 4단계: 퍼센타일 변환

#### 수식
```
P-Fit_percentile = rank(p_fit_raw) / N * 100
```

#### 해석
- P-Fit = 90: 여름철 활동량이 리그 상위 10% (혹서기에 강함)
- P-Fit = 50: 중위권 (평균적 유지)
- P-Fit = 10: 하위 10% (혹서기에 활동량 저하)

#### 구현 상세
```python
# src/hsi_calculator.py Line 143
hsi_results['p_fit_score'] = to_percentile(p_fit_raw).reindex(hsi_results['player_id']).fillna(50.0).values
```

### 여름철 프로필 분류 (보조 인사이트)

```python
# src/hsi_calculator.py Line 174-180

if summer_ret >= 1.05:
    summer_profile = f"여름철 성능 향상 ({summer_ret*100:.1f}%, 혹서기 강점)"
elif summer_ret >= 0.95:
    summer_profile = f"여름철 안정적 유지 ({summer_ret*100:.1f}%)"
else:
    summer_profile = f"여름철 성능 저하 ({summer_ret*100:.1f}%, 로테이션 고려)"
```

---

## C-Fit (문화 적합도) 계산

C-Fit은 선수의 출신 도시와 구단 소재 도시 간의 문화적 유사도를 평가합니다.

### 이론적 배경

World Values Survey (WVS) 데이터를 사용하여 도시별 문화 벡터를 구축합니다. WVS는 다음 6개 차원을 측정합니다:

1. Traditional vs Secular-Rational Values
2. Survival vs Self-Expression Values
3. Family Ties
4. Respect for Authority
5. Trust in People
6. Importance of Religion

### 1단계: 도시 벡터 로드

#### 데이터 구조
```
wvs_city_vectors.csv:
  - city_code: 도시 고유 코드
  - dim_1, dim_2, ..., dim_6: WVS 6차원 값

player_upbringing_city_map.csv:
  - player_name: 선수 이름
  - upbringing_city: 성장 도시 코드

kleague_team_city_map.csv:
  - team_name: 구단 이름
  - host_city: 홈 도시 코드
```

#### 구현 위치
```python
# src/wvs_city_fit.py Line 15-30

def load_wvs_city_vectors_and_stats():
    vectors_df = pd.read_csv(WVS_CITY_VECTORS_PATH)
    loc_labels_df = pd.read_csv(WVS_LOC_LABELS_PATH)
    city_vectors = dict(zip(vectors_df['city_code'], vectors_df[['dim_1', 'dim_2', ...]].values))
    return city_vectors, loc_labels_df, global_mean_vector
```

### 2단계: 코사인 유사도 계산

#### 수식
```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)

여기서:
A = 선수 출신 도시의 문화 벡터 (6차원)
B = 구단 소재 도시의 문화 벡터 (6차원)
A · B = 내적 (dot product)
||A|| = A의 L2 노름
```

#### 정규화
```
normalized_vector = vector / sqrt(Σ(vector_i^2))
```

모든 벡터를 단위 벡터로 정규화한 후 내적을 계산합니다.

#### 범위
```
-1 ≤ cosine_similarity ≤ 1

1.0: 완전히 동일한 문화
0.0: 직교 (무관한 문화)
-1.0: 정반대 문화
```

실제로는 대부분 0.7-1.0 범위에 분포합니다.

#### 구현 상세
```python
# src/wvs_city_fit.py Line 50-60

def compute_wvs_city_c_fit(player_name, team_name, city_vectors, ...):
    player_vector = city_vectors.get(player_city_code)
    host_vector = city_vectors.get(host_city_code)
    
    # 벡터 정규화
    norm_player = np.linalg.norm(player_vector)
    norm_host = np.linalg.norm(host_vector)
    
    # 코사인 유사도
    c_fit_score = float(np.dot(player_vector, host_vector) / (norm_player * norm_host))
    
    return c_fit_score
```

### 3단계: Fallback 전략

도시 데이터가 없는 경우 계층적 fallback을 적용합니다:

```
1순위: 도시 코드 → 도시 벡터
2순위: 도시 없음 → 국가 코드 → 국가 평균 벡터
3순위: 국가 없음 → 글로벌 평균 벡터
```

#### 구현 상세
```python
# src/wvs_city_fit.py Line 65-85

if player_city_code not in city_vectors:
    # 국가 평균으로 fallback
    player_country = player_info.get('country')
    if player_country in country_avg_vectors:
        player_vector = country_avg_vectors[player_country]
        reason = "국가 평균 사용"
    else:
        # 글로벌 평균으로 fallback
        player_vector = global_mean_vector
        reason = "글로벌 평균 사용"
```

### C-Fit 스케일링 (UI/PDF 표시용)

C-Fit은 0-1 범위이므로 차트 표시 시 100을 곱합니다.

```python
# app.py Line 1389-1391

c_fit_chart = player_hsi['c_fit_score'] * 100  # 0-1 → 0-100
```

---

## 통계적 안정화 기법

### 베이지안 스무딩 상세

#### 수학적 유도

경험적 베이즈(Empirical Bayes) 추정:

```
posterior_mean = (likelihood * prior) / evidence

단순화:
θ_i ≈ (n_i * x̄_i + k * μ) / (n_i + k)

여기서:
θ_i = 선수 i의 추정 모수
n_i = 선수 i의 샘플 크기
x̄_i = 선수 i의 표본 평균
k = 가상 샘플 크기 (prior strength)
μ = 전체 평균 (prior mean)
```

#### 분산 감소 효과

```
Var(θ_i) = Var(x̄_i) * (n_i / (n_i + k))^2

n_i가 작을수록 분산이 크게 감소
n_i가 클수록 원래 추정값 유지
```

#### 편향-분산 트레이드오프

- **Bias (편향)**: 스무딩으로 인해 전체 평균 쪽으로 편향
- **Variance (분산)**: 소표본의 높은 분산 감소
- **MSE**: 소표본에서는 분산 감소 > 편향 증가, 따라서 MSE 감소

### 퍼센타일 변환의 이점

#### 1. 이상치 견고성
```
원본 점수: [1, 2, 3, 4, 100]  → 평균 22 (이상치에 민감)
퍼센타일: [20, 40, 60, 80, 100] → 해석 용이
```

#### 2. 척도 불변성
```
점수 A: 0-5 범위
점수 B: 0-100 범위
→ 퍼센타일 변환 후 모두 0-100으로 정규화
```

#### 3. 순위 기반 해석
```
T-Fit = 75 → "상위 25% 선수"
직관적이고 비교 가능
```

---

## Adapt-Fit Score 계산

Adapt-Fit Score는 선수의 HSI 지표와 팀 템플릿 간의 방향성 유사도를 측정합니다.

### 방향성 유사도 (Directional Similarity)

일반적인 유사도와 달리, 선수가 팀 평균을 초과하는 경우 보너스를, 미달하는 경우 페널티를 부여합니다.

#### 수식
```
directional_similarity(x_player, x_team, weight) = 
  weight * [1 + α * (x_player - x_team) / scale]

여기서:
α = +0.5 if x_player > x_team (보너스)
    -0.5 if x_player < x_team (페널티)

scale = 표준화 스케일 (100 for T-Fit, P-Fit; 0.3 for C-Fit)
```

#### 비대칭성
```
x_player = 80, x_team = 60:
  → score = 1 + 0.5 * (20/100) = 1.1 (10% 보너스)

x_player = 60, x_team = 80:
  → score = 1 - 0.5 * (20/100) = 0.9 (10% 페널티)
```

선수가 팀 평균보다 높으면 시너지, 낮으면 약점으로 평가합니다.

### 포지션별 가중치

#### 가중치 행렬
```
Position    T-Fit    P-Fit    C-Fit
FW          0.35     0.35     0.30
MF          0.40     0.30     0.30
DF          0.45     0.25     0.30
```

#### 논리적 근거
- **수비수 (DF)**: 전술 적합도 중요 (팀 수비 전술에 맞아야 함)
- **미드필더 (MF)**: 전술과 체력의 균형
- **공격수 (FW)**: 체력과 전술의 균형 (고강도 압박 + 혹서기 지구력)

### 최종 점수 계산

#### 수식
```
Adapt-Fit Score = 
  w_t * directional_sim(T-Fit_player, T-Fit_team, 1.0) +
  w_p * directional_sim(P-Fit_player, P-Fit_team, 1.0) +
  w_c * directional_sim(C-Fit_player, C-Fit_team, 1.0)

정규화:
final_score = Adapt-Fit Score * 100 / (w_t + w_p + w_c)
```

최종 점수는 0-100 범위로 정규화됩니다.

#### 구현 상세
```python
# app.py Line 353-385

def calculate_adapt_fit_score(player_hsi, team_template, pos_group):
    weights = {
        'FW': {'t': 0.35, 'p': 0.35, 'c': 0.30},
        'MF': {'t': 0.40, 'p': 0.30, 'c': 0.30},
        'DF': {'t': 0.45, 'p': 0.25, 'c': 0.30}
    }
    
    w = weights.get(pos_group, {'t': 0.35, 'p': 0.35, 'c': 0.30})
    
    # T-Fit, P-Fit: 퍼센타일 비교
    t_contrib = w['t'] * directional_similarity(
        player_hsi['t_fit_score'], 
        team_template['t_fit_score'], 
        scale=100
    )
    
    p_contrib = w['p'] * directional_similarity(
        player_hsi['p_fit_score'], 
        team_template['p_fit_score'], 
        scale=100
    )
    
    # C-Fit: 0-1 스케일 비교
    c_contrib = w['c'] * directional_similarity(
        player_hsi['c_fit_score'], 
        team_template['c_fit_score'], 
        scale=0.3
    )
    
    raw_score = t_contrib + p_contrib + c_contrib
    final_score = (raw_score / (w['t'] + w['p'] + w['c'])) * 100
    
    return np.clip(final_score, 0, 100)
```

### 등급 분류

```python
# app.py Line 424-433

def get_grade(score):
    if score >= 85: return 'S'
    if score >= 75: return 'A'
    if score >= 65: return 'B'
    if score >= 55: return 'C'
    if score >= 45: return 'D'
    return 'E'
```

---

## AI 분석 시스템

### Google Gemini 1.5 Flash 통합

#### 모델 선택
```
Primary: gemini-1.5-flash
Fallback: gemini-pro

gemini-1.5-flash 선택 이유:
- 빠른 응답 속도 (5-10초)
- 긴 컨텍스트 지원 (128K 토큰)
- 무료 티어 제공
- 한국어 지원
```

#### API 설정
```python
# app.py Line 43-47

if GENAI_AVAILABLE:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
```

### 토큰 할당 전략

#### 설정
```python
# app.py Line 900-905

generation_config=genai.types.GenerationConfig(
    temperature=0.7,
    max_output_tokens=8192,  # 최대 8K 토큰
)
```

#### 토큰 경제학
```
입력 프롬프트: ~2,500 토큰
출력: 8,192 토큰
총: ~10,700 토큰/보고서

일일 무료 한도: 1,500,000 토큰
가능 보고서 수: ~140개/일
```

### 프롬프트 엔지니어링

#### 구조
```
1. 역할 정의: 유럽 5대 리그 스카우팅 디렉터
2. 작성 형식: 보고서 헤더 (팀명, 선수명, 날짜)
3. 데이터 제공: HSI 점수, 선수 인사이트, 팀 템플릿
4. 작성 지침: 
   - 기술적 세부사항 제외
   - 정량적 근거 제시
   - 실무적 제안
   - 전문 용어 사용
5. 출력 형식: 구조화된 섹션
```

#### 핵심 지침
```python
# app.py Line 659-720

주의사항:
- 기술적인 계산 과정이나 데이터 매핑 정책은 언급하지 마세요
- C-Fit 계산 방법, WVS 데이터 출처, fallback 정책 등 내부 기술 정보를 포함하지 마세요
- 오직 스카우팅 분석 결과와 실무적 제안만 작성하세요
- 축구 전문가가 읽는다고 가정하고, 구체적 수치와 근거를 들어 논리적으로 서술하세요
```

### Fallback 분석 시스템

Gemini API 호출 실패 시 자동으로 생성되는 규칙 기반 분석:

#### 트리거 조건
```python
# app.py Line 910-916

if hasattr(response, 'text') and response.text:
    result_text = response.text.strip()
    if len(result_text) < 500:
        # 너무 짧은 응답은 실패로 간주
        return _generate_fallback_analysis(...)
else:
    return _generate_fallback_analysis(...)
```

#### Fallback 로직
```python
# app.py Line 933-1000

def _generate_fallback_analysis(player_hsi, team_template, ...):
    # 퍼센타일 기반 분석
    t_score = player_hsi['t_fit_score']
    
    if t_score >= 75:
        t_analysis = "리그 상위 25% 수준의 뛰어난 전방 압박 능력"
    elif t_score >= 50:
        t_analysis = "중위권 수준의 전술 적합도"
    elif t_score >= 25:
        t_analysis = "하위 중위권의 전방 압박"
    else:
        t_analysis = "하위 25% 수준의 전방 압박"
    
    # ... P-Fit, C-Fit 동일 방식
```

---

## 데이터 파이프라인

### 파이프라인 구조

```
raw_data.csv → hsi_calculator.py → hsi_scores_2024.csv
                                 → player_insights.json

match_info.csv ↗

hsi_scores_2024.csv → team_profiler.py → team_templates.json

player_profile.csv → app.py → UI/PDF
foreign_birth_city_2024.csv ↗
wvs_city_vectors.csv ↗
```

### 실행 순서

```bash
# 전체 파이프라인
python src/pipeline_runner.py

# 또는 개별 실행
python src/hsi_calculator.py
python src/team_profiler.py
streamlit run app.py
```

### 데이터 검증

#### 필수 컬럼 검증
```python
# src/hsi_calculator.py Line 23-35

def validate_raw_data(df: pd.DataFrame) -> bool:
    required_cols = ['player_id', 'player_name_ko', 'game_id', 'type_name', 'start_x']
    missing = set(required_cols) - set(df.columns)
    
    if missing:
        logger.error(f"필수 컬럼 누락: {missing}")
        raise ValueError(f"필수 컬럼 누락: {missing}")
    
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        logger.warning(f"결측치 발견:\n{null_counts[null_counts > 0]}")
    
    return True
```

### 로깅 시스템

#### 설정
```python
# src/hsi_calculator.py Line 10-20

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/hsi_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

#### 로그 수준
```
INFO: 파이프라인 진행 상황
WARNING: 데이터 품질 이슈 (결측치 등)
ERROR: 치명적 오류 (파일 없음, 필수 컬럼 누락)
```

### 팀 템플릿 생성

#### 알고리즘
```python
# src/team_profiler.py Line 40-60

for team in teams:
    for position_group in ['FW', 'MF', 'DF']:
        team_players = hsi_df[
            (hsi_df['team_name'] == team) & 
            (hsi_df['position_group'] == position_group)
        ]
        
        template = {
            't_fit_score': team_players['t_fit_score'].mean(),
            'p_fit_score': team_players['p_fit_score'].mean(),
            'c_fit_score': team_players['c_fit_score'].mean()
        }
        
        templates[team][position_group] = template
```

포지션별 평균을 계산하여 각 팀의 전술 특성을 수치화합니다.

---

## 성능 최적화

### 캐싱 전략

#### Streamlit 캐싱
```python
# app.py Line 85-90

@st.cache_data
def load_data():
    hsi_df = pd.read_csv(HSI_SCORES_PATH)
    templates = load_json(TEAM_TEMPLATES_PATH)
    return hsi_df, templates
```

#### 캐시 무효화 조건
- 파일이 변경될 때 (타임스탬프 기반)
- Streamlit 재시작 시

### 데이터 로딩 최적화

#### 선택적 컬럼 로드
```python
# 필요한 컬럼만 로드
required_cols = ['player_id', 'player_name_ko', 't_fit_score', 'p_fit_score', 'c_fit_score']
hsi_df = pd.read_csv(HSI_SCORES_PATH, usecols=required_cols)
```

#### 데이터 타입 최적화
```python
# 메모리 사용량 감소
dtype_dict = {
    'player_id': 'int32',
    't_fit_score': 'float32',
    'p_fit_score': 'float32',
    'c_fit_score': 'float32'
}
```

### PDF 생성 최적화

#### 차트 이미지 재사용
```python
# app.py Line 1556-1620

# 1. 차트 생성 (메모리)
chart_path = reports_dir / f"report_{player_name}_{team_name}.png"

# 2. 이미지 저장
pdf_fig.write_image(str(chart_path), width=800, height=600)

# 3. PDF에 삽입
story.append(Image(str(chart_path), width=5*inch, height=4*inch))

# 4. 임시 파일 정리 (선택적)
chart_path.unlink()
```

### 병목 구간 분석

#### 처리 시간 분포
```
데이터 로딩: ~0.5초
HSI 계산 (449명): ~3초
C-Fit 계산 (1명): ~0.1초
AI 분석 생성: ~5-10초
PDF 생성: ~2초
```

#### 최적화 우선순위
1. AI 분석 (가장 느림) → 캐싱 불가 (선수별 다름)
2. HSI 계산 → 사전 계산 및 저장 (output/)
3. PDF 생성 → kaleido 최적화

---

## 수식 요약표

### T-Fit
```
press_ratio = high_press_count / total_def_count
t_fit_smoothed = (x * n + μ * k) / (n + k)
clean_play = 1 / (1 + penalty_rate / scale)
t_fit_final = t_fit_smoothed * (0.7 + 0.3 * clean_play)
T-Fit = percentile_rank(t_fit_final) * 100
```

### P-Fit
```
retention = (summer_events / summer_games) / (rest_events / rest_games)
p_fit_smoothed = (retention * n_summer + 1.0 * k) / (n_summer + k)
P-Fit = percentile_rank(p_fit_smoothed) * 100
```

### C-Fit
```
C-Fit = cosine_similarity(player_vector, team_vector)
     = (A · B) / (||A|| * ||B||)
```

### Adapt-Fit
```
directional_sim(x_p, x_t) = 1 + sign(x_p - x_t) * 0.5 * |x_p - x_t| / scale

Adapt-Fit = Σ(w_i * directional_sim_i) / Σ(w_i) * 100
```

---

## 하이퍼파라미터 튜닝 가이드

### 베이지안 스무딩 k 값

#### T-Fit k (현재: 5.0)
```
권장 범위: 3.0 - 10.0

k 증가 → 보수적 (전체 평균에 가까워짐)
k 감소 → 공격적 (개별 관측값에 의존)

조정 기준:
- 소표본 선수가 많으면 k 증가
- 데이터 품질이 높으면 k 감소
```

#### P-Fit k (현재: 3.0)
```
권장 범위: 2.0 - 5.0

T-Fit보다 작은 이유:
- 여름 경기가 적어 데이터 희소
- 더 많은 shrinkage 필요
```

### 방향성 유사도 α (현재: 0.5)
```
권장 범위: 0.3 - 0.7

α 증가 → 선수-팀 차이에 민감
α 감소 → 차이를 덜 강조

조정 기준:
- 전술 적합도를 강조하려면 α 증가
- 안정적인 점수를 원하면 α 감소
```

### 포지션별 가중치
```
현재 설정:
DF: (0.45, 0.25, 0.30)
MF: (0.40, 0.30, 0.30)
FW: (0.35, 0.35, 0.30)

조정 기준:
- 팀 전술에 따라 T-Fit 비중 조정
- 혹서기 중요도에 따라 P-Fit 비중 조정
- 외국인 비율에 따라 C-Fit 비중 조정
```

---

## 한계점 및 향후 개선 방향

### 현재 한계

1. **출전 시간 미반영**
   - 경기당 이벤트 수는 출전 시간에 종속
   - 교체 출전 선수는 과소평가 가능

2. **포지션 세분화 부족**
   - 현재: FW/MF/DF 3개 그룹
   - 개선: LW/RW/CF, CDM/CAM, CB/FB 등

3. **시간 가변성 미고려**
   - 선수의 성장/쇠퇴 추세 미반영
   - 최근 경기에 더 높은 가중치 필요

4. **상대 팀 강도 미고려**
   - 약한 팀 상대 압박과 강한 팀 상대 압박을 동일하게 취급

### 향후 개선 방향

1. **출전 시간 정규화**
```python
events_per_90min = (total_events / total_minutes) * 90
```

2. **시간 가중 평균**
```python
weighted_score = Σ(score_i * decay^(current_date - game_date_i))
decay = 0.95  # 최근 경기에 더 높은 가중치
```

3. **상대 팀 보정**
```python
adjusted_score = raw_score * opponent_strength_factor
```

4. **머신러닝 통합**
```python
# XGBoost를 사용한 Adapt-Fit 예측
model = xgb.XGBRegressor()
model.fit(X=[t_fit, p_fit, c_fit, ...], y=actual_performance)
```

---

문서 버전: 1.0  
최종 업데이트: 2025년 1월  
작성자: 백엔드 개발팀
대상 데이터: 2024 K리그 시즌

