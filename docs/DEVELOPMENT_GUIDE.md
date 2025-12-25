# K-Scout Adapt-Fit AI - 개발 관리 문서

> **프로젝트:** K리그 외국인 선수 전술 적합도 분석 시스템  
> **버전:** MVP 1.0  
> **최종 업데이트:** 2024-12-21

---

## 🏃‍♂️ Sprint Board (개발 Tasks)

### Status 정의
| Status | 설명 |
|--------|------|
| `Backlog` | 아이디어 단계, 구체화 필요 |
| `Todo` | 이번 스프린트에 할 일 |
| `In Progress` | 현재 작업 중 |
| `Code Review` | 코드 리뷰 대기 |
| `Done` | 완료 |

### Priority 정의
| Priority | 설명 |
|----------|------|
| `P0` | 🔴 긴급 - 즉시 해결 필요 |
| `P1` | 🟠 이번 주 필수 |
| `P2` | 🟡 시간 날 때 |
| `P3` | 🟢 나중에 |

### Size 정의 (일정 산정용)
| Size | 예상 소요 시간 |
|------|---------------|
| `S` | 1시간 이내 |
| `M` | 반나절 (3-4시간) |
| `L` | 하루 (8시간) |
| `XL` | 며칠 (2-3일) |

---

## 📋 현재 Sprint Tasks

### ✅ 완료된 작업 (Done)

| Task | Priority | Size | 완료일 |
|------|----------|------|--------|
| HSI 지표 계산 로직 개발 | P0 | L | 2024-12-15 |
| 팀별 전술 템플릿 생성 | P0 | M | 2024-12-15 |
| Streamlit 대시보드 MVP | P0 | XL | 2024-12-15 |
| OpenAI API 연동 (Gemini → GPT-4) | P1 | M | 2024-12-21 |
| 다크 테마 UI 개선 | P1 | M | 2024-12-21 |
| 추천 선수 랭킹 기능 | P2 | M | 2024-12-21 |
| PDF 보고서 생성 기능 | P1 | L | 2024-12-21 |

### 🔄 진행 중 (In Progress)

| Task | Priority | Size | 담당 | 비고 |
|------|----------|------|------|------|
| - | - | - | - | - |

### 📝 할 일 (Todo)

| Task | Priority | Size | 비고 |
|------|----------|------|------|
| Streamlit 버전 업그레이드 | P2 | M | 1.12.0 → 최신 버전 |
| 선수 비교 기능 (2인 동시) | P3 | L | UI 개선 후 진행 |
| 국적별 필터링 | P3 | S | 데이터 보강 필요 |

---

## 🗂️ Sprint Backlog (미래 작업)

| Task | 설명 | 예상 Size |
|------|------|-----------|
| 시즌 데이터 업데이트 자동화 | 2025 시즌 데이터 반영 파이프라인 | XL |
| 선수 이미지 추가 | 프로필 사진 연동 | M |
| 다국어 지원 | 영어 버전 추가 | L |
| 모바일 반응형 UI | 모바일에서도 사용 가능하게 | L |
| 히스토리 기능 | 과거 분석 결과 저장/조회 | XL |

---

## 🧪 실험실 (Model Experiments)

### 실험 기록

| 날짜 | 모델 버전 | 변경 사항 | 결과 | 결론 |
|------|----------|----------|------|------|
| 2024-12-07 | v0.1 | XGBoost 프록시 모델 초기 구현 | RMSE 0.15 | 기준선 설정 |
| 2024-12-15 | v0.2 | HSI 3요소 (T/P/C-Fit) 도입 | - | 직관적인 지표로 전환 |
| 2024-12-15 | v0.3 | 포지션별 가중치 적용 (FW/MF/DF) | 적합도 정밀도 향상 | 성공 ✅ |
| 2024-12-21 | v0.4 | 비대칭 유사도 함수 적용 | 상향 성능은 보너스 처리 | 성공 ✅ |

### HSI 가중치 설정 (현재 적용 중)

```python
# 포지션별 HSI 가중치
FW: T-Fit 50% + P-Fit 30% + C-Fit 20%
MF: T-Fit 40% + P-Fit 40% + C-Fit 20%
DF: T-Fit 30% + P-Fit 30% + C-Fit 40%
```

---

## 🏗️ Tech Wiki

### ADR (Architecture Decision Records)

#### ADR-001: Streamlit 선택 이유
- **결정:** React 대신 Streamlit 사용
- **이유:** 
  - 빠른 프로토타이핑 (MVP 목적)
  - Python 기반으로 데이터 처리와 통합 용이
  - 별도 프론트엔드 개발 불필요
- **트레이드오프:** 커스터마이징 제한, 복잡한 UI 구현 어려움

#### ADR-002: OpenAI API 전환
- **결정:** Google Gemini → OpenAI GPT-4o-mini
- **이유:**
  - `google-generativeai` 패키지 deprecated
  - Python 버전 호환성 문제 (3.9)
  - GPT-4o-mini의 한국어 성능 우수
- **날짜:** 2024-12-21

#### ADR-003: HSI (Harmonic Synergy Index) 설계
- **결정:** 단일 점수 대신 3요소 분리 (T-Fit, P-Fit, C-Fit)
- **이유:**
  - 각 요소별 해석 가능성 향상
  - 포지션별 가중치 적용 유연성
  - 스카우터에게 명확한 인사이트 제공

### API 문서

#### 핵심 함수

```python
# 적합도 점수 계산
calculate_adapt_fit_score(player_hsi, team_template, pos_group) -> float
# 반환: 0-100 사이의 적합도 점수

# 전체 선수 랭킹
get_all_player_scores(team_name, templates, hsi_df, foreigners_df) -> list
# 반환: 점수순 정렬된 선수 리스트

# AI 분석 생성
get_ai_analysis_for_pdf(api_key, player_hsi, team_template, ...) -> str
# 반환: 스카우팅 리포트 텍스트
```

### Onboarding (환경 설정 가이드)

#### 1. 저장소 클론
```bash
git clone <repository-url>
cd kleague
```

#### 2. 가상환경 설정
```bash
# Python 3.9 가상환경 활성화
source k-scout-env-py39/bin/activate

# 또는 새로 생성
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. 환경변수 설정
```bash
# .env 파일 생성
echo "OPENAI_API_KEY=sk-your-api-key" > .env
```

#### 4. 앱 실행
```bash
streamlit run app.py
# 브라우저에서 http://localhost:8501 접속
```

#### 5. 데이터 파이프라인 (필요시)
```bash
python src/hsi_calculator.py    # HSI 점수 계산
python src/team_profiler.py     # 팀 템플릿 생성
```

---

## ⚠️ 알려진 이슈 및 제약사항

### Streamlit 1.12.0 제약사항
| 기능 | 지원 여부 | 대안 |
|------|----------|------|
| `st.cache_data` | ❌ | `@st.cache(allow_output_mutation=True)` |
| `use_container_width` | ❌ | 파라미터 제거 |
| `hide_index` | ❌ | 파라미터 제거 |
| `gap` in columns | ❌ | 파라미터 제거 |
| 컬럼 중첩 | ❌ | 단일 레벨만 사용 |

### 다크 테마 이슈
- 커스텀 HTML이 제대로 렌더링되지 않을 수 있음
- 해결: Streamlit 네이티브 컴포넌트 사용 권장
- CSS에서 `color: #ffffff !important` 적용

---

## 📁 프로젝트 구조

```
kleague/
├── app.py                    # 🎯 메인 Streamlit 앱
├── requirements.txt          # 의존성 패키지
├── README.md                 # 프로젝트 소개
├── .env                      # API 키 (git 제외)
│
├── src/                      # 소스 코드
│   ├── hsi_calculator.py     # HSI 지표 계산
│   ├── team_profiler.py      # 팀 템플릿 생성
│   ├── analyzer.py           # 분석 로직
│   └── data_extractor.py     # 데이터 추출
│
├── data/                     # 데이터
│   ├── raw/                  # 원본 K리그 데이터
│   └── processed/            # 가공된 데이터
│
├── output/                   # 분석 결과
│   ├── hsi_scores_2024.csv
│   ├── team_templates.json
│   └── 2024_foreigners_list.csv
│
├── docs/                     # 문서
│   ├── DEVELOPMENT_GUIDE.md  # 이 문서
│   ├── SYSTEM_SPECIFICATION.md
│   └── FINAL_REPORT.md
│
└── k-scout-env-py39/         # Python 가상환경
```

---

## 📞 연락처

- **프로젝트 담당:** [담당자 이름]
- **이메일:** [이메일 주소]
- **GitHub:** [저장소 URL]

---

*이 문서는 프로젝트 진행에 따라 지속적으로 업데이트됩니다.*

