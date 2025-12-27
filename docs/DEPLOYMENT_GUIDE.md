# Streamlit Cloud 배포 가이드 (한글 폰트 설정 포함)

## 📦 배포 준비 파일

### 1. `packages.txt` (필수)
Streamlit Cloud에서 한글 폰트를 설치하기 위한 시스템 패키지 목록입니다.

```txt
fonts-nanum
fonts-nanum-coding
fonts-nanum-extra
```

이 파일은 **프로젝트 루트**에 위치해야 합니다.

---

## 🚀 Streamlit Cloud 배포 절차

### 1단계: GitHub 저장소 확인
```bash
# packages.txt가 푸시되었는지 확인
git status
git add packages.txt
git commit -m "Add Korean font support for deployment"
git push origin main
```

### 2단계: Streamlit Cloud 설정

1. **Streamlit Cloud 접속**
   - https://share.streamlit.io/ 접속
   - GitHub 계정으로 로그인

2. **새 앱 배포**
   - "New app" 클릭
   - Repository: `danga011/Kleague` 선택
   - Branch: `main`
   - Main file path: `app.py`

3. **환경 변수 설정 (Secrets)**
   - "Advanced settings" 클릭
   - "Secrets" 섹션에 다음 추가:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```

4. **배포 시작**
   - "Deploy!" 클릭
   - 빌드 로그 확인 (약 3-5분 소요)

---

## 🔍 배포 후 확인사항

### 한글 폰트 로드 확인
배포 후 터미널 로그에서 다음 메시지를 확인하세요:

**성공 시:**
```
✓ 한글 폰트 로드 성공: NanumGothic (/usr/share/fonts/truetype/nanum/NanumGothic.ttf)
```

**실패 시:**
```
⚠️ 한글 폰트를 찾지 못해 기본 폰트로 생성합니다.
```

### PDF 생성 테스트
1. 배포된 앱에서 선수 선택
2. "PDF 보고서 생성" 버튼 클릭
3. 생성된 PDF에서 한글이 정상적으로 표시되는지 확인

---

## 🐛 문제 해결

### 문제 1: "한글 폰트를 찾지 못해 기본 폰트로 생성합니다" 경고

**원인:** `packages.txt`가 제대로 인식되지 않았거나, 파일 위치가 잘못됨

**해결 방법:**
```bash
# 1. packages.txt 위치 확인 (프로젝트 루트여야 함)
ls -la packages.txt

# 2. 파일 내용 확인
cat packages.txt
# 출력:
# fonts-nanum
# fonts-nanum-coding
# fonts-nanum-extra

# 3. Git에 푸시되었는지 확인
git ls-files | grep packages.txt

# 4. Streamlit Cloud에서 재배포
# Streamlit Cloud 대시보드 → "Reboot app" 클릭
```

### 문제 2: PDF에서 한글이 깨짐 (□□□로 표시)

**원인:** 폰트가 설치되지 않았거나, 폰트 경로가 잘못됨

**해결 방법:**
1. Streamlit Cloud 로그 확인
   ```
   Settings → Logs → "View full logs"
   ```

2. 다음 로그 확인:
   ```
   Reading package lists...
   Building dependency tree...
   The following NEW packages will be installed:
     fonts-nanum fonts-nanum-coding fonts-nanum-extra
   ```

3. 로그에 폰트 설치 메시지가 없으면:
   - `packages.txt` 파일명 확인 (대소문자 정확히)
   - 파일 인코딩 확인 (UTF-8)
   - 빈 줄 없이 작성되었는지 확인

### 문제 3: Plotly 차트가 PDF에 표시되지 않음

**원인:** `kaleido` 패키지 문제

**해결 방법:**
`requirements.txt`에 kaleido 버전 고정:
```txt
kaleido==0.2.1
```

---

## 📋 배포 체크리스트

배포 전:
- [ ] `packages.txt` 파일이 프로젝트 루트에 있는지 확인
- [ ] `packages.txt` 내용이 정확한지 확인
- [ ] `.env` 파일이 `.gitignore`에 포함되었는지 확인
- [ ] Git에 푸시 완료

배포 후:
- [ ] Streamlit Cloud에서 빌드 성공 확인
- [ ] Secrets에 `GEMINI_API_KEY` 등록 확인
- [ ] 앱이 정상적으로 로드되는지 확인
- [ ] PDF 생성 테스트
- [ ] 한글 폰트 정상 표시 확인
- [ ] 레이더 차트 정상 표시 확인

---

## 📊 파일 구조 (배포용)

```
kleague/
├── app.py                    # 메인 애플리케이션
├── requirements.txt          # Python 패키지
├── packages.txt              # 시스템 패키지 (한글 폰트) ⭐
├── .env                      # API 키 (로컬용, Git 제외)
├── .gitignore                # Git 제외 파일
├── README.md
├── src/
├── data/
└── output/
```

---

## 🎯 배포 환경별 폰트 경로

| 환경 | 폰트 경로 | 우선순위 |
|------|----------|---------|
| **Streamlit Cloud** | `/usr/share/fonts/truetype/nanum/NanumGothic.ttf` | 1순위 |
| macOS | `/System/Library/Fonts/AppleGothic.ttf` | 2순위 |
| Windows | `C:\Windows\Fonts\malgun.ttf` | 3순위 |

현재 `app.py`는 위 순서대로 폰트를 찾아서 자동으로 로드합니다.

---

## 🔗 참고 링크

- **Streamlit Cloud 문서**: https://docs.streamlit.io/streamlit-community-cloud
- **packages.txt 가이드**: https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/app-dependencies#apt-get-dependencies
- **나눔폰트**: https://hangeul.naver.com/font

---

## 💡 추가 팁

### 로컬에서 배포 환경 테스트 (Ubuntu/Debian)
```bash
# 나눔폰트 설치
sudo apt-get update
sudo apt-get install -y fonts-nanum fonts-nanum-coding fonts-nanum-extra

# 폰트 캐시 업데이트
fc-cache -fv

# 설치 확인
fc-list | grep Nanum
```

### Streamlit Cloud 로그 실시간 확인
배포 중 문제가 발생하면:
1. Streamlit Cloud 대시보드
2. 해당 앱 클릭
3. "Manage app" → "Logs"
4. 실시간 로그 확인

---

**최종 업데이트**: 2024년 12월 27일  
**버전**: v1.1 (한글 폰트 지원)  
**상태**: 배포 준비 완료 ✅

