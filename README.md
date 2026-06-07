MediVIse_KAIST_Seocho_25

<img width="300" height="141" alt="kidsafe_logo_final" src="https://github.com/user-attachments/assets/a62db558-d60a-45b1-81dc-f63d89311db3" />
KIDsAfe API 서버
어린이보호구역 위험도 예측 API (XGBoost 이진분류)
---
파일 구조
```
kidsafe_server/
├── main.py              ← FastAPI 서버
├── requirements.txt     ← 패키지 목록
├── render.yaml          ← Render 배포 설정
├── index.html           ← 프론트엔드
├── models/
│   ├── xgb_model.pkl    ← 학습된 XGBoost 모델 (Colab에서 복사)
│   └── scaler.pkl       ← MinMaxScaler (Colab에서 복사)
└── data/
    ├── 서울시 잔여시간표시기 관련 정보-위경도(추가).csv
    ├── 서울시 교차로 관련 정보-위경도(추가).csv
    ├── 서울시 음향신호기 관련 정보-위경도(추가).csv
    ├── 서울시 안전표지 관련 정보-위경도(추가).csv
    ├── 서울시 신호등 관련 정보-위경도(추가).csv
    ├── 서울시_불법주정차단속_위치정보_병합.csv
    └── 서울어린이보호구역(2019-2024)기준_교통사고다발지역_500M.csv
```
---
배포 순서
1단계: Google Colab에서 파일 다운로드
```python
# Colab 실행 후 models 폴더에서 다운로드
from google.colab import files
files.download('/content/drive/MyDrive/.../models_162/xgb_model.pkl')
files.download('/content/drive/MyDrive/.../models_162/scaler.pkl')
```
2단계: GitHub 저장소 생성 및 업로드
```bash
git init
git add .
git commit -m "KIDsAfe API 초기 배포"
git remote add origin https://github.com/{유저명}/kidsafe-api.git
git push -u origin main
```
> ⚠ data/ 폴더 CSV 파일이 크면 Git LFS 사용:
> ```bash
> git lfs track "data/*.csv"
> git add .gitattributes
> ```
3단계: Render.com 배포
https://render.com 접속 → 회원가입
New → Web Service
GitHub 저장소 연결
설정:
Runtime: Python 3
Build Command: `pip install -r requirements.txt`
Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
Deploy 클릭
배포 완료 후 URL 확인 (예: `https://kidsafe-api.onrender.com`)
4단계: 프론트엔드 API URL 수정
`index.html` 5번째 줄:
```javascript
const API_URL = 'https://kidsafe-api.onrender.com';  // ← 실제 URL로 교체
```
5단계: 프론트엔드 배포 (GitHub Pages)
GitHub 저장소 → Settings → Pages
Source: main 브랜치 / root 폴더
Save → `https://{유저명}.github.io/kidsafe-api` 로 접속 가능
---
API 명세
POST /predict
요청
```json
{
  "lat": 37.5665,
  "lng": 126.9780
}
```
응답
```json
{
  "lat": 37.5665,
  "lng": 126.9780,
  "위험도등급": 1,
  "위험도명": "위험",
  "안전확률": 0.2341,
  "위험확률": 0.7659,
  "features": {
    "도로폭": 12.3,
    "CCTV대수": 1.8,
    "잔여시간표시기수": 45,
    "음향신호기수": 52,
    "안전표지수": 380,
    "신호등수": 240,
    "교차로수": 18,
    "불법주차_구간": 2,
    "불법주차_원본건수": 6200
  }
}
```
GET /health
서버 상태 확인
---
로컬 테스트
```bash
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000/docs 에서 Swagger UI 확인
```
