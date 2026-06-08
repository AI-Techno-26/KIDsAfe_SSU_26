# KIDsAfe API 서버

어린이보호구역 위험도 예측 API (XGBoost 이진분류)
---
<img width="300" height="141" alt="kidsafe_logo_final" src="https://github.com/user-attachments/assets/a62db558-d60a-45b1-81dc-f63d89311db3" />

---

## 파일 구조

```
KIDsAfe_SSU_26-main/
├── main.py                
├── index.html             
├── requirements.txt
├── render.yaml
├── gitignore              
├── LICENSE
├── README.md
├── data/
│   └── 서울어린이보호구역_500M.csv  
└── models/
    ├── xgb_model.pkl      
    ├── scaler.pkl         
    ├── base_stats.pkl     
    ├── tree_교차로수.pkl   
    ├── tree_불법주차수.pkl  
    ├── tree_신호등수.pkl   
    ├── tree_안전표지수.pkl  
    ├── tree_음향신호기수.pkl 
    └── tree_잔여시간표시기수.pkl 
```

---

## 배포 순서

### 1단계: Google Colab에서 파일 다운로드

```python
# Colab 실행 후 models 폴더에서 다운로드
from google.colab import files
files.download('/content/drive/MyDrive/.../models_162/xgb_model.pkl')
files.download('/content/drive/MyDrive/.../models_162/scaler.pkl')
```

### 2단계: GitHub 저장소 생성 및 업로드

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

### 3단계: Render.com 배포

1. https://render.com 접속 → 회원가입
2. New → Web Service
3. GitHub 저장소 연결
4. 설정:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Deploy 클릭
6. 배포 완료 후 URL 확인 (예: `https://kidsafe-api.onrender.com`)

### 4단계: 프론트엔드 API URL 수정

`index.html` 5번째 줄:
```javascript
const API_URL = 'https://kidsafe-api.onrender.com';  // ← 실제 URL로 교체
```

### 5단계: 프론트엔드 배포 (GitHub Pages)

1. GitHub 저장소 → Settings → Pages
2. Source: main 브랜치 / root 폴더
3. Save → `https://{유저명}.github.io/kidsafe-api` 로 접속 가능

---

## API 명세

### POST /predict

**요청**
```json
{
  "lat": 37.5665,
  "lng": 126.9780
}
```

**응답**
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

### GET /health
서버 상태 확인

---

## 로컬 테스트

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000/docs 에서 Swagger UI 확인
```
