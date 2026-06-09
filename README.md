# KIDsAfe — 어린이보호구역 위험도 예측 AI

<img width="300" alt="kidsafe_logo" src="https://github.com/user-attachments/assets/a62db558-d60a-45b1-81dc-f63d89311db3"/>

> **K**ids **I**ntelligent **D**anger **S**afety **A**I for **F**amily **E**nvironment  
> 서울시 어린이보호구역 주변 8개 요인을 분석해 교통사고 위험도를 예측하는 AI 서비스

🔗 **서비스 바로가기**: https://kidsafe-ssu-26.onrender.com/app

<img width="250" height="250" alt="qrcode_360077965_e01eb64a0dcf9435260bd8e172168213" src="https://github.com/user-attachments/assets/e7dd4a97-daa1-4aee-910b-ab4ce5c73aa6" />
<img width="550" height="250" alt="image" src="https://github.com/user-attachments/assets/851d2f2f-c63e-404b-b351-4755b90ddcdb" />

---

## 📌 프로젝트 개요

어린이보호구역 반경 500m 내의 **8가지 요인**을 분석하여 해당 구역의 위험도를 **안전 / 위험** 으로 예측합니다.

| 구분 | 요인 | 예상 효과 |
|---|---|---|
| 🛡 보호 요인 | 잔여시간표시기, 음향신호기, 안전표지, 신호등, CCTV, 도로폭 | 수↑ → 사고율↓ |
| ⚠ 위험 요인 | 교차로, 불법주차 | 수↑ → 사고율↑ |

---

## 🗂 파일 구조

```
KIDsAfe_SSU_26/
├── main.py              ← FastAPI 서버 (API 전체 로직)
├── index.html           ← 프론트엔드 (HTML + Vanilla JS)
├── requirements.txt     ← Python 패키지 목록
├── render.yaml          ← Render.com 배포 설정
├── .gitignore
└── models/
    ├── xgb_model.pkl            ← 학습된 XGBoost 모델
    ├── scaler.pkl               ← MinMaxScaler
    ├── base_stats.pkl           ← 도로폭/CCTV 평균, 불법주차 분위수
    ├── tree_잔여시간표시기수.pkl  ← BallTree (반경 카운팅용)
    ├── tree_교차로수.pkl
    ├── tree_음향신호기수.pkl
    ├── tree_안전표지수.pkl
    ├── tree_신호등수.pkl
    └── tree_불법주차수.pkl
```

---

## 🤖 모델 정보

| 항목 | 내용 |
|---|---|
| **모델** | XGBoost (이진분류) |
| **분류** | 안전(0): 사고건수 ≤ 33건 / 위험(1): 사고건수 > 33건 |
| **학습 데이터** | 서울 어린이보호구역 162개 (중복 제거 후) |
| **반경** | 500m |
| **독립변수** | 8개 (보호요인 6 + 위험요인 2) |
| **CV F1 Score** | 0.700 (5-Fold Stratified CV) |
| **CV Accuracy** | 0.759 |

### 분석 한계점

1. **표본 수 부족** — 중복 제거 후 162개로 통계적 안정성 낮음
2. **공간 자기상관** — 500m 반경 겹침 93.8% → 독립 관측치 가정 위반
3. **서울 한정** — 서울시 시설물 데이터만 사용 → 타 지역 일반화 불가
4. **역인과 가능성** — 안전표지·신호등은 사고 다발 구역에 사후 설치되는 경향

---

## 🚀 API 명세

### `POST /predict`

위도/경도를 입력하면 위험도를 예측합니다.

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

| 엔드포인트 | 설명 |
|---|---|
| `GET /` | 서버 상태 확인 |
| `GET /health` | 헬스체크 |
| `GET /app` | 프론트엔드 UI |
| `POST /predict` | 위험도 예측 |
| `GET /docs` | Swagger UI |

---

## 🛠 로컬 실행

```bash
# 패키지 설치
pip install -r requirements.txt

# 서버 실행
uvicorn main:app --reload

# 접속
http://localhost:8000/app     ← 프론트엔드
http://localhost:8000/docs    ← Swagger UI
```

---

## 📊 데이터 출처

| 데이터 | 출처 |
|---|---|
| 전국 어린이보호구역 표준데이터 | 공공데이터포털 |
| 전국 교통사고 다발지역 표준데이터 | 공공데이터포털 |
| 서울시 시설물 데이터 (신호등, 교차로 등) | 서울 열린데이터광장 |
| 서울시 불법주정차 단속 위치정보 | 서울 열린데이터광장 |

---

## 👥 팀 정보

**숭실대학교 AI프로그래밍 6조**

---

## 📄 License

MIT License
