# main.py
"""
KIDsAfe API 서버 v3
위도/경도 입력 → 어린이보호구역 위험도 예측 (XGBoost 이진분류)
CSV 로드 없이 pkl 파일만 사용 → 메모리 절약
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle, os
import numpy as np
from sklearn.neighbors import BallTree

# ── 앱 초기화 ─────────────────────────────────────────────────────────────────
app = FastAPI(title="KIDsAfe API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── 경로 ─────────────────────────────────────────────────────────────────────
RADIUS_M     = 500
R_RAD        = RADIUS_M / 6_371_000
FEATURE_COLS = ['도로폭', 'CCTV대수', '잔여시간표시기수', '음향신호기수',
                '안전표지수', '신호등수', '교차로수', '불법주차_구간']
RISK_NAMES   = {0: '안전', 1: '위험'}
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, 'models')

# ── pkl 파일 로드 ─────────────────────────────────────────────────────────────
print("▶ 모델 및 BallTree pkl 로드 중...")

def load_pkl(filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, 'rb') as f:
        return pickle.load(f)

# XGBoost 모델 & scaler
xgb_model = load_pkl('xgb_model.pkl')
scaler    = load_pkl('scaler.pkl')
print("  ✅ xgb_model, scaler 로드 완료")

# BallTree (시설물 6종)
FACILITY_TREES = {}
for nm in ['잔여시간표시기수', '교차로수', '음향신호기수',
           '안전표지수', '신호등수', '불법주차수']:
    FACILITY_TREES[nm] = load_pkl(f'tree_{nm}.pkl')
    print(f"  ✅ tree_{nm}.pkl 로드 완료")

# 기준 통계 (도로폭 평균, CCTV 평균, 불법주차 분위수)
stats     = load_pkl('base_stats.pkl')
MEAN_ROAD = stats['MEAN_ROAD']
MEAN_CCTV = stats['MEAN_CCTV']
P33       = stats['P33']
P67       = stats['P67']

print(f"\n✅ 서버 준비 완료")
print(f"   도로폭 평균={MEAN_ROAD:.1f}m | CCTV 평균={MEAN_CCTV:.1f}대")
print(f"   불법주차 P33={P33:.0f} / P67={P67:.0f}")


# ── 스키마 ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    lat: float = Field(..., example=37.5665, description="위도")
    lng: float = Field(..., example=126.9780, description="경도")

class FeatureDetail(BaseModel):
    도로폭: float
    CCTV대수: float
    잔여시간표시기수: int
    음향신호기수: int
    안전표지수: int
    신호등수: int
    교차로수: int
    불법주차_구간: int
    불법주차_원본건수: int

class PredictResponse(BaseModel):
    lat: float
    lng: float
    위험도등급: int
    위험도명: str
    안전확률: float
    위험확률: float
    features: FeatureDetail


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def cnt(nm: str, lat: float, lng: float) -> int:
    """BallTree로 반경 500m 내 시설물 수 반환"""
    return int(FACILITY_TREES[nm].query_radius(
        np.radians([[lat, lng]]), r=R_RAD, count_only=True)[0])


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "KIDsAfe API", "version": "3.0.0", "status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    lat, lng = req.lat, req.lng

    if not (37.2 <= lat <= 37.8 and 126.5 <= lng <= 127.3):
        raise HTTPException(status_code=400,
            detail="서울 내 좌표를 입력해주세요. (위도 37.2~37.8 / 경도 126.5~127.3)")

    # 피처 생성
    fv = {'도로폭': MEAN_ROAD, 'CCTV대수': MEAN_CCTV}
    for nm in ['잔여시간표시기수', '음향신호기수', '안전표지수', '신호등수', '교차로수']:
        fv[nm] = cnt(nm, lat, lng)

    park_cnt          = cnt('불법주차수', lat, lng)
    fv['불법주차_구간'] = 0 if park_cnt <= P33 else (1 if park_cnt <= P67 else 2)

    # 예측
    x        = np.array([[fv.get(f, 0) for f in FEATURE_COLS]])
    x_scaled = scaler.transform(x)
    grade    = int(xgb_model.predict(x_scaled)[0])
    probs    = xgb_model.predict_proba(x_scaled)[0]

    return PredictResponse(
        lat=lat, lng=lng,
        위험도등급=grade,
        위험도명=RISK_NAMES[grade],
        안전확률=round(float(probs[0]), 4),
        위험확률=round(float(probs[1]), 4),
        features=FeatureDetail(
            도로폭=round(fv['도로폭'], 2),
            CCTV대수=round(fv['CCTV대수'], 2),
            잔여시간표시기수=fv['잔여시간표시기수'],
            음향신호기수=fv['음향신호기수'],
            안전표지수=fv['안전표지수'],
            신호등수=fv['신호등수'],
            교차로수=fv['교차로수'],
            불법주차_구간=fv['불법주차_구간'],
            불법주차_원본건수=park_cnt,
        )
    )
