# main.py
"""
KIDsAfe API 서버 (전국 모델 · SMOTE 적용)
위도/경도 입력 → 어린이보호구역 위험도 3단계 예측 (XGBoost, 9피처)
피처: 8개 시설물 + 교통량지표 / pkl만 로드 (메모리 절약)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pickle, os
import numpy as np
from sklearn.neighbors import BallTree

# ── 앱 초기화 ─────────────────────────────────────────────────────────────────
app = FastAPI(title="KIDsAfe API (전국)", version="4.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── 상수 ──────────────────────────────────────────────────────────────────────
RADIUS_M = 500
R_RAD = RADIUS_M / 6_371_000
RISK_NAMES = {0: "안전", 1: "주의", 2: "위험"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 학습 시 피처 순서와 정확히 일치해야 함 (8개 시설물 + 교통량지표)
FACILITY_KEYS = [
    "과속방지턱수",
    "도로안내표지수",
    "무인단속카메라수",
    "보행자우선도로수",
    "신호등수",
    "일방통행도로수",
    "지역특화거리수",
    "옐로카펫수",
]
CONTROL_KEYS = ["주차장수", "버스정류장수"]
FEATURE_COLS = FACILITY_KEYS + ["교통량지표"]

# ── pkl 파일 로드 ─────────────────────────────────────────────────────────────
print("▶ 모델 및 BallTree pkl 로드 중...")


def load_pkl(filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# XGBoost 모델 & 스케일러 (SMOTE 적용 최종 모델)
model = load_pkl("best_model.pkl")
scaler = load_pkl("scaler.pkl")              # MinMaxScaler (9피처)
traffic_scaler = load_pkl("traffic_scaler.pkl")  # StandardScaler (주차장,버스정류장)
print("  ✅ model, scaler, traffic_scaler 로드 완료")

# BallTree (시설물 8종 + 통제변수 2종)
FACILITY_TREES = {}
for nm in FACILITY_KEYS + CONTROL_KEYS:
    FACILITY_TREES[nm] = load_pkl(f"tree_{nm}.pkl")
    print(f"  ✅ tree_{nm}.pkl 로드 완료")

# 어린이보호구역 데이터 (지도 표시용)
zones_data = load_pkl("zones.pkl")
print(f"  ✅ zones.pkl 로드 완료")

print(f"\n✅ 서버 준비 완료 (전국 모델, {len(FEATURE_COLS)}피처)")


# ── 스키마 ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    lat: float = Field(..., example=37.5665, description="위도")
    lng: float = Field(..., example=126.9780, description="경도")


class PredictResponse(BaseModel):
    lat: float
    lng: float
    위험도등급: int
    위험도명: str
    안전확률: float
    주의확률: float
    위험확률: float
    features: dict


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def cnt(nm: str, lat: float, lng: float) -> int:
    """BallTree로 반경 500m 내 시설물 수 반환"""
    return int(
        FACILITY_TREES[nm].query_radius(
            np.radians([[lat, lng]]), r=R_RAD, count_only=True
        )[0]
    )


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "KIDsAfe API", "version": "4.0.0", "scope": "전국", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/app")
def frontend():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/zones")
def get_zones():
    return {"zones": zones_data}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    lat, lng = req.lat, req.lng

    # 전국 범위 검증 (한국 본토)
    if not (33.0 <= lat <= 39.0 and 124.0 <= lng <= 132.0):
        raise HTTPException(
            status_code=400,
            detail="대한민국 내 좌표를 입력해주세요. (위도 33~39 / 경도 124~132)",
        )

    # ── 8개 시설물 카운팅 ─────────────────────────────────────────────────────
    counts = {nm: cnt(nm, lat, lng) for nm in FACILITY_KEYS}

    # ── 통제변수(주차장, 버스정류장) → 교통량지표 생성 ────────────────────────
    ctrl = np.array([[cnt(nm, lat, lng) for nm in CONTROL_KEYS]])
    ctrl_scaled = traffic_scaler.transform(ctrl)   # 학습 시와 동일 StandardScaler
    traffic_idx = float(ctrl_scaled.sum())          # 표준화 합산
    counts["교통량지표"] = traffic_idx

    # ── 피처 벡터 구성 (학습 순서대로) ────────────────────────────────────────
    x = np.array([[counts[f] for f in FEATURE_COLS]])
    x_scaled = scaler.transform(x)

    grade = int(model.predict(x_scaled)[0])
    probs = model.predict_proba(x_scaled)[0]   # 3개 확률

    # 응답용 features (교통량지표는 소수, 나머지는 정수)
    feat_out = {nm: int(counts[nm]) for nm in FACILITY_KEYS}
    feat_out["주차장수"] = int(ctrl[0][0])
    feat_out["버스정류장수"] = int(ctrl[0][1])
    feat_out["교통량지표"] = round(traffic_idx, 3)

    return PredictResponse(
        lat=lat,
        lng=lng,
        위험도등급=grade,
        위험도명=RISK_NAMES[grade],
        안전확률=round(float(probs[0]), 4),
        주의확률=round(float(probs[1]), 4),
        위험확률=round(float(probs[2]), 4),
        features=feat_out,
    )
