# main.py
"""
KIDsAfe API 서버
위도/경도 입력 → 어린이보호구역 위험도 예측 (XGBoost 이진분류)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import os

# ── 앱 초기화 ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KIDsAfe API",
    description="어린이보호구역 위험도 예측 API",
    version="1.0.0"
)

# CORS 설정 (프론트엔드에서 호출 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 상수 ──────────────────────────────────────────────────────────────────────
RADIUS_M     = 500
FEATURE_COLS = ['도로폭', 'CCTV대수', '잔여시간표시기수', '음향신호기수',
                '안전표지수', '신호등수', '교차로수', '불법주차_구간']
RISK_NAMES   = {0: '안전', 1: '위험'}
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
DATA_DIR     = os.path.join(BASE_DIR, 'data')


# ── 모델 & 데이터 로드 (서버 시작 시 1회) ────────────────────────────────────
print("▶ 모델 로드 중...")

with open(os.path.join(MODEL_DIR, 'xgb_model.pkl'), 'rb') as f:
    xgb_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

print("▶ 시설물 데이터 로드 중...")

FACILITY_PATHS = {
    '잔여시간표시기수': os.path.join(DATA_DIR, '서울시 잔여시간표시기 관련 정보-위경도(추가).csv'),
    '교차로수'        : os.path.join(DATA_DIR, '서울시 교차로 관련 정보-위경도(추가).csv'),
    '음향신호기수'    : os.path.join(DATA_DIR, '서울시 음향신호기 관련 정보-위경도(추가).csv'),
    '안전표지수'      : os.path.join(DATA_DIR, '서울시 안전표지 관련 정보-위경도(추가).csv'),
    '신호등수'        : os.path.join(DATA_DIR, '서울시 신호등 관련 정보-위경도(추가).csv'),
    '불법주차수'      : os.path.join(DATA_DIR, '서울시_불법주정차단속_위치정보_병합.csv'),
}

# 시설물 BallTree 사전 빌드 (요청마다 재구성하지 않고 캐시)
FACILITY_TREES = {}
FACILITY_SIZES = {}

def build_tree(path):
    """csv 로드 후 BallTree 생성"""
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='cp949')

    COORD_CANDS = [('Y','X'),('위도','경도'),('y좌표','x좌표'),('lat','lng')]
    cols = set(df.columns)
    lat_c = lng_c = None
    for lc, nc in COORD_CANDS:
        if lc in cols and nc in cols:
            lat_c, lng_c = lc, nc; break

    if lat_c is None:
        raise ValueError(f"좌표 컬럼 없음: {path}")

    fac = df[[lat_c, lng_c]].apply(pd.to_numeric, errors='coerce').dropna()
    tree = BallTree(np.radians(fac[[lat_c, lng_c]].values), metric='haversine')
    return tree, len(fac)

for nm, path in FACILITY_PATHS.items():
    FACILITY_TREES[nm], FACILITY_SIZES[nm] = build_tree(path)
    print(f"  ✓ {nm}: {FACILITY_SIZES[nm]:,}건 인덱싱 완료")

# 기준 데이터 (도로폭/CCTV 평균, 불법주차 분위수)
df_base = pd.read_csv(
    os.path.join(DATA_DIR, '서울어린이보호구역(2019-2024)기준_교통사고다발지역_500M.csv')
)
df_base[['위도','경도']] = df_base[['위도','경도']].apply(pd.to_numeric, errors='coerce')
df_base = df_base.dropna(subset=['위도','경도']).drop_duplicates(subset=['위도','경도'])

def _to_median(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    for d in ['~', '-']:
        if d in s:
            try:
                parts = [float(x.strip()) for x in s.split(d) if x.strip()]
                if len(parts) == 2: return sum(parts) / 2
            except: return np.nan
    try: return float(s)
    except: return np.nan

df_base['도로폭']   = df_base['보호구역도로폭'].apply(_to_median) if '보호구역도로폭' in df_base.columns else np.nan
df_base['CCTV대수'] = pd.to_numeric(df_base.get('CCTV설치대수', np.nan), errors='coerce')

# 불법주차 분위수 캐시
_park_tree = FACILITY_TREES['불법주차수']
_r_rad = RADIUS_M / 6_371_000
_base_rad = np.radians(df_base[['위도','경도']].values)
_park_counts = _park_tree.query_radius(_base_rad, r=_r_rad, count_only=True)
P33 = float(np.percentile(_park_counts, 33))
P67 = float(np.percentile(_park_counts, 67))

MEAN_ROAD_WIDTH = float(df_base['도로폭'].mean())
MEAN_CCTV       = float(df_base['CCTV대수'].mean())

print(f"\n✅ 서버 준비 완료")
print(f"   도로폭 평균: {MEAN_ROAD_WIDTH:.1f}m | CCTV 평균: {MEAN_CCTV:.1f}대")
print(f"   불법주차 분위수: P33={P33:.0f} / P67={P67:.0f}")


# ── 요청/응답 스키마 ──────────────────────────────────────────────────────────
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


# ── 유틸: 반경 내 카운팅 ──────────────────────────────────────────────────────
def count_one_point(tree: BallTree, lat: float, lng: float) -> int:
    point_rad = np.radians([[lat, lng]])
    r_rad     = RADIUS_M / 6_371_000
    return int(tree.query_radius(point_rad, r=r_rad, count_only=True)[0])


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "KIDsAfe API", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    lat, lng = req.lat, req.lng

    # 서울 범위 검증
    if not (37.2 <= lat <= 37.8 and 126.5 <= lng <= 127.3):
        raise HTTPException(
            status_code=400,
            detail="서울 내 좌표를 입력해주세요. (위도 37.2~37.8 / 경도 126.5~127.3)"
        )

    # ── 피처 생성 ──────────────────────────────────────────────────────────────
    feature_vals = {
        '도로폭'  : MEAN_ROAD_WIDTH,
        'CCTV대수': MEAN_CCTV,
    }

    for nm in ['잔여시간표시기수', '음향신호기수', '안전표지수', '신호등수', '교차로수']:
        feature_vals[nm] = count_one_point(FACILITY_TREES[nm], lat, lng)

    park_cnt = count_one_point(FACILITY_TREES['불법주차수'], lat, lng)
    feature_vals['불법주차_구간'] = 0 if park_cnt <= P33 else (1 if park_cnt <= P67 else 2)

    # ── 예측 ───────────────────────────────────────────────────────────────────
    x        = np.array([[feature_vals.get(f, 0) for f in FEATURE_COLS]])
    x_scaled = scaler.transform(x)
    grade    = int(xgb_model.predict(x_scaled)[0])
    probs    = xgb_model.predict_proba(x_scaled)[0]

    return PredictResponse(
        lat=lat,
        lng=lng,
        위험도등급=grade,
        위험도명=RISK_NAMES[grade],
        안전확률=round(float(probs[0]), 4),
        위험확률=round(float(probs[1]), 4),
        features=FeatureDetail(
            도로폭=round(feature_vals['도로폭'], 2),
            CCTV대수=round(feature_vals['CCTV대수'], 2),
            잔여시간표시기수=feature_vals['잔여시간표시기수'],
            음향신호기수=feature_vals['음향신호기수'],
            안전표지수=feature_vals['안전표지수'],
            신호등수=feature_vals['신호등수'],
            교차로수=feature_vals['교차로수'],
            불법주차_구간=feature_vals['불법주차_구간'],
            불법주차_원본건수=park_cnt,
        )
    )
