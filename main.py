# main.py
"""
KIDsAfe API 서버
위도/경도 입력 → 어린이보호구역 위험도 예측 (XGBoost 이진분류)
데이터: 서버 시작 시 Google Drive에서 자동 다운로드 (캐시 있으면 재사용)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle, os, requests
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ── 앱 초기화 ─────────────────────────────────────────────────────────────────
app = FastAPI(title="KIDsAfe API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
RADIUS_M     = 500
FEATURE_COLS = ['도로폭', 'CCTV대수', '잔여시간표시기수', '음향신호기수',
                '안전표지수', '신호등수', '교차로수', '불법주차_구간']
RISK_NAMES   = {0: '안전', 1: '위험'}
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
DATA_DIR     = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ── Google Drive 파일 ID ──────────────────────────────────────────────────────
DRIVE_FILES = {
    '잔여시간표시기_위경도.csv': '1RF825dRdxpd2C1ykNEMxPC_N9MeFzqwJ',
    '교차로_위경도.csv'        : '1bRd1HqPzA9aB0t2MxJA0yjcst7dP6FUk',
    '음향신호기_위경도.csv'    : '1EBGE0A-Mw2RwcMckois12dtkwFh2f1qV',
    '안전표지_위경도.csv'      : '148I6zXvVdMjmFFUv3le1QhC-9qmRcJqK',
    '신호등_위경도.csv'        : '19GO5VDUbh8jwd-GEJhcPpXArEXCzKsmN',
    '불법주차_위경도.csv'      : '1XETYNinvk5YEI5ipRiCUlicyvwDI96yL',
    '보호구역_500M.csv'        : '1dbXVUz_0fArkUV5uPGu5oeoF-8bYOicH',
}

FACILITY_PATHS = {
    '잔여시간표시기수': os.path.join(DATA_DIR, '잔여시간표시기_위경도.csv'),
    '교차로수'        : os.path.join(DATA_DIR, '교차로_위경도.csv'),
    '음향신호기수'    : os.path.join(DATA_DIR, '음향신호기_위경도.csv'),
    '안전표지수'      : os.path.join(DATA_DIR, '안전표지_위경도.csv'),
    '신호등수'        : os.path.join(DATA_DIR, '신호등_위경도.csv'),
    '불법주차수'      : os.path.join(DATA_DIR, '불법주차_위경도.csv'),
}
BASE_DATA_PATH = os.path.join(DATA_DIR, '보호구역_500M.csv')


# ── Google Drive 다운로드 ─────────────────────────────────────────────────────
def download_from_drive(file_id: str, dest_path: str):
    """gdown으로 Google Drive 파일 다운로드"""
    import gdown
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False, fuzzy=True)

    # 검증
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1000:
        raise ValueError(f"다운로드 실패: {file_id}")
    print(f"  ✅ {os.path.basename(dest_path)} "
          f"({os.path.getsize(dest_path)/1024/1024:.1f} MB)")
# def download_from_drive(file_id: str, dest_path: str):
#     """Google Drive 대용량 파일 확인 토큰 처리 포함 다운로드"""
#     session = requests.Session()

#     # 1차 요청
#     url = f"https://drive.google.com/uc?export=download&id={file_id}"
#     res = session.get(url, stream=True)

#     # 확인 토큰 추출 (대용량 파일 경고 페이지 대응)
#     token = None
#     for key, value in res.cookies.items():
#         if key.startswith('download_warning'):
#             token = value
#             break

#     # 토큰이 없으면 HTML에서 직접 추출 시도
#     if token is None:
#         content = res.content.decode('utf-8', errors='ignore')
#         import re
#         match = re.search(r'confirm=([0-9A-Za-z_\-]+)', content)
#         if match:
#             token = match.group(1)

#     # 2차 요청 (확인 토큰 포함)
#     if token:
#         res = session.get(url, params={'confirm': token,
#                                        'uuid': file_id}, stream=True)

#     # 저장
#     os.makedirs(os.path.dirname(dest_path), exist_ok=True)
#     with open(dest_path, 'wb') as f:
#         for chunk in res.iter_content(chunk_size=32768):
#             if chunk:
#                 f.write(chunk)

#     # 검증: 저장된 파일이 HTML이면 삭제 후 예외 발생
#     with open(dest_path, 'rb') as f:
#         header = f.read(512).decode('utf-8', errors='ignore')
#     if '<!DOCTYPE' in header or '<html' in header:
#         os.remove(dest_path)
#         raise ValueError(f"Drive 다운로드 실패 (HTML 반환): {file_id}")

#     size_mb = os.path.getsize(dest_path) / 1024 / 1024
#     print(f"  ✅ 다운로드 완료: {os.path.basename(dest_path)} ({size_mb:.1f} MB)")

#HuggingFace dataset 적용버전
def download_file(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    res = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in res.iter_content(chunk_size=32768):
            if chunk: f.write(chunk)
    print(f"  ✅ {os.path.basename(dest_path)} "
          f"({os.path.getsize(dest_path)/1024/1024:.1f} MB)")

def ensure_data_files():
    for filename, file_id in DRIVE_FILES.items():
        dest = os.path.join(DATA_DIR, filename)
        if not os.path.exists(dest):
            print(f"  ⬇ {filename} 다운로드 중...")
            download_from_drive(file_id, dest)
            print(f"  ✅ {filename} ({os.path.getsize(dest)/1024/1024:.1f} MB)")
        else:
            print(f"  ✓  {filename} (캐시)")


# ── 서버 시작 시 초기화 ───────────────────────────────────────────────────────
print("▶ 데이터 파일 확인 중...")
ensure_data_files()

print("▶ 모델 로드 중...")
with open(os.path.join(MODEL_DIR, 'xgb_model.pkl'), 'rb') as f:
    xgb_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
print("  ✅ XGBoost 모델 로드 완료")

print("▶ BallTree 빌드 중...")

def load_csv(path):
    opts = dict(
        on_bad_lines='skip',      # 파싱 오류 행 건너뜀
        quoting=0,                # 따옴표 처리 허용
        engine='python',          # C 파서 대신 Python 파서 사용
    )
    try:
        return pd.read_csv(path, encoding='utf-8', **opts)
    except:
        return pd.read_csv(path, encoding='cp949', **opts)

# def build_tree(path):
#     df = load_csv(path)
#     for lc, nc in [('Y','X'),('위도','경도'),('y좌표','x좌표'),('lat','lng')]:
#         if lc in df.columns and nc in df.columns:
#             fac  = df[[lc, nc]].apply(pd.to_numeric, errors='coerce').dropna()
#             tree = BallTree(np.radians(fac[[lc, nc]].values), metric='haversine')
#             return tree, len(fac)
#     raise ValueError(f"좌표 컬럼 없음: {path}")
def build_tree(path):
    try:
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip', engine='python')
    except:
        try:
            df = pd.read_csv(path, encoding='cp949', on_bad_lines='skip', engine='python')
        except Exception as e:
            raise ValueError(f"CSV 로드 실패: {path} / {e}")

    print(f"  [디버그] {os.path.basename(path)}")
    print(f"  [디버그] 컬럼: {list(df.columns)}")
    print(f"  [디버그] shape: {df.shape}")
    print(f"  [디버그] 첫행: {df.iloc[0].to_dict() if len(df) > 0 else 'EMPTY'}")

    for lc, nc in [('Y','X'),('위도','경도'),('y좌표','x좌표'),
                   ('Y좌표','X좌표'),('lat','lng'),('latitude','longitude')]:
        if lc in df.columns and nc in df.columns:
            fac  = df[[lc, nc]].apply(pd.to_numeric, errors='coerce').dropna()
            tree = BallTree(np.radians(fac[[lc, nc]].values), metric='haversine')
            return tree, len(fac)

    raise ValueError(f"좌표 컬럼 없음: {path} / 전체 컬럼: {list(df.columns)}")

FACILITY_TREES = {}
for nm, path in FACILITY_PATHS.items():
    FACILITY_TREES[nm], size = build_tree(path)
    print(f"  ✓ {nm}: {size:,}건")

# 기준 데이터 — 도로폭/CCTV 평균 & 불법주차 분위수
df_base = load_csv(BASE_DATA_PATH)
df_base[['위도','경도']] = df_base[['위도','경도']].apply(pd.to_numeric, errors='coerce')
df_base = df_base.dropna(subset=['위도','경도']).drop_duplicates(subset=['위도','경도'])

def to_median(val):
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

df_base['도로폭']   = df_base['보호구역도로폭'].apply(to_median) \
                      if '보호구역도로폭' in df_base.columns else np.nan
df_base['CCTV대수'] = pd.to_numeric(df_base.get('CCTV설치대수', pd.Series(dtype=float)),
                                     errors='coerce')

R_RAD        = RADIUS_M / 6_371_000
base_rad     = np.radians(df_base[['위도','경도']].values)
park_counts  = FACILITY_TREES['불법주차수'].query_radius(base_rad, r=R_RAD, count_only=True)
P33          = float(np.percentile(park_counts, 33))
P67          = float(np.percentile(park_counts, 67))
MEAN_ROAD    = float(df_base['도로폭'].mean())
MEAN_CCTV    = float(df_base['CCTV대수'].mean())

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

    if not (37.2 <= lat <= 37.8 and 126.5 <= lng <= 127.3):
        raise HTTPException(status_code=400,
            detail="서울 내 좌표를 입력해주세요. (위도 37.2~37.8 / 경도 126.5~127.3)")

    fv = {'도로폭': MEAN_ROAD, 'CCTV대수': MEAN_CCTV}

    def cnt(nm): return int(FACILITY_TREES[nm].query_radius(
        np.radians([[lat, lng]]), r=R_RAD, count_only=True)[0])

    for nm in ['잔여시간표시기수','음향신호기수','안전표지수','신호등수','교차로수']:
        fv[nm] = cnt(nm)

    park_cnt          = cnt('불법주차수')
    fv['불법주차_구간'] = 0 if park_cnt <= P33 else (1 if park_cnt <= P67 else 2)

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
