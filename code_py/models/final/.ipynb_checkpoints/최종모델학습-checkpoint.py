# -----------------------------------------------------------
# 데이터 불러오기
# -----------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'

import pandas as pd
df = pd.read_parquet(
    'data.parquet',
    engine='pyarrow'         # 저장 시 사용한 엔진과 동일하게 지정
)
test_loaded = pd.read_parquet(
    'test.parquet',
    engine='pyarrow'         # 저장 시 사용한 엔진과 동일하게 지정
)

## 최종 모델 : xgboost + 전체 데이터 학습

# feature / target 정의
ordered_cols = ['Direction', 'time_period']
cat_cols     = [
                'station_number'
                , 'address'
               ] + ordered_cols
num_cols = [
    'HM','RN_DAY','RN_HR1',
    'WD','WS'
    ,'STN'
    ,'sin_dom','cos_dom','sin_dow','cos_dow','sin_hod','cos_hod'
    ,'sin_wom','cos_wom','sin_woy','cos_woy','sin_doy','cos_doy'
    ,'day','day_of_year','hour'
    ,'is_day_before_holiday','is_day_after_holiday','is_holiday','is_weekend'
    ,'month','transfer','week_of_month','week_of_year','weekday','year'
    ,'신설역', '신규관측소'
]
feature_cols = num_cols + ordered_cols + cat_cols
target_col   = 'Congestion'

results = []
final_results = []
print('완료')

import os
import time
import gc
import numpy as np
import pandas as pd
import joblib
import random
from tqdm import tqdm
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# RMSE·R2 평가함수 (train 성능 확인용)
def evaluate_model(name, model, line, X_train, y_train):
    t0 = time.time()
    y_pred = model.predict(X_train)
    elapsed = time.time() - t0
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    r2 = r2_score(y_train, y_pred)
    return {'Model': name, 'Line': line, 'Time(s)': elapsed, 'RMSE': rmse, 'R2': r2}

results = []
final_results = []



# -----------------------------------------------------------
# 호선별 모델 학습 시작
# -----------------------------------------------------------
for line in df['Line'].unique():
    print(f"\n📘 [Line {line}] Training model...")

    df_line   = df[df['Line'] == line].copy()
    test_line = test_loaded[test_loaded['Line'] == line].copy()

    # 카테고리형 처리
    for col in ['STN', 'address']:
        df_line[col]   = df_line[col].astype('category')
        test_line[col] = test_line[col].astype('category')

    df_line = df_line.sort_values('TM')

    # feature / target / test 준비
    X       = df_line[feature_cols]
    y       = df_line[target_col].astype(int)
    X_test  = test_line[feature_cols]


    # one-hot encoding
    X_enc      = pd.get_dummies(X,       columns=cat_cols)
    X_enc      = X_enc.loc[:, ~X_enc.columns.duplicated()]
    # 1) X_test_enc 생성
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols)
    
    # 2) 중복 컬럼 제거 (테스트용에도 반드시!)
    X_test_enc = X_test_enc.loc[:, ~X_test_enc.columns.duplicated()]
    
    # 3) X_enc 기준으로 컬럼 맞추기
    X_test_enc = X_test_enc.reindex(columns=X_enc.columns, fill_value=0)


    # scaling
    mm             = MinMaxScaler()
    X_scaled       = mm.fit_transform(X_enc).astype(np.float32)
    X_test_scaled  = mm.transform(X_test_enc).astype(np.float32)
    
    # -----------------------------------------------------------
    # 하이퍼파라미터 테스트를 통해 최종 학습된 하이퍼파라미터로 학습
    # -----------------------------------------------------------
    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=12,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=0.8,
        min_child_weight=3,
        gamma=0,
        tree_method='hist',
        verbosity=0,
        random_state=42
    )

    # 전체 train으로 학습
    model.fit(X_scaled, y)

    # train 성능 확인
    res = evaluate_model('XGB', model, line, X_scaled, y)
    results.append(res)
    print('Train Result:', res)

    # test 예측
    y_pred = np.round(model.predict(X_test_scaled)).astype(int)
    y_pred = np.clip(y_pred, 0, None)

    temp = test_line[['hour', 'Line', 'station_number']].copy()
    temp['Predicted_Congestion'] = y_pred
    final_results.append(temp)

    # 모델 저장
    os.makedirs('./models', exist_ok=True)
    joblib.dump(model, f"./models/xgb_line{line}.pkl", compress=3)

    # 메모리 해제
    del df_line, test_line, X, y, X_test
    del X_enc, X_test_enc, X_scaled, X_test_scaled, model, y_pred, temp
    gc.collect()


# -----------------------------------------------------------
# 결과 합치기 및 성능 검증
# -----------------------------------------------------------
final_df = pd.concat(final_results)
output_df = final_df[['Predicted_Congestion']].rename(columns={'Predicted_Congestion': 'Congestion'})
os.makedirs('./test', exist_ok=True)
output_df.to_csv('./test/250206-all.csv', index=False, encoding='utf-8')

results_df = pd.DataFrame(results)
print("\n 성능요약:")
print(results_df)

gap = pd.read_csv('./test/250206-all.csv') # 내 데이터
gap.shape

import os
제출 = pd.read_csv('minjeong.csv') # 성능 잘 나왔던 비교용 데이터

import numpy as np

from sklearn.metrics import root_mean_squared_error
rmse = np.sqrt(mean_squared_error(
    제출['Congestion'],
    gap['Congestion']    # squared=False 하면 RMSE 를 직접 계산해 줌
))

print(f"RMSE: {rmse:.4f}")