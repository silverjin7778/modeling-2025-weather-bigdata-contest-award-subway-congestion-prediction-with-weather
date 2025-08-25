# ---------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------
import os
import pandas as pd
import numpy as np

# 2021~2023년 지하철 데이터 불러오기 (세 연도 concat)
df23 = pd.read_csv('./data/train_subway23.csv', encoding='CP949')
df22 = pd.read_csv('./data/train_subway22.csv', encoding='CP949')
df21 = pd.read_csv('./data/train_subway21.csv', encoding='CP949')
df = pd.concat([df21, df22, df23], axis=0)

# 환승역 정보 불러오기
t = pd.read_excel('./data/환승역.xlsx', names =['Line','station_name','transfer'], header=0)

# 역 주소 데이터 불러오기
address = pd.read_csv('./data/result_address.csv', encoding='CP949')

# 일부 신규 역(수기로 추가된 역) merge 준비
subway_13 = pd.DataFrame({'역명':['성수E', '응암S','불암산']
             ,'주소':['서울 성동구 아차산로 100','서울 은평구 증산로 477','서울 노원구 상계로 305']})
address = pd.concat([address, subway_13], axis=0).reset_index(drop=True)

# 컬럼 정리
address.columns=['station_name','address']
# 괄호 안 제거 (ex. 자양(뚝섬) -> 자양)
address.station_name = address.station_name.apply(lambda x: x.split('(')[0].strip() if '(' in x else x)
# 주소 앞부분만 추출 (서울/경기/인천 구분)
address.address = address.address.apply(lambda x: x.split()[0] if '서울' not in x else x.split()[1])
addr = address['address']  
address['address'] = np.where(addr.str.contains('인천'), '인천',np.where(addr.str.contains('경기'), '경기', addr))

print(df.shape)
df.head()


# ---------------------------------------------------------------
# 데이터 형변환
# ---------------------------------------------------------------

# 날짜형 변환
df['TM'] = pd.to_datetime(df['TM'], format='%Y%m%d%H')
df = df.sort_values('TM').reset_index(drop=True)

# 범주형 컬럼 정의 및 변환
cat_columns = 'Line station_number STN station_name Direction'.split()
for col in cat_columns:
    df[col] = df[col].astype('category')


# ---------------------------------------------------------------
# 결측치 생성/처리
# ---------------------------------------------------------------
# 기상 변수에서 -99, 음수 등을 NaN 처리
df['WD'] = df['WD'].where(df['WD'] >= 0, np.nan)
df['WS'] = df['WS'].replace(-99.0, np.nan)
df['RN_DAY'] = df['RN_DAY'].replace(-99.0, np.nan)
df['RN_HR1'] = df['RN_HR1'].replace(-99.0, np.nan)
df['TA'] = df['TA'].replace(-99.0, np.nan)
df['ta_chi'] = df['ta_chi'].replace(-99.0, np.nan)
df['SI'] = df['SI'].replace(-99.0, np.nan)
df['HM'] = df['HM'].replace(-99.0, np.nan)

# SI는 결측 아닌 값만 1, 나머지 0 (Binary Feature)
df['SI'] = df['SI'].notna().astype(int)


# ---------------------------------------------------------------
# 기본 파생변수
# ---------------------------------------------------------------
# 역명 통일 (중복/특수 표기 제거)
df.station_name= df.station_name.replace({'당고개':'불암산','자양(뚝섬한강공원)':'자양','신촌(지하)':'신촌'})

# 환승역 정보 merge
df = pd.merge(df, t, on=['Line','station_name'], how='left') 
df['transfer'] = df['transfer'].fillna(0).astype(int)

# 주소 merge
df = pd.merge(df, address, on=['station_name'], how='left')

# key 생성 (Line_역명_Direction 조합)
df['key'] = (df['Line'].astype(str) + '_' +df['station_name'].astype(str) + '_' +df['Direction'].astype(str))

# 날짜 기반 파생 변수 생성
df['year'] = df['TM'].dt.year - 2021 # 연도(상대값)
df['month'] = df['TM'].dt.month
df['day'] = df['TM'].dt.day
df['hour'] = df['TM'].dt.hour
df['weekday'] = df['TM'].dt.weekday
df['week_of_month'] = (df['TM'].dt.day.sub(1) // 7) + 1
df['week_of_year'] = df['TM'].dt.isocalendar().week
df['day_of_year'] = df['TM'].dt.dayofyear

# 공휴일 여부 / 전후일 여부 / 주말 여부 생성
from holidayskr import year_holidays
dates_only1 = [d[0] for d in year_holidays('2021')]
dates_only2 = [d[0] for d in year_holidays('2022')]
dates_only3 = [d[0] for d in year_holidays('2023')]
dates_only3 = [d[0] for d in year_holidays('2024')]

cond1 = df['TM'].isin(dates_only1)
cond2 = df['TM'].isin(dates_only2)
cond3 = df['TM'].isin(dates_only3)
cond4 = df['TM'].isin(dates_only3)

df['is_holiday'] = (cond1 | cond2 | cond3 | cond4).astype(int)
df['is_weekend'] = df['TM'].dt.dayofweek # → 주말 여부는 0/1 처리 권장
df['is_day_before_holiday'] = df['TM'].shift(-1).isin(dates_only1 + dates_only2 + dates_only3).astype(int)
df['is_day_after_holiday'] = df['TM'].shift(1).isin(dates_only1 + dates_only2 + dates_only3).astype(int)

# 시간대 구분 (출근/퇴근/낮/저녁/밤)
df['time_period'] = np.where(df['hour'].isin([7,8,9]), '출근',
                                np.where(df['hour'].isin([17,18,19]), '퇴근',
                                np.where((df['hour']>9)&(df['hour']<17), '낮',
                                np.where((df['hour']>19)&(df['hour']<21), '저녁',
                                '밤'))))


# ---------------------------------------------------------------
# 데이터 형 변환 (순서형 카테고리 인코딩)
# ---------------------------------------------------------------
from pandas.api.types import CategoricalDtype

# 방향 순서 지정
direction_order = ['상선', '하선', '외선', '내선']
time_period_order = ['밤', '출근', '낮', '저녁', '퇴근']

# 카테고리형을 순서형 코드로 변환
df['Direction'] = df['Direction'].astype(CategoricalDtype(categories=direction_order, ordered=True)).cat.codes
df['time_period'] = df['time_period'].astype(CategoricalDtype(categories=time_period_order, ordered=True)).cat.codes


# ---------------------------------------------------------------
# 주기성 변수 (sin/cos 변환) → 계절성과 주기성 반영
# ---------------------------------------------------------------
df['sin_hod'] = np.sin(df['hour'] * (2 * np.pi / 21)) # 하루(21시간) 내 시간
df['cos_hod'] = np.cos(df['hour'] * (2 * np.pi / 21))
df['sin_dow'] = np.sin(df['weekday'] * (2 * np.pi / 7)) # 요일
df['cos_dow'] = np.cos(df['weekday'] * (2 * np.pi / 7))
df['sin_dom'] = np.sin(df['day'] * (2 * np.pi / 31)) # 월별 날짜
df['cos_dom'] = np.cos(df['day'] * (2 * np.pi / 31))
df['sin_wom'] = np.sin(df['week_of_month'] * (2 * np.pi / 5)) # 월별 주차
df['cos_wom'] = np.cos(df['week_of_month'] * (2 * np.pi / 5))
df['sin_woy'] = np.sin(df['week_of_year'] * (2 * np.pi / 52)) # 연중 주차
df['cos_woy'] = np.cos(df['week_of_year'] * (2 * np.pi / 52))
df['sin_doy'] = np.sin(df['day_of_year'] * (2 * np.pi / 365)) # 연중 일차
df['cos_doy'] = np.cos(df['day_of_year'] * (2 * np.pi / 365))

# key, TM 제거 후 정렬
df = df.sort_values(['key','TM'])
df = df.drop(columns=['TM','key'])


# ---------------------------------------------------------------
# 선형 보간으로 결측치 보완
# ---------------------------------------------------------------
columns_to_fill = 'WD RN_DAY RN_HR1 TA ta_chi SI HM WS'.split()
df[columns_to_fill] = df[columns_to_fill].interpolate(method='linear', limit_direction='both')
print('보간 후 남은 결측값:', df[columns_to_fill].isna().sum())


# ---------------------------------------------------------------
# 스케일링 및 타깃 로그 변환
# ---------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer

# 로그 변환 (혼잡도 값 안정화)
df['Congestion'] = np.log1p(df['Congestion'])

# 주요 수치형 변수별 스케일링 기법 다르게 적용
scalers = {
    'ta_chi': StandardScaler(),
    'HM': QuantileTransformer(output_distribution='normal', random_state=0),
    'RN_HR1': MinMaxScaler(),
    'RN_DAY': MinMaxScaler(),
    'WS': PowerTransformer(method='yeo-johnson'),
    'TA': RobustScaler()
}
for col, scaler in scalers.items():
    if 'RN' in col:
        df[col] = np.log1p(df[col])  # 강우량 로그 변환
    df[col] = scaler.fit_transform(df[[col]])


# ---------------------------------------------------------------
# 모델 입력 구성
# ---------------------------------------------------------------
from tensorflow.keras import layers, Model, Input

# Temporal Stream: 시간 관련 파생 변수
temporal_input = Input(shape=(X_train_temporal.shape[1],), name='temporal')
temporal_out = layers.Dense(64, activation='relu')(temporal_input)

# Station Stream: 역/노선 Embedding
station_input = Input(shape=(1,), name='station')
station_emb = layers.Embedding(input_dim=unknown_idx + 1, output_dim=8)(station_input)
station_flat = layers.Flatten()(station_emb)

# Weather Stream: 기상 변수
weather_input = Input(shape=(X_train_weather.shape[1],), name='weather')
weather_out = layers.Dense(32, activation='relu')(weather_input)

# Address Stream: 지역 Embedding
address_input = Input(shape=(1,), name='address')
address_emb = layers.Embedding(input_dim=train_df['address'].nunique() + 2, output_dim=4)(address_input)
address_flat = layers.Flatten()(address_emb)

# Time Period Stream: 출근/퇴근 Embedding
tp_input = Input(shape=(1,), name='time_period')
tp_emb = layers.Embedding(input_dim=train_df['tp_idx'].nunique() + 2, output_dim=4)(tp_input)
tp_flat = layers.Flatten()(tp_emb)

# 멀티스트림 결합
x = layers.concatenate([temporal_out, station_flat, weather_out, address_flat, tp_flat])
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1)(x)

# 모델 정의
model = Model(inputs=[temporal_input, station_input, weather_input, address_input, tp_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()


# ---------------------------------------------------------------
# Train / Validation 데이터 분리
# ---------------------------------------------------------------

# 인스턴스 key (노선_역명_방향) 생성
df['key'] = df['Line'].astype(str) + '_' + df['station_name'].astype(str) + '_' + df['Direction'].astype(str)

# 사용할 피처 정의
features = num_cols + ordered_cols + cat_cols + ['key', 'Congestion']

# 연도 기준 분리: 2021,2022 → train / 2023 → validation
train_df = pd.concat([df[df['year'] == 0], df[df['year'] == 1]])[features]
val_df   = df[df['year'] == 2][features]

# 범주형 인코딩: train의 카테고리를 기준으로 val도 맞춰줌
for col in ['Line', 'station_name', 'address']:
    train_df[col] = train_df[col].astype('category')
    val_df[col]   = val_df[col].astype('category')
    # validation이 train과 같은 카테고리 집합을 가지도록 설정
    val_df[col]   = val_df[col].cat.set_categories(train_df[col].cat.categories)
    # 카테고리를 코드(0,1,2…)로 변환
    train_df[col] = train_df[col].cat.codes
    val_df[col]   = val_df[col].cat.codes

# ---------------------------------------------------------------
# station index 매핑 (Embedding용 인덱스 생성)
# ---------------------------------------------------------------
station_keys = train_df['key'].unique().tolist()
station_dict = {k: i for i, k in enumerate(station_keys)}
unknown_idx = len(station_dict)  # train에 없는 key → unknown 처리
train_df['station_idx'] = train_df['key'].map(lambda x: station_dict.get(x, unknown_idx))
val_df['station_idx']   = val_df['key'].map(lambda x: station_dict.get(x, unknown_idx))

# ---------------------------------------------------------------
# time_period 인코딩
# ---------------------------------------------------------------
le_tp = pd.concat([train_df['time_period'], val_df['time_period']]).astype('category')
train_df['tp_idx'] = le_tp.loc[train_df.index].cat.codes
val_df['tp_idx']   = le_tp.loc[val_df.index].cat.codes


# ---------------------------------------------------------------
# 스트림별 데이터 분리 (입력 배열 준비)
# ---------------------------------------------------------------

# 기상 변수 / 시간 파생 변수 분리
weather_cols = ['TA', 'WD', 'WS', 'RN_DAY', 'RN_HR1', 'HM', 'ta_chi']
num_cols = [col for col in num_cols if col not in weather_cols]

# Temporal: 기상 제외한 시간 변수
X_train_temporal = train_df[num_cols].values
X_val_temporal   = val_df[num_cols].values

# Weather: 기상 변수
X_train_weather = train_df[weather_cols].values
X_val_weather   = val_df[weather_cols].values

# Station: station_idx
X_train_station = train_df['station_idx'].values.reshape(-1, 1)
X_val_station   = val_df['station_idx'].values.reshape(-1, 1)

# Address: address index
X_train_address = train_df['address'].values.reshape(-1, 1)
X_val_address   = val_df['address'].values.reshape(-1, 1)

# Time Period: tp index
X_train_tp = train_df['tp_idx'].values.reshape(-1, 1)
X_val_tp   = val_df['tp_idx'].values.reshape(-1, 1)

# 타깃 (로그 변환된 혼잡도)
y_train = train_df['Congestion'].values
y_val   = val_df['Congestion'].values


# ---------------------------------------------------------------
# 데이터 타입 변환 (float32 / int32)
# ---------------------------------------------------------------
X_train_temporal = X_train_temporal.astype(np.float32)
X_val_temporal   = X_val_temporal.astype(np.float32)

X_train_weather  = X_train_weather.astype(np.float32)
X_val_weather    = X_val_weather.astype(np.float32)

X_train_station  = X_train_station.astype(np.int32)
X_val_station    = X_val_station.astype(np.int32)

X_train_address  = X_train_address.astype(np.int32)
X_val_address    = X_val_address.astype(np.int32)

X_train_tp       = X_train_tp.astype(np.int32)
X_val_tp         = X_val_tp.astype(np.int32)

y_train = y_train.astype(np.float32)
y_val   = y_val.astype(np.float32)


# ---------------------------------------------------------------
# Custom Callback: 로그 변환된 타깃을 원래 스케일로 복원해 RMSE 계산
# ---------------------------------------------------------------
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf

class OrigValRMSE(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        # validation_data: ({입력 딕셔너리}, 로그스케일 y_val)
        self.x_val, self.y_val_log = validation_data

    def on_epoch_end(self, epoch, logs=None):
        # 1) validation 예측 (로그 스케일)
        y_pred_log = self.model.predict(self.x_val, verbose=0)
        # 2) 로그 스케일 → 원래 스케일(exp 역변환)
        y_pred = np.expm1(y_pred_log).ravel()
        y_true = np.expm1(self.y_val_log).ravel()
        # 3) RMSE 계산
        rmse_orig = mean_squared_error(y_true, y_pred, squared=False)
        # 4) 결과 출력 및 logs에 기록
        print(f' — orig_val_RMSE: {rmse_orig:.4f}')
        if logs is not None:
            logs['orig_val_RMSE'] = rmse_orig

# 콜백 객체 생성
orig_rmse_cb = OrigValRMSE(
    validation_data=(
        {
            'temporal': X_val_temporal,
            'station':  X_val_station,
            'weather':  X_val_weather,
            'address':  X_val_address,
            'time_period': X_val_tp
        },
        y_val   # 로그 변환된 y_val
    )
)


# ---------------------------------------------------------------
# 모델 학습
# ---------------------------------------------------------------
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

save_path = os.path.join('checkpoints', 'multi_hybrid_extended.keras')

history = model.fit(
    x={
        'temporal': X_train_temporal,
        'station':  X_train_station,
        'weather':  X_train_weather,
        'address':  X_train_address,
        'time_period': X_train_tp
    },
    y=y_train,  # 로그 변환된 y_train
    validation_data=(
        {
            'temporal': X_val_temporal,
            'station':  X_val_station,
            'weather':  X_val_weather,
            'address':  X_val_address,
            'time_period': X_val_tp
        },
        y_val  # 로그 변환된 y_val
    ),
    epochs=50,
    batch_size=1024,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),    # 조기 종료
        ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True),  # 체크포인트 저장
        orig_rmse_cb   # 원래 스케일 RMSE 모니터링
    ]
)


# ---------------------------------------------------------------
# 예측 및 역변환
# ---------------------------------------------------------------

# 모델 불러오기
from tensorflow.keras.models import load_model
model = load_model('./checkpoints/multi_hybrid_extended.keras')

# 예측 (로그 스케일 → 원래 스케일)
y_pred_log = model.predict({
    'temporal': X_val_temporal,
    'station':  X_val_station,
    'weather':  X_val_weather,
    'address':  X_val_address,
    'time_period': X_val_tp
}, batch_size=1024)

y_pred = np.expm1(y_pred_log)   # expm1로 복원

# 음수값 방지: 0 이하 값은 0으로 처리
y_pred = np.where(y_pred <= 0, 0, y_pred)

# 정수 반올림 (submission 형식 맞춤)
y_int = np.rint(y_pred.ravel()).astype(int)

# 저장
submission = pd.DataFrame({'Congestion': y_int})
submission.to_csv('./test/submission.csv', index=False)

# 결과 확인
pd.set_option('display.float_format', '{:.0f}'.format)
print(pd.read_csv('./test/submission.csv')['Congestion'].describe())

