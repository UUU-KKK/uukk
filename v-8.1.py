"""
V8.1 GPU加速版：加入PyTorch神经网络，目标LB > 0.954
"""
import time, warnings, gc, json
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def log(msg):
    print(msg, flush=True)


# ====================== PyTorch模型定义 ======================
class F1Dataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)


class F1PitPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(512, 0.3),
            ResidualBlock(512, 0.3),
            ResidualBlock(256, 0.3),
            ResidualBlock(256, 0.3)
        )

        self.downsample = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks[0](x)
        x = self.residual_blocks[1](x)
        x = self.downsample(x)
        x = self.residual_blocks[2](x)
        x = self.residual_blocks[3](x)
        return self.output_layer(x)


def train_nn_fold(X_tr, y_tr, X_val, y_val, input_dim, device, seed=42):
    """训练单个折的神经网络（GPU优化版）"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 标准化
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    # 创建数据加载器（GPU优化batch size）
    train_dataset = F1Dataset(X_tr_scaled, y_tr)
    val_dataset = F1Dataset(X_val_scaled, y_val)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=0, pin_memory=True)

    # 初始化模型
    model = F1PitPredictor(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, min_lr=1e-6)

    # 训练循环
    best_auc = 0
    best_state = None
    patience = 15
    counter = 0

    for epoch in range(100):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_dataset)

        # 验证阶段
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                outputs = model(batch_X).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        val_auc = roc_auc_score(val_true, val_preds)
        scheduler.step(val_auc)

        # 早停
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    # 加载最佳模型
    model.load_state_dict(best_state)
    model.eval()

    # 生成验证集预测
    val_preds = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            outputs = model(batch_X).squeeze()
            val_preds.extend(outputs.cpu().numpy())

    return model, scaler, np.array(val_preds), best_auc


def predict_nn(model, scaler, X, device):
    """使用训练好的模型进行预测（GPU优化版）"""
    X_scaled = scaler.transform(X)
    dataset = F1Dataset(X_scaled)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch_X in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            outputs = model(batch_X).squeeze()
            preds.extend(outputs.cpu().numpy())

    return np.array(preds)


# ============================================================

SEEDS = [42, 123, 2024, 456, 789]
NF = 5
t0 = time.time()

log("=" * 70)
log("V8.1 GPU加速版 - 加入PyTorch神经网络")
log("=" * 70)

# 检测GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"\n使用设备: {device}")
if torch.cuda.is_available():
    log(f"显卡型号: {torch.cuda.get_device_name(0)}")
    log(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

# ============================================================
# 1. LOAD DATA
# ============================================================
log("\n[1/6] Loading data...")
train = pd.read_csv('playground-series-s6e5/train.csv')
test = pd.read_csv('playground-series-s6e5/test.csv')
train['PitNextLap'] = train['PitNextLap'].astype(int)
log(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
log("\n[2/6] Feature engineering...")

compound_order = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
gm = 0.1990


def add_features(df):
    df = df.copy()
    df['Compound_ord'] = df['Compound'].map(compound_order)
    co = df['Compound_ord']

    # Core interactions
    df['CxTyreLife'] = co * df['TyreLife']
    df['CxLapNumber'] = co * df['LapNumber']
    df['CxRaceProgress'] = co * df['RaceProgress']
    df['CxDegradation'] = co * df['Cumulative_Degradation']
    df['CxPosition'] = co * df['Position']

    # TyreLife transforms
    tls = df['TyreLife'].clip(lower=1)
    df['TL_sq'] = df['TyreLife'] ** 2
    df['TL_sqrt'] = np.sqrt(df['TyreLife'])
    df['TL_log'] = np.log1p(df['TyreLife'])
    df['TL_cu'] = df['TyreLife'] ** 3

    # TL thresholds
    df['TL_gt_15'] = (df['TyreLife'] > 15).astype(np.int8)
    df['TL_gt_25'] = (df['TyreLife'] > 25).astype(np.int8)
    df['TL_gt_35'] = (df['TyreLife'] > 35).astype(np.int8)
    df['TL_gt_50'] = (df['TyreLife'] > 50).astype(np.int8)

    # Degradation features
    df['Deg_per_lap'] = df['Cumulative_Degradation'] / tls
    df['LTD_per_lap'] = df['LapTime_Delta'] / tls
    df['LT_per_lap'] = df['LapTime (s)'] / tls
    df['Deg_abs'] = np.abs(df['Cumulative_Degradation'])
    df['Deg_sq'] = df['Cumulative_Degradation'] ** 2

    # Position
    df['Pos_sq'] = df['Position'] ** 2
    df['Is_Top10'] = (df['Position'] <= 10).astype(np.int8)

    # Race/Lap
    df['RP_sq'] = df['RaceProgress'] ** 2
    df['RP_cu'] = df['RaceProgress'] ** 3
    df['LN_sq'] = df['LapNumber'] ** 2
    df['LN_sqrt'] = np.sqrt(df['LapNumber'])

    # Stint
    df['Stint_x_TL'] = df['Stint'] * df['TyreLife']
    df['Stint_x_LN'] = df['Stint'] * df['LapNumber']
    df['TL_div_Stint'] = df['TyreLife'] / df['Stint'].clip(lower=1)
    df['LN_div_Stint'] = df['LapNumber'] / df['Stint'].clip(lower=1)

    # Year flags
    for yr in [2022, 2023, 2024, 2025]:
        df[f'Year_{yr}'] = (df['Year'] == yr).astype(np.int8)

    # Other
    df['PChange_abs'] = np.abs(df['Position_Change'])
    df['Has_Pitted'] = df['PitStop'].astype(np.int8)
    df['Tire_Wear'] = -df['Cumulative_Degradation']
    df['LTD_abs'] = np.abs(df['LapTime_Delta'])
    df['LT_sq'] = df['LapTime (s)'] ** 2

    # Label encodings
    df['Driver_le'] = LabelEncoder().fit_transform(df['Driver'])
    df['Race_le'] = LabelEncoder().fit_transform(df['Race'])

    # Key interactions
    df['PitStop_x_TL'] = df['PitStop'] * df['TyreLife']
    df['LT_x_Compound'] = df['LapTime (s)'] * co
    df['RP_x_TL'] = df['RaceProgress'] * df['TyreLife']
    df['Pos_x_TL'] = df['Position'] * df['TyreLife']
    df['LN_x_TL'] = df['LapNumber'] * df['TyreLife']
    df['Deg_x_RP'] = df['Cumulative_Degradation'] * df['RaceProgress']
    rp_safe = df['RaceProgress'].clip(lower=0.01)
    df['TL_div_RP'] = df['TyreLife'] / rp_safe
    df['Stint_x_RP'] = df['Stint'] * df['RaceProgress']

    # 按比赛和车手分组，计算前3圈的平均圈速和退化
    df['Prev3LapAvg'] = df.groupby(['Race', 'Driver'])['LapTime (s)'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    df['Prev3LapDegAvg'] = df.groupby(['Race', 'Driver'])['Cumulative_Degradation'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1))

    # 圈速变化趋势
    df['LapTime_Change'] = df.groupby(['Race', 'Driver'])['LapTime (s)'].diff().fillna(0)
    df['LapTime_Change_Avg'] = df.groupby(['Race', 'Driver'])['LapTime_Change'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1))

    # V8 new features
    compound_avg_tl = {'SOFT': 12.0, 'MEDIUM': 18.0, 'HARD': 25.0, 'INTERMEDIATE': 10.0, 'WET': 8.0}
    df['Compound_avg_TL'] = df['Compound'].map(compound_avg_tl)
    df['Dev_compound_TL'] = df['TyreLife'] - df['Compound_avg_TL']
    df['TL_ratio_compound'] = df['TyreLife'] / df['Compound_avg_TL'].clip(lower=1)

    df['Deg_x_Compound'] = df['Cumulative_Degradation'] * co
    df['LTD_x_Compound'] = df['LapTime_Delta'] * co
    df['RP_div_Stint'] = df['RaceProgress'] / df['Stint'].clip(lower=1)
    df['TL_x_Deg'] = df['TyreLife'] * df['Cumulative_Degradation']
    df['TL_x_LTD'] = df['TyreLife'] * df['LapTime_Delta']
    df['Pos_x_RP'] = df['Position'] * df['RaceProgress']
    df['PitStop_x_RP'] = df['PitStop'] * df['RaceProgress']
    df['Stint_x_Compound'] = df['Stint'] * co
    df['LN_minus_RP'] = df['LapNumber'] - df['RaceProgress'] * df['LapNumber'].max()

    return df.fillna(0).replace([np.inf, -np.inf], 0)


train_fe = add_features(train)
test_fe = add_features(test)

# Non-target group stats
drv_avg_tyre = train.groupby('Driver')['TyreLife'].mean()
drv_avg_pos = train.groupby('Driver')['Position'].mean()
drv_avg_laptime = train.groupby('Driver')['LapTime (s)'].mean()
drv_avg_degrad = train.groupby('Driver')['Cumulative_Degradation'].mean()
drv_avg_ltd = train.groupby('Driver')['LapTime_Delta'].mean()
race_avg_degrad = train.groupby('Race')['Cumulative_Degradation'].mean()
race_avg_laptime = train.groupby('Race')['LapTime (s)'].mean()
race_avg_tyre = train.groupby('Race')['TyreLife'].mean()
race_avg_pos = train.groupby('Race')['Position'].mean()
race_avg_ltd = train.groupby('Race')['LapTime_Delta'].mean()
compound_avg_degrad = train.groupby('Compound')['Cumulative_Degradation'].mean()
compound_avg_laptime = train.groupby('Compound')['LapTime (s)'].mean()

for df in [train_fe, test_fe]:
    df['Drv_avg_tyre'] = df['Driver'].map(drv_avg_tyre)
    df['Drv_avg_pos'] = df['Driver'].map(drv_avg_pos)
    df['Drv_avg_laptime'] = df['Driver'].map(drv_avg_laptime)
    df['Drv_avg_degrad'] = df['Driver'].map(drv_avg_degrad)
    df['Drv_avg_ltd'] = df['Driver'].map(drv_avg_ltd)
    df['Race_avg_degrad'] = df['Race'].map(race_avg_degrad)
    df['Race_avg_laptime'] = df['Race'].map(race_avg_laptime)
    df['Race_avg_tyre'] = df['Race'].map(race_avg_tyre)
    df['Race_avg_pos'] = df['Race'].map(race_avg_pos)
    df['Race_avg_ltd'] = df['Race'].map(race_avg_ltd)
    df['Compound_avg_degrad'] = df['Compound'].map(compound_avg_degrad)
    df['Compound_avg_laptime'] = df['Compound'].map(compound_avg_laptime)
    df['Dev_drv_tyre'] = df['TyreLife'] - df['Drv_avg_tyre']
    df['Dev_drv_pos'] = df['Position'] - df['Drv_avg_pos']
    df['Dev_drv_degrad'] = df['Cumulative_Degradation'] - df['Drv_avg_degrad']
    df['Dev_drv_ltd'] = df['LapTime_Delta'] - df['Drv_avg_ltd']
    df['Dev_race_degrad'] = df['Cumulative_Degradation'] - df['Race_avg_degrad']
    df['Dev_race_laptime'] = df['LapTime (s)'] - df['Race_avg_laptime']
    df['Dev_race_tyre'] = df['TyreLife'] - df['Race_avg_tyre']
    df['Dev_race_pos'] = df['Position'] - df['Race_avg_pos']
    df['Dev_race_ltd'] = df['LapTime_Delta'] - df['Race_avg_ltd']
    df['Dev_compound_degrad'] = df['Cumulative_Degradation'] - df['Compound_avg_degrad']
    df['Dev_compound_laptime'] = df['LapTime (s)'] - df['Compound_avg_laptime']

# ============================================================
# 3. KFOLD TARGET ENCODING
# ============================================================
log("\n[3/6] KFold target encoding...")

y_all = train_fe['PitNextLap'].values
te_groups = [
    ('Driver', 'Driver_te'),
    ('Race', 'Race_te'),
    ('Compound', 'Compound_te'),
    ('Stint', 'Stint_te'),
    ('Year', 'Year_te'),
    (['Compound', 'Race'], 'Compound_Race_te'),
    (['Driver', 'Race'], 'Driver_Race_te'),
    (['Compound', 'Stint'], 'Compound_Stint_te'),
    (['Driver', 'Compound'], 'Driver_Compound_te'),
    (['Driver', 'Stint'], 'Driver_Stint_te'),
    (['Race', 'Stint'], 'Race_Stint_te'),
    (['Compound', 'Race', 'Stint'], 'CRStint_te'),
]

skf_enc = StratifiedKFold(n_splits=NF, shuffle=True, random_state=42)
for c_name in [te[1] for te in te_groups]:
    train_fe[c_name] = gm

for tr_idx, val_idx in skf_enc.split(train_fe, y_all):
    tr_fold = train_fe.iloc[tr_idx]
    vi = train_fe.index[val_idx]
    for grp, te_name in te_groups:
        if isinstance(grp, str):
            grp_mean = tr_fold.groupby(grp)['PitNextLap'].agg(['mean', 'count'])
            grp_mean['smooth'] = (grp_mean['mean'] * grp_mean['count'] + gm * 100) / (grp_mean['count'] + 100)
            train_fe.loc[vi, te_name] = train_fe.loc[vi, grp].map(grp_mean['smooth']).fillna(gm)
        else:
            grp_mean = tr_fold.groupby(list(grp))['PitNextLap'].agg(['mean', 'count'])
            grp_mean['smooth'] = (grp_mean['mean'] * grp_mean['count'] + gm * 30) / (grp_mean['count'] + 30)
            idx_vals = train_fe.loc[vi].set_index(list(grp)).index
            train_fe.loc[vi, te_name] = idx_vals.map(grp_mean['smooth']).fillna(gm).values

for grp, te_name in te_groups:
    if isinstance(grp, str):
        full_mean = train.groupby(grp)['PitNextLap'].mean()
        test_fe[te_name] = test_fe[grp].map(full_mean).fillna(gm)
    else:
        full_mean = train.groupby(list(grp))['PitNextLap'].mean()
        test_fe[te_name] = test_fe.set_index(list(grp)).index.map(lambda x: full_mean.get(x, gm))

# Risk multipliers
train_fe['Risk_CD'] = train_fe['Compound_te'] * train_fe['Driver_te']
train_fe['Risk_DR'] = train_fe['Driver_te'] * train_fe['Race_te']
train_fe['Risk_CR'] = train_fe['Compound_te'] * train_fe['Race_te']
train_fe['Risk_CDR'] = train_fe['Compound_te'] * train_fe['Driver_te'] * train_fe['Race_te']
train_fe['Risk_CS'] = train_fe['Compound_te'] * train_fe['Stint_te']
train_fe['Risk_DS'] = train_fe['Driver_te'] * train_fe['Stint_te']
train_fe['Risk_RS'] = train_fe['Race_te'] * train_fe['Stint_te']
test_fe['Risk_CD'] = test_fe['Compound_te'] * test_fe['Driver_te']
test_fe['Risk_DR'] = test_fe['Driver_te'] * test_fe['Race_te']
test_fe['Risk_CR'] = test_fe['Compound_te'] * test_fe['Race_te']
test_fe['Risk_CDR'] = test_fe['Compound_te'] * test_fe['Driver_te'] * test_fe['Race_te']
test_fe['Risk_CS'] = test_fe['Compound_te'] * test_fe['Stint_te']
test_fe['Risk_DS'] = test_fe['Driver_te'] * test_fe['Stint_te']
test_fe['Risk_RS'] = test_fe['Race_te'] * test_fe['Stint_te']

# 新增：频率编码特征
for col in ['Driver', 'Race', 'Compound', 'Stint']:
    freq = train[col].value_counts(normalize=True)
    train_fe[f'{col}_freq'] = train_fe[col].map(freq)
    test_fe[f'{col}_freq'] = test_fe[col].map(freq).fillna(0)

# ============================================================
# 4. FEATURE LIST
# ============================================================
FEATURES = [
    # Original features
    'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'Position_Change',
    'Compound_ord', 'Driver_le', 'Race_le',
    # Target encodings
    'Driver_te', 'Race_te', 'Compound_te', 'Stint_te', 'Year_te',
    'Compound_Race_te', 'Driver_Race_te', 'Compound_Stint_te',
    'Driver_Compound_te', 'Driver_Stint_te', 'Race_Stint_te', 'CRStint_te',
    # Risk multipliers
    'Risk_CD', 'Risk_DR', 'Risk_CR', 'Risk_CDR', 'Risk_CS', 'Risk_DS', 'Risk_RS',
    # Compound interactions
    'CxTyreLife', 'CxLapNumber', 'CxRaceProgress', 'CxDegradation', 'CxPosition',
    # TyreLife transforms
    'TL_sq', 'TL_sqrt', 'TL_log', 'TL_cu',
    'TL_gt_15', 'TL_gt_25', 'TL_gt_35', 'TL_gt_50',
    # Degradation features
    'Deg_per_lap', 'LTD_per_lap', 'LT_per_lap', 'Deg_abs', 'Deg_sq',
    # Position
    'Pos_sq', 'Is_Top10',
    # Race/Lap
    'RP_sq', 'RP_cu', 'LN_sq', 'LN_sqrt',
    # Stint
    'Stint_x_TL', 'Stint_x_LN', 'TL_div_Stint', 'LN_div_Stint',
    # Year
    'Year_2022', 'Year_2023', 'Year_2024', 'Year_2025',
    # Other
    'PChange_abs', 'Has_Pitted', 'Tire_Wear', 'LTD_abs', 'LT_sq',
    # Group stats + deviations
    'Drv_avg_tyre', 'Drv_avg_pos', 'Drv_avg_laptime', 'Drv_avg_degrad', 'Drv_avg_ltd',
    'Race_avg_degrad', 'Race_avg_laptime', 'Race_avg_tyre', 'Race_avg_pos', 'Race_avg_ltd',
    'Compound_avg_degrad', 'Compound_avg_laptime',
    'Dev_drv_tyre', 'Dev_drv_pos', 'Dev_drv_degrad', 'Dev_drv_ltd',
    'Dev_race_degrad', 'Dev_race_laptime', 'Dev_race_tyre', 'Dev_race_pos', 'Dev_race_ltd',
    'Dev_compound_degrad', 'Dev_compound_laptime',
    # Key interactions
    'PitStop_x_TL', 'LT_x_Compound', 'RP_x_TL', 'Pos_x_TL', 'LN_x_TL', 'Deg_x_RP', 'TL_div_RP', 'Stint_x_RP',
    # V8 new features
    'Compound_avg_TL', 'Dev_compound_TL', 'TL_ratio_compound',
    'Deg_x_Compound', 'LTD_x_Compound',
    'RP_div_Stint',
    'TL_x_Deg', 'TL_x_LTD',
    'Pos_x_RP', 'PitStop_x_RP',
    # 新增：频率编码特征
    'Driver_freq', 'Race_freq', 'Compound_freq', 'Stint_freq',
    'Stint_x_Compound',
]

train_fe = train_fe.fillna(0).replace([np.inf, -np.inf], 0)
test_fe = test_fe.fillna(0).replace([np.inf, -np.inf], 0)

missing = [f for f in FEATURES if f not in train_fe.columns]
if missing:
    log(f"WARNING: Missing features: {missing}")
    FEATURES = [f for f in FEATURES if f in train_fe.columns and f in test_fe.columns]

X = train_fe[FEATURES].values.astype(np.float32)
y = train_fe['PitNextLap'].values.astype(np.int32)
X_test = test_fe[FEATURES].values.astype(np.float32)

# 新增：自动删除高相关特征（相关系数>0.95）
corr_matrix = pd.DataFrame(X).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [FEATURES[i] for i in upper.columns if any(upper[i] > 0.95)]
FEATURES = [f for f in FEATURES if f not in to_drop]
log(f"Removed {len(to_drop)} highly correlated features")

X = train_fe[FEATURES].values.astype(np.float32)
X_test = test_fe[FEATURES].values.astype(np.float32)
log(f"Features: {len(FEATURES)} | Train: {X.shape} | Test: {X_test.shape}")

# ============================================================
# 5. MULTI-CONFIG TRAINING (GPU加速)
# ============================================================
log("\n[4/6] Multi-config training...")

all_oof = {}
all_test = {}

# --- Config 1: LightGBM GBDT (deep, low lr) ---
cfg_name = 'lgb_deep'
log(f"\n  Training {cfg_name}...")
oof_acc = np.zeros(len(X))
test_acc = np.zeros(len(X_test))
for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X))
    test_seed = np.zeros(len(X_test))
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(
            objective='binary', metric='auc', boosting_type='gbdt',
            n_estimators=6000, learning_rate=0.012, num_leaves=383,
            max_depth=11, min_data_in_leaf=25, feature_fraction=0.35,
            bagging_fraction=0.7, bagging_freq=3, lambda_l1=1.5,
            lambda_l2=2.5, min_gain_to_split=0.008,
            verbose=-1, random_state=SEED, n_jobs=-1,
            device='gpu' if torch.cuda.is_available() else 'cpu'
        )
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], eval_metric='auc',
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof_seed[val] = m.predict_proba(X[val])[:, 1]
        test_seed += m.predict_proba(X_test)[:, 1] / NF
        log(f"    {cfg_name} seed={SEED} fold={fold + 1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
    auc_seed = roc_auc_score(y, oof_seed)
    log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
    oof_acc += oof_seed / len(SEEDS)
    test_acc += test_seed / len(SEEDS)
all_oof[cfg_name] = oof_acc
all_test[cfg_name] = test_acc
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_acc):.6f}")
gc.collect()

# --- Config 2: LightGBM GBDT (medium depth, more regularization) ---
cfg_name = 'lgb_reg'
log(f"\n  Training {cfg_name}...")
oof_acc = np.zeros(len(X))
test_acc = np.zeros(len(X_test))
for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X))
    test_seed = np.zeros(len(X_test))
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(
            objective='binary', metric='auc', boosting_type='gbdt',
            n_estimators=5000, learning_rate=0.02, num_leaves=127,
            max_depth=7, min_data_in_leaf=50, feature_fraction=0.5,
            bagging_fraction=0.8, bagging_freq=5, lambda_l1=2.0,
            lambda_l2=3.0, min_gain_to_split=0.02,
            verbose=-1, random_state=SEED, n_jobs=-1,
            device='gpu' if torch.cuda.is_available() else 'cpu'
        )
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], eval_metric='auc',
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof_seed[val] = m.predict_proba(X[val])[:, 1]
        test_seed += m.predict_proba(X_test)[:, 1] / NF
        log(f"    {cfg_name} seed={SEED} fold={fold + 1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
    auc_seed = roc_auc_score(y, oof_seed)
    log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
    oof_acc += oof_seed / len(SEEDS)
    test_acc += test_seed / len(SEEDS)
all_oof[cfg_name] = oof_acc
all_test[cfg_name] = test_acc
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_acc):.6f}")
gc.collect()

# --- Config 3: XGBoost (deep) ---
cfg_name = 'xgb_deep'
log(f"\n  Training {cfg_name}...")
oof_acc = np.zeros(len(X))
test_acc = np.zeros(len(X_test))
for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X))
    test_seed = np.zeros(len(X_test))
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            n_estimators=6000, learning_rate=0.012, max_depth=9,
            min_child_weight=15, subsample=0.7, colsample_bytree=0.35,
            colsample_bylevel=0.35, reg_alpha=1.5, reg_lambda=2.5,
            gamma=0.08, tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            random_state=SEED, n_jobs=-1, early_stopping_rounds=300, verbosity=0,
        )
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
        oof_seed[val] = m.predict_proba(X[val])[:, 1]
        test_seed += m.predict_proba(X_test)[:, 1] / NF
        log(f"    {cfg_name} seed={SEED} fold={fold + 1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
    auc_seed = roc_auc_score(y, oof_seed)
    log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
    oof_acc += oof_seed / len(SEEDS)
    test_acc += test_seed / len(SEEDS)
all_oof[cfg_name] = oof_acc
all_test[cfg_name] = test_acc
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_acc):.6f}")
gc.collect()

# --- Config 4: XGBoost (shallow, regularized) ---
cfg_name = 'xgb_reg'
log(f"\n  Training {cfg_name}...")
oof_acc = np.zeros(len(X))
test_acc = np.zeros(len(X_test))
for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X))
    test_seed = np.zeros(len(X_test))
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            n_estimators=4000, learning_rate=0.02, max_depth=6,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.5,
            colsample_bylevel=0.5, reg_alpha=2.0, reg_lambda=3.0,
            gamma=0.2, tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            random_state=SEED, n_jobs=-1, early_stopping_rounds=200, verbosity=0,
        )
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
        oof_seed[val] = m.predict_proba(X[val])[:, 1]
        test_seed += m.predict_proba(X_test)[:, 1] / NF
        log(f"    {cfg_name} seed={SEED} fold={fold + 1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
    auc_seed = roc_auc_score(y, oof_seed)
    log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
    oof_acc += oof_seed / len(SEEDS)
    test_acc += test_seed / len(SEEDS)
all_oof[cfg_name] = oof_acc
all_test[cfg_name] = test_acc
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_acc):.6f}")
gc.collect()

# --- Config 5: CatBoost GPU ---
cfg_name = 'cat_gpu'
log(f"\n  Training {cfg_name}...")
oof_acc = np.zeros(len(X))
test_acc = np.zeros(len(X_test))
for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X))
    test_seed = np.zeros(len(X_test))
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = CatBoostClassifier(
            iterations=4000, learning_rate=0.02, depth=8,
            l2_leaf_reg=5, border_count=128, random_strength=0.5,
            bagging_temperature=0.5, od_type='Iter', od_wait=200,
            verbose=0, random_seed=SEED,
            task_type='GPU' if torch.cuda.is_available() else 'CPU',
            use_best_model=True, grow_policy='Lossguide', min_data_in_leaf=30,
        )
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[val], y[val]), verbose=False)
        oof_seed[val] = m.predict_proba(X[val])[:, 1]
        test_seed += m.predict_proba(X_test)[:, 1] / NF
        log(f"    {cfg_name} seed={SEED} fold={fold + 1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
    auc_seed = roc_auc_score(y, oof_seed)
    log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
    oof_acc += oof_seed / len(SEEDS)
    test_acc += test_seed / len(SEEDS)
all_oof[cfg_name] = oof_acc
all_test[cfg_name] = test_acc
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_acc):.6f}")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# --- Config 6: PyTorch Residual MLP (GPU加速) ---
cfg_name = 'nn_residual'
log(f"\n  Training {cfg_name}...")
oof_acc = np.zeros(len(X))
test_acc = np.zeros(len(X_test))
input_dim = X.shape[1]

for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X))
    test_seed = np.zeros(len(X_test))

    for fold, (tr, val) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]

        model, scaler, val_preds, fold_auc = train_nn_fold(
            X_tr, y_tr, X_val, y_val, input_dim, device, seed=SEED + fold
        )

        oof_seed[val] = val_preds
        test_seed += predict_nn(model, scaler, X_test, device) / NF

        log(f"    {cfg_name} seed={SEED} fold={fold + 1}: {fold_auc:.6f}")

        del model, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    auc_seed = roc_auc_score(y, oof_seed)
    log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
    oof_acc += oof_seed / len(SEEDS)
    test_acc += test_seed / len(SEEDS)

all_oof[cfg_name] = oof_acc
all_test[cfg_name] = test_acc
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_acc):.6f}")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
# 6. ENSEMBLE + STACKING
# ============================================================
log("\n[5/6] Ensemble optimization...")

cfg_names = list(all_oof.keys())
log(f"Model configs: {cfg_names}")

oof_stack = np.column_stack([all_oof[c] for c in cfg_names])
test_stack = np.column_stack([all_test[c] for c in cfg_names])

log(f"Stack shape: OOF={oof_stack.shape}, Test={test_stack.shape}")

# Correlation analysis
log("\nModel OOF correlations:")
for i, c1 in enumerate(cfg_names):
    for j, c2 in enumerate(cfg_names):
        if j > i:
            corr = np.corrcoef(all_oof[c1], all_oof[c2])[0, 1]
            log(f"  {c1} vs {c2}: {corr:.4f}")

# Simple average
simple_avg_oof = np.mean(oof_stack, axis=1)
log(f"\nSimple average AUC: {roc_auc_score(y, simple_avg_oof):.6f}")

# L-BFGS-B权重优化（稳定不卡住）
from scipy.optimize import minimize

n_models = len(cfg_names)


def neg_auc(w):
    w = np.clip(w, 0, 1)
    s = w.sum()
    if s < 1e-8:
        return 1.0
    w = w / s
    return -roc_auc_score(y, oof_stack @ w)


best_auc = 0
best_weights = np.ones(n_models) / n_models
np.random.seed(42)

# 只进行20次试验，足够找到最优解
for trial in range(20):
    w0 = np.random.dirichlet(np.ones(n_models))
    res = minimize(neg_auc, w0, method='L-BFGS-B',
                   bounds=[(0, 1)] * n_models,
                   options={'maxiter': 500, 'ftol': 1e-5})
    w_opt = np.clip(res.x, 0, 1)
    w_opt = w_opt / w_opt.sum()
    a = roc_auc_score(y, oof_stack @ w_opt)
    if a > best_auc:
        best_auc = a
        best_weights = w_opt.copy()

log(f"Best optimized AUC: {best_auc:.6f}")
log(f"Best weights: {dict(zip(cfg_names, [f'{w:.4f}' for w in best_weights]))}")

test_ens_opt = test_stack @ best_weights

# 多模型堆叠（替换原来的Ridge堆叠）
from sklearn.linear_model import Ridge, LogisticRegression

log("\nMulti-model stacking...")
skf_stack = StratifiedKFold(n_splits=NF, shuffle=True, random_state=42)

# 训练多个第二层模型
models = [
    ('ridge_100', Ridge(alpha=100)),
    ('ridge_500', Ridge(alpha=500)),
    ('logistic', LogisticRegression(C=0.1, max_iter=1000)),
]

best_stack_auc = 0
best_stack_test = None
best_stack_oof = None

for name, model in models:
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr, val) in enumerate(skf_stack.split(oof_stack, y)):
        model.fit(oof_stack[tr], y[tr])
        oof_preds[val] = model.predict(oof_stack[val])
        test_preds += model.predict(test_stack) / NF

    auc = roc_auc_score(y, oof_preds)
    log(f"  {name}: AUC={auc:.6f}")

    if auc > best_stack_auc:
        best_stack_auc = auc
        best_stack_test = test_preds
        best_stack_oof = oof_preds

log(f"Best stacking AUC: {best_stack_auc:.6f}")

# ============================================================
# 7. FINAL SUBMISSION
# ============================================================
log("\n[6/6] Final submission...")

# 最终融合：权重优化 + 最优堆叠
final_test = (test_ens_opt + best_stack_test) / 2
final_auc = roc_auc_score(y, (oof_stack @ best_weights + best_stack_oof) / 2)
method = "Weighted average + Stacking ensemble"

# Clip predictions
final_test = np.clip(final_test, 0.001, 0.999)

log(f"\n{'=' * 70}")
log(f"FINAL RESULTS (V8.1 GPU版)")
log(f"{'=' * 70}")
for c in cfg_names:
    log(f"  {c}: {roc_auc_score(y, all_oof[c]):.6f}")
log(f"  Optimized weights AUC: {best_auc:.6f}")
log(f"  Best Stacking AUC: {best_stack_auc:.6f}")
log(f"  Final method: {method}")
log(f"  Final OOF AUC: {final_auc:.6f}")
log(f"  Prediction mean: {final_test.mean():.4f}")
log(f"  Elapsed: {(time.time() - t0):.0f}s")

sub = pd.read_csv('playground-series-s6e5/sample_submission.csv')
sub['PitNextLap'] = final_test
sub.to_csv('submission_v8_gpu.csv', index=False)
log(f"\nSaved: submission_v8_gpu.csv")

# Save OOF and test predictions
for c in cfg_names:
    np.save(f'oof_{c}_v8_gpu.npy', all_oof[c])
    np.save(f'test_{c}_v8_gpu.npy', all_test[c])

results = {
    'models': {c: float(roc_auc_score(y, all_oof[c])) for c in cfg_names},
    'optimized_auc': float(best_auc),
    'ridge_auc': float(best_stack_auc),
    'final_auc': float(final_auc),
    'method': method,
    'n_features': len(FEATURES),
    'elapsed_s': time.time() - t0,
}
with open('experiment_v8_gpu.json', 'w') as f:
    json.dump(results, f, indent=2)
log("Saved: experiment_v8_gpu.json")