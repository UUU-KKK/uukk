# F1 进站预测 — Kaggle Playground S6E5

Kaggle Playground Series S6E5 的二分类任务,拿 F1 圈级数据预测车手下一圈是否进站,评测指标 ROC-AUC,目标 LB > 0.954。主文件 `v-8.1.py`,单文件闭环:特征工程 → 6 模型交叉验证 → 权重优化 + 二层堆叠 → 输出提交。LightGBM / XGBoost / CatBoost / PyTorch 残差网络一起上,GPU/CPU 自适应。

## 怎么跑

```bash   ”“bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost torch scipy

# 数据放这里
# playground-series-s6e5/
# ├── train.csv
# ├── test.csv
# └── sample_submission.csv

python v-8.1.py
```

跑完生成 `submission_v8_gpu.csv`,直接传 Kaggle。无 GPU 自动降级 CPU。

环境:Python 3.8+,CUDA 11.x+,显存 6 GB+。LightGBM 4.0+ 的 GPU 参数从 `device='gpu'` 改成了 `device='cuda'`,代码已兼容。CPU 跑 5 折 × 5 种子 × 6 模型较慢。

## 数据

Kaggle Playground Series S6E5,圈级数据,目标列 `PitNextLap`(0/1),正负比约 1:5,全局先验 `gm = 0.1990`。数据放 `playground-series-s6e5/` 下,与代码同级。

## 项目结构

```
.
├── v-8.1.py                      # 主程序
├── playground-series-s6e5/       # 数据
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── submission_v8_gpu.csv         # 提交文件
├── experiment_v8_gpu.json        # 实验记录
├── oof_*.npy                     # 各模型 OOF 预测(6 个)
└── test_*.npy                    # 各模型测试集预测(6 个)
```

## 代码组织

单文件线性 pipeline,7 个模块单向依赖:

1. 依赖与工具(文件开头 ~ `log`)
2. 深度学习组件(`F1Dataset` ~ `predict_nn`)
3. 全局配置(全局常量 ~ GPU 检测)
4. 数据与特征(数据加载 ~ 频率编码)
5. 特征筛选(`FEATURES` ~ 矩阵输出)
6. 多模型训练(5 折 × 5 种子 CV)
7. 集成输出(融合、堆叠、提交)

执行阶段日志:`[1/6] Loading data` → `[2/6] Feature engineering` → `[3/6] KFold target encoding` → `[4/6] Multi-config training` → `[5/6] Ensemble optimization` → `[6/6] Final submission`。

## 特征工程

`add_features` 把原始十来维扩到 100+ 维。

**轮胎配方交互**:Compound 有序编码 × TyreLife / LapNumber / RaceProgress / Degradation / Position。

**轮胎寿命非线性变换**:`TL_sq`、`TL_cu`、`TL_sqrt`、`TL_log`、`TL_gt_15`、`TL_gt_25`、`TL_gt_35`、`TL_gt_50`。

**退化衍生**:`Deg_per_lap`、`LTD_per_lap`、`Deg_abs`、`Deg_sq`。

**位置与排名**:`Pos_sq`、`Is_Top10`。

**比赛进度与停站交叉**:`Stint_x_TL`、`PitStop_x_TL`、`Stint_x_RP`、`TL_div_RP`。

**时序滑动窗口**:按 (Race, Driver) 分组,前 3 圈 rolling mean + shift(1),只用历史圈防泄露。窗口试过 5 圈会稀释信号,3 圈最优。

**轮胎配方基准偏差**:配方典型寿命 SOFT 12 / MEDIUM 18 / HARD 25 / INTERMEDIATE 10 / WET 8,生成 `Dev_compound_TL`、`TL_ratio_compound`。

**分组统计与偏差**:Driver / Race / Compound 三维分组均值 + 当前值偏差。

**5 折交叉目标编码**:12 组(Driver、Race、Compound、Stint、Year 单特征 5 组;双特征 6 组;三特征 1 组)。折内统计,测试集用全量映射。带先验平滑:

```
smooth = (mean × count + gm × α) / (count + α)平滑度=（均值&次数；统计gm &次数；α） /（统计&alpha；）
```

单特征 α=100,组合特征 α=30。

**风险乘积特征**:`Risk_CD`、`Risk_CR`、`Risk_CDR`(目标编码两两、三三相乘)。

**频率编码**:Driver / Race / Compound / Stint 的训练集频率。

## 模型

6 个基模型,算法异构 + 配置差异。统一走 5 种子 × 5 折 StratifiedKFold,种子 `[42, 123, 2024, 456, 789]`,输出 OOF + 测试预测,每模型后 `gc.collect()`,GPU 模型额外 `torch.cuda.empty_cache()`。

**lgb_deep** — LightGBM,6000 树,lr 0.012,383 叶,深度 11,min_data_in_leaf 25,feature_fraction 0.35,bagging 0.7 每 3 树,L1/L2 = 1.5/2.5,min_gain 0.008,早停 200。主力大容量。

**lgb_reg** — LightGBM,5000 树,lr 0.02,127 叶,深度 7,min_data_in_leaf 50,feature_fraction 0.5,bagging 0.8 每 5 树,L1/L2 = 2.0/3.0,min_gain 0.02,早停 200。高正则泛化互补。

**xgb_deep** — XGBoost,6000 树,lr 0.012,深度 9,min_child_weight 15,subsample 0.7,colsample_bytree/bylevel 0.35/0.35,L1/L2 = 1.5/2.5,gamma 0.08,早停 300。算法差异化。

**xgb_reg** — XGBoost,4000 树,lr 0.02,深度 6,min_child_weight 50,subsample 0.8,colsample_bytree/bylevel 0.5/0.5,L1/L2 = 2.0/3.0,gamma 0.2,早停 200。轻量泛化。

**cat_gpu** — CatBoost,Ordered Boosting,Lossguide,4000 迭代,lr 0.02,深度 8,min_data_in_leaf 30,L2 = 5,border_count 128,random_strength 0.5,bagging_temperature 0.5,早停 200 留最优。类别特征原生支持,分裂算法差异大。

**nn_residual** — PyTorch 残差 MLP,input → 512(Linear+BN+ReLU+Dropout 0.2)→ 2 残差块(每块 2 层 FC+BN+ReLU+Dropout 0.3)→ 下采样 256 → 2 残差块 → 128 → 1(Sigmoid)。AdamW,lr 0.0015,weight_decay 1e-5,ReduceLROnPlateau(patience 5,factor 0.5,min_lr 1e-6),BCELoss,batch 1024/2048,100 epoch,早停 15 轮,折内 StandardScaler。单模比树模型低约 0.005,但错误模式差异最大,集成仍有增益。

## 集成策略

三级结构:线性权重优化 + 二层堆叠 + 最终融合。

先算 6 模型 OOF 两两皮尔逊相关,相关性越低融合增益越高。

**第一层 L-BFGS-B 权重优化**:目标最小化负验证 AUC,约束权重非负且和为 1,20 次狄利克雷采样初始值迭代避免局部最优。

**第二层 5 折交叉堆叠**:6 模型 OOF 作 6 列新特征,三个元模型并行——Ridge α=100、Ridge α=500、LogisticRegression C=0.1,折内训练,取验证 AUC 最高的输出。试过 LightGBM 元模型,线上线下 gap 变大,退回 Ridge。

**最终融合**:权重优化结果与最优堆叠结果等权平均,平滑波动。

## 输出

- `submission_v8_gpu.csv` — 提交文件
- `experiment_v8_gpu.json` — `models`(单模 AUC)、`optimized_auc`、`ridge_auc`、`final_auc`、`method`、`n_features`、`elapsed_s`
- `oof_*.npy` — 6 模型 OOF
- `test_*.npy` — 6 模型测试预测

预测裁剪到 `[0.001, 0.999]`,控制台打印单模 AUC / 集成 AUC / 最终方法 / 总耗时。

## 模型效果

- lgb_deep: —   - lgb_deep: &mdash；
- lgb_reg: —   - lgb_reg: &mdash；
- xgb_deep: —   - xgb_deep: —
- xgb_reg: —   - xgb_reg: —
- cat_gpu: —   - cat_gpu: &mdash；
- nn_residual: —
- 权重优化集成: —
- 堆叠集成: —
- 最终融合: —
- Kaggle 线上: —(目标 LB > 0.954)

## 常见坑

**`FileNotFoundError: 'train.csv'`** — 数据放 `playground-series-s6e5/` 或改代码路径。

**`ValueError: Expected 2D array, got 1D`** — 堆叠循环变量名与全局 `oof_stack` 重名,二维被一维覆盖。

**GPU 没识别** — 装 CUDA 版 PyTorch,确认 `torch.cuda.is_available()`。LightGBM 4.0+ 用 `device='cuda'`。

**线下高线上低** — 查目标编码、时序特征泄露,验证折划分,加正则。

**模型相关性高** — 加差异化模型,拉开参数差异,或新特征维度。

**NN 弱于树模型** — 表格数据正常,NN 价值在差异化预测。

## 调参方向

**全局**:`SEEDS` 5 个,金牌区建议 5–10;`NF` 5,10 折更准但耗时翻倍。

**树模型**:榨精度加迭代降学习率,同步加早停;压过拟合加 L1/L2、`min_child_weight`/`min_data_in_leaf`、降特征采样;压方差加 bagging 频率。

**神经网络**:lr 0.001–0.003 最敏感;Dropout 0.2–0.4;残差块加减看显存。

**集成**:元模型试 LR / Ridge 不同 α;权重优化初始值迭代可加但收益递减;拉多样性可加 sklearn GradientBoosting(慢)。

## License

仓库内容仅供学习交流,遵循 Kaggle Playground Series 赛事规则。
