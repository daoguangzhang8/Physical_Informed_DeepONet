# 当前训练与泛化问题讨论整理

本文档整理近期围绕 PI-DeepONet 项目的讨论，重点包括并行训练修正、多频多震源散射波场预测能力、训练集重构、高频能力判断以及 scheduler 测试设计。

## 1. 当前项目训练配置概况

当前主配置来自 `config.py`：

```python
load_path = '/home/sharedata/zdg'
vel_filename = 'multifreq_merged1/freesurface_full_5sources_velocity.npy'
backgroundfield_filename = 'multifreq_merged1/freesurface_full_5sources_background.npy'
wavefield_filename = 'multifreq_merged1/freesurface_full_5sources_wavefield.npy'
freq_filename = 'multifreq_merged1/freesurface_full_5sources_freq_used.npy'

nvel_train = 3600
source_list = [0, 1, 2, 3, 4]
sampling_mode = 'halton'
halton_sample_ratio = 0.5
batch_size = 1600
batch_size_v = 40
accumulation_steps = 2

NIter = 2001
lr = 2e-4
scheduler_type = 'cosine'
use_warmup = True
warmup_epochs = 100
cosine_T_0 = 2001
cosine_eta_min = 1e-5

branch2_type = 'conv'
```

已确认主训练数据形状：

```text
velocity:    (6000, 160, 180)
background:  (30000, 2, 160, 180)
wavefield:   (30000, 2, 160, 180)
freq_used:   (6000,)
```

频率覆盖 15 个离散频率：

```text
3, 5, 7, 9, 11,
12, 13, 15, 17, 18,
19, 21, 22, 23, 25 Hz
```

当前数据可理解为：

```text
约 2000 个基础速度模型 × 3 个频段阶段 = 6000 个 velocity-frequency 条目
每个条目包含 5 个震源
总背景场/波场样本 = 6000 × 5 = 30000
```

## 2. 并行训练问题与修正方向

此前多卡训练表现为不仅弱于单卡，而是存在明显不收敛风险。代码检查后确认，原 DDP 路径与单卡训练不是同一个优化过程。

主要问题：

1. DDP 原先每个 velocity batch 只更新一次，而单卡按 `accumulation_steps` 在坐标 batch 内多次更新。
2. DDP 原先默认 `lr * world_size`，多卡学习率被放大，对 PDE/PINN 训练可能过激。
3. DDP 原先固定使用 `ReduceLROnPlateau`，而单卡读取 `scheduler_type='cosine'`。
4. DDP 原先 `no_sync()` 只包住 backward，没有包住 forward，不符合 PyTorch DDP 预期。
5. DDP 原先即使 `use_fno_as_label=False` 也创建 FNO，额外消耗随机数，影响后续 shuffle。

已执行的 DDP 对齐修改：

```python
ddp_scale_lr = False
ddp_split_batch_size_v = True
```

并行路径现在默认：

```text
DDP 学习率与单卡一致
每卡 batch_size_v = batch_size_v / world_size，使全局 batch 接近单卡
DDP scheduler 与单卡一致读取 scheduler_type
DDP loss 缩放为 loss / accumulation_steps
DDP optimizer.step 按 accumulation_steps 执行
no_sync 包住 forward + backward
不使用 FNO 软标签时不创建 FNO
```

静态检查已通过：

```bash
python -m py_compile config.py main2.py model/train_distributed.py model/utils.py model/PI_DeepOnet.py
```

## 3. 当前并行策略

当前项目使用的是数据并行 DDP，不是模型并行。

```text
每张 GPU 上都有一份完整 Pi_DeepONet
训练集 velocity/source 样本由 DistributedSampler 分片
每张 GPU 独立前向/反向
DDP all-reduce 平均梯度
rank 0 负责保存、画图、验证
```

注意：`train_y` 坐标 batch 目前没有按 GPU 分片，各 rank 会遍历同一批坐标点。

预估加速：

```text
两卡：约 1.3x - 1.7x
八卡：对齐单卡模式下约 2.5x - 4.5x
八卡吞吐优先模式下约 4.5x - 6.5x
```

主要瓶颈：

```text
每个 rank 重复读取完整 .npy
train_y 没有分片
PDE loss 与二阶梯度计算重
DDP all-reduce 同步成本
```

## 4. Branch2 CNN 与多频多震源能力判断

当前网络目标是：

```text
(velocity, background wavefield UU0, coordinate y) -> scattered wavefield UU - UU0
```

Branch2 输入：

```python
UU0: [B, 2, NZ, NX]
```

其中两个通道是背景波场的实部和虚部。`UU0` 本身由频率、震源位置、背景速度、边界条件共同决定，因此理论上携带：

```text
频率信息：波长、相位振荡密度、实/虚部空间周期
震源信息：波前中心、传播方向、近源强振幅区、相位中心
```

用户已同步测试结果：

```text
FNO 作为 branch2 时，对 unseen 10Hz / 20Hz 几乎没有有效波形输出
CNN branch2 对 unseen freq 表现合格
ResNet 与 CNN 在单一速度模型测试中效果接近
因此当前选择更省显存的 CNN 是合理工程取舍
```

更新后的判断：

```text
FNO branch2 不适合当前任务中的频率外推
CNN branch2 的局部卷积归纳偏置更适合从 UU0 提取频率/震源条件
当前瓶颈不主要在 branch2 是否能识别 freq/source
更可能在 unseen velocity 与 unseen frequency 的组合泛化
```

## 5. 当前测试结果解读

用户同步了 CNN 网络在 unseen velocity + unseen frequency 上的测试结果。

结论：

```text
网络不是完全失效
已经具备一定波形预测能力
但实际预测精度未达到要求
```

图像表现：

```text
低/中频波前结构大体可见
高频存在相位偏移、局部振幅不准、波纹破碎
20Hz 附近表现尤其不稳定
```

这说明：

```text
CNN branch2 已经能从 UU0 提取部分频率/震源信息
但 Branch1 velocity 表征与 Branch1/Branch2/Trunk 融合对组合泛化仍不足
```

阶段性判断：

```text
多频条件识别能力：有
多震源条件识别能力：大概率有
散射场基础预测能力：有
unseen vel + unseen freq 组合泛化：不足
```

## 6. 参数量与训练集配置判断

当前网络参数量约 2500 万。

结合当前数据规模：

```text
训练样本：3600 个 velocity-frequency 条目 × 5 个震源 = 18000 个样本
每个样本有约 145 × 150 个空间监督点，实部/虚部
还有 PDE loss
```

判断：

```text
2500 万参数不是明显过大
当前问题不应优先通过盲目增大参数解决
训练集组织、速度模型复杂度、组合泛化评估更关键
```

建议不要优先加大 Branch2。若要调整参数分配，更建议关注：

```text
Branch1 velocity encoder
Branch1/Branch2 fusion
Trunk 与融合处的表达能力
```

## 7. 当前训练状态与参数判断

用户同步了以下图像：

1. 测试集 11Hz，epoch 1600，预测效果较好。
2. 训练集 25Hz，epoch 1600，仍有明显噪声与相位问题。
3. Data Loss 曲线持续下降，Valid Data Loss 明显低于 Training Data Loss。

判断：

```text
训练没有发散
低频/中低频预测能力已经形成
高频 25Hz 即使在训练集上也没有拟合好
当前更像稳定但高频欠拟合
```

Valid loss 低于 train loss 的重要原因可能是当前验证集构造偏乐观：

```python
remaining_idx = [i for i in range(len(vel)) if i not in selected_idx_set]
valid = remaining_idx[:valid_num]
```

由于合并数据按频段排列，`remaining_idx[:valid_num]` 可能偏向低频或更简单样本，因此 valid loss 不能可靠代表高频泛化。

当前参数判断：

```text
lr = 2e-4：稳定，合理
warmup_epochs = 100：合理
batch_size_v = 40：稳定，合理
branch2 = cnn：合理
NIter = 2001：偏短
cosine_T_0 = 2001：可能后期学习率降得偏早
halton_sample_ratio = 0.5：对 25Hz 可能偏低
valid 划分方式：需要修正
```

建议短期训练参数：

```python
NIter = 4001
cosine_T_0 = 4001
lr = 2e-4
use_warmup = True
warmup_epochs = 100
halton_sample_ratio = 0.75
batch_size = 1600
batch_size_v = 40
accumulation_steps = 2
a = 1
b = 1
```

若 `halton_sample_ratio=0.75` 成本过高，可退到 0.6。

## 8. Scheduler 选择讨论

当前不建议使用 `cosine_T_0 = NIter / 3` 作为第一选择。

原因：

```text
当前 loss 仍持续下降
高频训练集仍未拟合
问题不是平台期，而是仍需要持续有效学习率
频繁 warm restart 可能扰动高频相位
```

推荐顺序：

```text
主训练：cosine_T_0 = NIter，单周期 cosine
后期若平台：小学习率 Plateau fine-tune
暂不优先使用 NIter/3 restart
```

推荐主训练配置：

```python
NIter = 4001
scheduler_type = 'cosine'
cosine_T_0 = 4001
cosine_T_mult = 1
cosine_eta_min = 1e-5
use_warmup = True
warmup_epochs = 100
```

可选后期 fine-tune：

```python
NIter = 1001
scheduler_type = 'plateau'
lr = 5e-5
factor = 0.7
patience = 50
min_lr = 1e-6
```

## 9. Scheduler 测试设计

为了确认最合适的 scheduler，建议做 4 组主实验 + 1 个可选实验。

统一基础设置：

```python
NIter = 4001
lr = 2e-4
use_warmup = True
warmup_epochs = 100
cosine_eta_min = 1e-5

batch_size = 1600
batch_size_v = 40
accumulation_steps = 2
halton_sample_ratio = 0.5

save_fig_every = 200
validate_every = 200
save_model_every = 1000
```

固定评估样本：

```text
训练集 25Hz
验证集 11Hz
Marmousi unseen vel + 10Hz
Marmousi unseen vel + 20Hz
```

实验 1：单周期 Cosine

```python
scheduler_type = 'cosine'
cosine_T_0 = NIter
cosine_T_mult = 1
cosine_eta_min = 1e-5
```

实验 2：短周期 Cosine Restart

```python
scheduler_type = 'cosine'
cosine_T_0 = NIter // 3
cosine_T_mult = 1
cosine_eta_min = 1e-5
```

实验 3：Plateau

```python
scheduler_type = 'plateau'
lr = 2e-4
factor = 0.7
patience = 80
min_lr = 1e-6
```

实验 4：Constant + 手动降学习率

```text
0-3000 epoch: lr = 2e-4
3000-4000 epoch: lr = 5e-5
```

可选实验 5：Cosine 主训 + Plateau fine-tune

```text
stage1: cosine, lr = 2e-4, 3000 epoch
stage2: plateau, lr = 5e-5, 1000 epoch
```

评价指标：

```text
train_25Hz_MSE
valid_11Hz_MSE
marmousi_10Hz_MSE
marmousi_20Hz_MSE
预测图是否出现高频碎裂
学习率曲线
```

若只能先跑两个实验，优先：

```text
1. cosine_T_0 = NIter
2. constant 3000 + lr drop 1000
```

## 10. 训练集重构方案

训练集重构是必要方向。

核心原则：

```text
不要再按 velocity-frequency 条目随机划分
要按物理 velocity model 作为最小单位划分
```

目标：

```text
避免同一物理速度模型的不同频率版本同时出现在 train / valid / test
真实评估 unseen velocity + unseen frequency
```

建议显式索引：

```text
base_velocity_id
freq_id
source_id
```

逻辑形态：

```text
velocity[base_vel_id, freq_id, z, x]
background[base_vel_id, freq_id, source_id, 2, z, x]
wavefield[base_vel_id, freq_id, source_id, 2, z, x]
freq_used[base_vel_id, freq_id]
```

推荐划分：

```text
train velocity IDs: 70%
valid velocity IDs: 15%
test velocity IDs: 15%
```

例如 2000 个基础速度模型：

```text
train: 1400
valid: 300
test: 300
```

每个 base velocity 的所有频率、所有震源只属于一个集合。

推荐新数据目录：

```text
/home/sharedata/zdg/multifreq_grouped
```

结构：

```text
multifreq_grouped/
├── train/
│   ├── velocity.npy
│   ├── background.npy
│   ├── wavefield.npy
│   ├── freq_used.npy
│   └── meta.npz
├── valid/
│   ├── velocity.npy
│   ├── background.npy
│   ├── wavefield.npy
│   ├── freq_used.npy
│   └── meta.npz
└── test/
    ├── velocity.npy
    ├── background.npy
    ├── wavefield.npy
    ├── freq_used.npy
    └── meta.npz
```

`meta.npz` 至少保存：

```text
base_velocity_ids
freq_values
source_ids
original_velocity_indices
```

建议第一版保持与当前 dataloader 尽量兼容的展平格式，减少代码改动。

## 11. 当前暂缓方向

根据用户反馈，以下方向暂缓：

```text
stage training
wavenumber PE / 高频位置编码扩展
envelope loss
phase-aware loss
多尺度 loss
更大 ResNet/FNO branch2
```

原因：

```text
stage training 判断周期过长，需要先完成前两个 stage，时间成本不接受
高频扩展模块已测试，提升不足且明显拖慢运行速度
FNO branch2 对 freq 外推表现差
ResNet 与 CNN 差异不明显，CNN 更省显存
```

## 12. 当前优先级建议

短期优先：

```text
1. 修正 valid 划分方式，避免低频/简单样本偏置
2. 设计并执行 scheduler 对比实验
3. 延长训练到 4001 epoch，使用单周期 cosine
4. 尝试 halton_sample_ratio = 0.6 / 0.75
5. 构建 velocity-group split 数据集
```

中期优先：

```text
1. 用 grouped 数据重新评估 unseen velocity + unseen frequency
2. 增加复杂速度模型或按速度梯度复杂度重采样
3. 固定 CNN branch2，优先检查 Branch1 与 fusion 是否是组合泛化瓶颈
```

当前最关键的问题不是“模型完全不会”，而是：

```text
模型已经具备多频多震源散射波场预测能力的雏形，
但 unseen velocity + unseen frequency 的组合泛化仍不足。
```

