# PI-DeepONet 项目完整上下文

> 用途：作为与 Gemini 讨论数学问题的完整上下文 prompt

---

## 1. 项目目标

用 Physics-Informed DeepONet (PI-DeepONet) 求解带 PML 吸收边界条件的频域声波 Helmholtz 方程（散射场形式）。输入为速度场 v(z,x) 和背景场 UU0(z,x)，输出为散射波场 ΔU(z,x)（复数，实部+虚部）。

---

## 2. 物理/数学方程

### 2.1 Helmholtz 方程（散射场形式）

总波场 U = U₀ + ΔU，其中 U₀ 是均匀背景场（参考速度 v₀ = 1500 m/s），ΔU 是散射场：

$$\nabla^2 \Delta U + \frac{\omega^2}{v^2} \Delta U = -\omega^2\left(\frac{1}{v^2} - \frac{1}{v_0^2}\right) U_0$$

其中 ω = 2πf·10⁻³ 是角频率（缩放后），v = v(z,x) 是非均匀速度场。

### 2.2 PML 完美匹配层吸收边界

PML 引入衰减因子，将方程修改为带阻尼的形式。定义衰减函数：

- x 方向: $l_x = \text{ReLU}\left(\frac{(L_d - 0.5) \cdot dh - x}{(L_d - 0.5) \cdot dh}\right) + \text{ReLU}\left(\frac{x - x_{boundary}}{(L_d - 0.5) \cdot dh}\right)$
- z 方向 (自由表面模式): $l_z = \text{ReLU}\left(\frac{z - z_{boundary}}{(L_d - 0.5) \cdot dh}\right)$

其中 L_d = pml_active = 5（网格数），dh = 20m 是网格间距。

PML 辅助变量：
```
pml_tmp1 = C² · l_x² · l_z²
pml_tmp2 = C² · l_x⁴
pml_tmp3 = C² · l_z⁴
pml_tmp4 = C · (l_z² - l_x²)
pml_tmp5 = C · (l_x² + l_z²)
```
其中 C = a₀ · f₀/f 是频率相关的衰减系数，a₀ = 1.79, f₀ = 10。

### 2.3 修正的一阶导数（PML 吸收）

原始一阶导数 ∂ΔU/∂z, ∂ΔU/∂x 经过 PML 修正：

```
eu_zr = (1 + pml1)/(1 + pml3) · ΔU_z_r - pml4/(1 + pml3) · ΔU_z_i
eu_xr = (1 + pml1)/(1 + pml2) · ΔU_x_r + pml4/(1 + pml2) · ΔU_x_i
eu_zi = pml4/(1 + pml3) · ΔU_z_r + (1 + pml1)/(1 + pml3) · ΔU_z_i
eu_xi = -pml4/(1 + pml2) · ΔU_x_r + (1 + pml1)/(1 + pml2) · ΔU_x_i
```

### 2.4 PDE 残差

实部残差：
```
R_r = ∂²ΔU_r/∂z² + ∂²ΔU_r/∂x²
    + (1-pml1)·ω²·(k_r·(ΔU_r+U0_r) - k_i·(ΔU_i+U0_i))
    + pml5·ω²·(k_r·(ΔU_i+U0_i) + k_i·(ΔU_r+U0_r))
    + (1-pml1)·ω²·(-k0_r·U0_r + k0_i·U0_i)
    + pml5·ω²·(-k0_r·U0_i - k0_i·U0_r)
```

虚部残差 R_i 类似结构（符号变化）。PDE Loss = mean(R_r² + R_i²)。

### 2.5 品质因子 Q 的引入

波数修正（考虑衰减）：
```
α = 1/Q, Q = 15
ρ = (1 - α/π · ln(f/50) - j·α/2)²
k_r = k · Re(ρ), k_i = k · Im(ρ)
k = 1/v²
```

---

## 3. 网络架构

### 3.1 整体结构：PI-DeepONet

```
输入:
  vel [B_v, 1, Z, X]     — 速度场
  UU0 [B_v, 2, Z, X]     — 背景波场（实部+虚部）
  y [B_v, N, 2]           — 查询坐标点 (z, x)

Branch (两个 FNO2d):
  B1 = Branch1(vel)  → FNO(1ch → 256ch) → ChannelAttention → GaussianWeightedLayer → [B_v, N, 256]
  B2 = Branch2(UU0)  → FNO(2ch → 256ch) → ChannelAttention → GaussianWeightedLayer → [B_v, N, 256]
  B = AttenGate(B1, B2)  — 可学习加权融合
  B_encoded = SmoothBlockEncoder(B1_raw + B2_raw, y_normalized) → [B_v, N, 256]

Trunk (FiLM Trunk):
  y → PositionalEncoding → [B_v, N, 16]  (4维 sin/cos × 2方向)
  → FiLMTrunk(y_encoded, condition=B_encoded) → [B_v, N, 256]

输出:
  output = final_layer(B · T_raw)  → [B_v, N, 2]  (实部, 虚部)
```

### 3.2 FNO2d (Fourier Neural Operator)

4 层 Fourier 卷积，每层：
```
x → SpectralConv2d(低频截断) + Conv2d(局部) → GeLU → 残差连接
```
SpectralConv2d: 将输入 FFT → 截取前 modes1×modes2 个低频系数 → 复数乘法可训练权重 → iFFT 回空域。modes1=modes2=16, width=32。

### 3.3 FiLM Trunk（核心创新点）

3 层 FiLM 调制，每层结构：
```
Linear → Sin 激活 → FiLM(γ, β 调制)
```

FiLM 层的具体实现：
```python
class FiLMLayer:
    def __init__(self, cond_dim, feat_dim):
        self.film_layer = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, x, cond):
        gamma_beta = self.film_layer(cond)       # [B, N, feat_dim*2]
        gamma, beta = gamma_beta.chunk(2, dim=-1) # 各 [B, N, feat_dim]
        return (1 + gamma) * x + beta
```

完整 Trunk：
```
输入: y_trunk [B, pts, 16], branch_cond [B, pts, 256]

fc1(16→256) → Sin → FiLM₁(cond=branch_cond)
fc2(256→256) → Sin → FiLM₂(cond=branch_cond)
fc3(256→256) → Sin → FiLM₃(cond=branch_cond)
fc4(256→256) → 输出 [B, pts, 256]
```

**关键点**：γ 和 β 由 Branch 网络提取的速度场特征（B_encoded）生成，使得 Trunk 的基函数成为速度场调制的坐标基函数 t(y; v, UU0)，而非纯坐标基函数 t(y)。

### 3.4 GaussianWeightedLayer

在 Branch 特征提取后，对每个查询坐标 y，以高斯权重从 2D 特征图上提取局部特征：
```
weight(z,x) = exp(-((z-z_y)² + (x-x_y)²) / (2σ²))
B_feat = Σ weight(z,x) · B_raw(z,x)
```

### 3.5 SmoothBlockEncoder

将 Branch 的 2D 特征图通过 grid_sample 映射到查询坐标点，生成空间平滑的特征编码，作为 FiLM 的条件输入。

---

## 4. 损失函数

```
L_total = a · L_data + b · L_PDE
```

- **L_data**: MSE(预测值, 标签) — 数据拟合损失
  - 标签通过将 labels [B, 2, Z, X] 在坐标 y 处采样得到
  - `pred_y = ΔU[:, :n_y, :]` (前 n_y 个点是数据点)

- **L_PDE**: mean(R_r² + R_i²) — 物理残差损失
  - 坐标 y_combined = [y_data, y_ran] 拼接（数据点 + 物理配点）
  - 对 y_combined 做 forward，用 autograd 计算一阶、二阶导数
  - 二阶导数通过 autograd.grad 实现（计算图保留）

- 初始权重：a=1, b=1
- 动态调整：从 epoch 2000 开始，a 衰减（使 PDE loss 权重相对增大）

---

## 5. 训练配置

| 参数 | 值 | 含义 |
|------|------|------|
| nz, nx | 140×140 | 物理网格尺寸 |
| dh | 20m | 网格间距 |
| pml_total | 20 | PML 总层数 |
| pml_crop | 15 | 裁剪 PML 层数 |
| pml_active | 5 | 实际参与训练的 PML 层数 |
| batch_size | 800 | 坐标采样点数 |
| batch_size_v | 1 | 速度场 batch |
| accumulation_steps | 4 | 梯度累加 |
| lr | 1e-4 | 初始学习率 |
| NIter | 5001 | 总 epoch |
| nvel_train | 1500 | 训练速度模型数 |
| boundary_type | free_surface | 顶部自由表面 + 其他三边 PML |

空间采样：Halton 准随机采样，采样比例 50%（约 9800 个点）。

---

## 6. y_ran 自适应采样（当前实现 & 改进方案）

### 6.1 当前实现

每 batch 生成 y_ran [B_v, 900, 2]：
- 50% 按速度场梯度幅值 multinomial 采样
- 50% 在 z < 2 个网格深度（约 40m）均匀采样

问题：依赖当前 vel_batch，无法跨 batch 共享；显存约束要求 y_ran 为 [1, N, 2]。

### 6.2 改进方案（已设计，待实施）

**Epoch 级残差引导自适应采样**：

1. 训练前：基于训练集速度场统计生成初始重要性图
   - 速度梯度幅值（捕获地层界面）
   - 波长因子 1/v²（低速区需要更密采样）

2. 每 50 epoch：在粗网格(35×35)上评估 5 个速度模型的 PDE 残差
   - 残差图上采样到完整网格 (140×140)
   - 混合：70% 残差图 + 30% 初始统计先验
   - 从混合重要性图重新采样 y_ran [1, 900, 2]

3. 采样点构成：20% 表层（z < 5 个网格 ≈ 100m）+ 80% 重要性加权

---

## 7. FiLM 的物理意义（已分析）

在 Helmholtz 方程求解器中，FiLM 的仿射变换 (1+γ)·x + β 的物理含义：

**γ (缩放) — 波长/振幅调制**：
- γ > 0: 放大某些频率分量（低速区需要更短波长基函数）
- γ < 0: 反转并抑制（模拟波阻抗界面的相位反转）
- γ ≈ -1: 关闭该通道（该基函数对当前速度结构不相关）
- 原文发现：γ 远比 β 重要（移除 γ 精度降 65.4% vs β 降 1.0%）

**β (偏移) — 基线/相位校正**：
- 调整预测基线到合理的残余散射场水平
- 移动各基函数通道的激活阈值
- 等效于调整空间频率分量的参考相位

**(1+γ) 的设计**：
- 初始化时 γ=0 → 恒等变换，Trunk 是纯坐标网络
- 训练过程 γ 渐变 → 基函数从通用型变为速度感知型
- 保证初始梯度流

**三层层次化调制**：
- FiLM₁: 粗尺度 — 区分高低速区域
- FiLM₂: 中尺度 — 识别阻抗界面和梯度带
- FiLM₃: 细尺度 — 捕捉局部散射体和绕射

**核心数学意义**：FiLM 使 DeepONet 从线性算子逼近器（固定基函数线性组合）升级为非线性算子逼近器（输入函数调制的基函数组合）。

---

## 8. 当前数学问题方向（讨论起点）

可继续探讨的数学方向：
- Helmholtz 方程在 PML 边界下的变分形式与 PINN 残差的等价性
- FNO 在函数空间上的通用逼近定理 (Universal Approximation Theorem for Neural Operators)
- FiLM 调制下的 DeepONet 表达能力分析
- PDE 残差作为损失函数的收敛性保证
- 自适应配点采样的最优性分析
- 频域散射问题的适定性 (well-posedness)
- 复数波场的神经网络参数化（实部/虚部 vs 幅值/相位）
