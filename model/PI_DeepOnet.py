import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Labconfig import *
from model.utils import *
from model.dataloader import *
from model.net_module import *

class Pi_DeepONet(nn.Module):
    """
    物理信息神经网络 (PI-DeepONet)
    结合 FNO (Branch) 与 FiLM/Fourier 编码 (Trunk)，用于求解带 PML 边界条件的 Helmholtz 方程。
    """
    def __init__(self, args):
        super().__init__()
        self.args = args  # 保存args以便其他方法使用
        self.device = args.device
        self.feat_dim = 256  # 特征维度，必须能被注意力头数整除

        # --- 超参数 ---
        input_shape_branch1 = args.input_shape_branch1
        input_shape_branch2 = args.input_shape_branch2
        self.b2 = args.batch_size
        
        # --- 编码器与特征提取 ---
        self.pos_encoder = PositionalEncoding(embed_dim=4)
        # self.fencoder = FourierFeatureEncoder(input_dim=2, mapping_size=self.feat_dim)  # 未使用，已注释

        # --- 网络分支 (Branch) ---
        self.branch1 = nn.Sequential(
            FNO2d(input_shape_branch1[1], self.feat_dim, modes1=16, modes2=16, width=32),
        )
        self.branch2 = nn.Sequential(
            FNO2d(input_shape_branch2[1], self.feat_dim, modes1=16, modes2=16, width=32),
        )
        
        # --- 注意力与特征融合 ---
        self.channel_attention1 = ChannelAttention(self.feat_dim, reduction=8)
        self.channel_attention2 = ChannelAttention(self.feat_dim, reduction=8)
        self.combinedlayer1 = GaussianWeightedLayer(self.feat_dim, dh=args.dh)
        self.combinedlayer2 = GaussianWeightedLayer(self.feat_dim, dh=args.dh)
        self.attengate = AttenGate(use_softmax=True)

        self.smooth_feature_encoder = SmoothBlockEncoder(self.feat_dim, self.feat_dim, grid_size=20)

        # --- 主干网络 (Trunk) 与输出层 ---
        self.trunk = FiLMTrunk(input_dim=16, width=self.feat_dim)
        self.final_layer = nn.Linear(self.feat_dim, 2)  # 输出实部和虚部
        
        # --- 损失函数组件 ---
        self.loss_function = nn.MSELoss(reduction='mean')
        # self.loss_function_point = nn.MSELoss(reduction='none')  # 未使用，已注释

        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

    def forward(self, vel, y, UU0):
        """
        前向传播
        Args:
            vel: 速度场模型 [B_v, C, Z, X]
            y: 查询坐标点 [B_v, B_pts, 2]
            UU0: 背景波场 [B_v, 2, Z, X]
        Returns:
            outputs: 预测波场残差 (实部和虚部) [B_v, B_pts, 2]
        """
        x_dim = vel.shape[-1]
        
        # --- 1. 坐标预处理与 Trunk 特征提取 (Query) ---
        y_normalized = 2 * (y - 0) / (self.args.dh * x_dim - 0) - 1
        z_normalized = y_normalized[:, :, 0].unsqueeze(-1)
        x_normalized = y_normalized[:, :, 1].unsqueeze(-1)
        
        z_encoded = self.pos_encoder(z_normalized)
        x_encoded = self.pos_encoder(x_normalized)
        y_encoded = torch.cat([z_encoded, x_encoded], dim=2)  # [B_v, B_pts, 16]
        
        # --- 2. Branch 特征提取与 Tokenization (Memory/Key-Value) ---
        B1_raw = self.branch1(vel)
        B2_raw = self.branch2(UU0)
        
        B1_raw = self.channel_attention1(B1_raw)
        B2_raw = self.channel_attention2(B2_raw)
        
        B1_feat = self.combinedlayer1(vel, y[0], B1_raw)
        B2_feat = self.combinedlayer2(vel, y[0], B2_raw, False)
        
        # 注意力门控与特征平滑融合
        B = self.attengate(B1_feat, B2_feat)
        B_encoded = self.smooth_feature_encoder(B1_raw + B2_raw, y_normalized)
        
        # --- 3. Trunk 与 Branch 融合输出 ---
        T_raw = self.trunk(y_encoded, B_encoded)
        outputs = self.final_layer(B * T_raw)
        
        return outputs
                        
    def loss_BC(self, vel, y, UU0, labels):
        """计算数据拟合损失 (Data/BC Loss)"""
        pred = self.forward(vel, y, UU0)
        loss_u = self.loss_function(pred, labels)
        return loss_u

    def dynamic_barrier_loss(self, error, r0=8, lambda_aux=1.0):
        """
        带动态自适应系数的流形屏障惩罚函数。
        在安全区 (r0) 内部，牵引力系数连续衰减，在圆心处严格为 0。
        """
        x = torch.clamp(error / (r0 + 1e-8), min=0.0, max=1.0)
        dynamic_coeff = lambda_aux * (x ** 2)
        return dynamic_coeff * error
        
    def loss_PDE_Scatter_pml(self, vel, y, UU0, freq_batch=None):
        """
        计算包含 PML 吸收边界条件的散射场 Helmholtz 方程物理残差损失。
        （向后兼容接口，内部调用 _compute_pde_residual）

        Args:
            freq_batch: 每个样本对应的频率值 [B_v]。若为 None 则使用默认值 5 Hz。
        """
        y.requires_grad_(True)
        Delta_U = self.forward(vel, y, UU0)
        return self._compute_pde_residual(vel, y, UU0, Delta_U, freq_batch=freq_batch)

    def _compute_pde_residual(self, vel, y, UU0, Delta_U, freq_batch=None):
        """
        计算包含 PML 吸收边界条件的散射场 Helmholtz 方程物理残差。
        接受已计算好的 Delta_U（forward 输出），避免重复前向传播。

        Args:
            vel: 速度场 [B_v, 1, Z, X]
            y: 坐标点 [B_v, N, 2]，需要 requires_grad=True
            UU0: 背景波场 [B_v, 2, Z, X]
            Delta_U: forward 输出 [B_v, N, 2]
            freq_batch: 每个样本对应的频率值 [B_v]
        """
        batch_size_v = vel.shape[0]
        batch_size_pts = y.shape[1]
        y_sample = y.expand(batch_size_v, -1, -1)

        Z_dim = vel.shape[2]
        X_dim = vel.shape[3]
        SPATIAL_SCALE = float(self.args.dh)

        # --- 1. 坐标归一化与 Grid 构造 ---
        z_pixel = y_sample[:, :, 0] / SPATIAL_SCALE
        x_pixel = y_sample[:, :, 1] / SPATIAL_SCALE
        z_norm = 2 * (z_pixel / (Z_dim - 1)) - 1
        x_norm = 2 * (x_pixel / (X_dim - 1)) - 1

        grid = torch.stack([x_norm, z_norm], dim=-1).unsqueeze(1)  # [B_v, 1, B_pts, 2]

        # --- 2. 可微双线性插值采样 ---
        c_sampled = F.grid_sample(vel[:, :1, :, :], grid, mode='bilinear', padding_mode='border', align_corners=True)
        c = c_sampled.view(batch_size_v, batch_size_pts).detach()

        U0_sampled = F.grid_sample(UU0, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(2)
        U0_real = U0_sampled[:, 0, :].detach()
        U0_imag = U0_sampled[:, 1, :].detach()

        # --- 3. 物理常数与衰减因子准备 ---
        c0 = torch.ones_like(c) * 1.5
        f0 = 10
        if freq_batch is not None:
            f = freq_batch.unsqueeze(1).expand(batch_size_v, batch_size_pts)
        else:
            f = float(self.args.default_freq)
        omega = 2 * torch.pi * f * 1e-3
        k = (1 / c) ** 2
        k0 = (1 / c0) ** 2

        Q = 15
        alpha = 1 / Q
        rhot = (1 - alpha / torch.pi * torch.log(f / 50) - 1j * alpha / 2) ** 2

        kr, ki = k * torch.real(rhot), k * torch.imag(rhot)
        k0r, k0i = k0 * torch.real(rhot), k0 * torch.imag(rhot)

        a0 = 1.79
        C = a0 * f0 / f

        # --- 4. 一阶与二阶导数计算 (Autograd) ---
        Delta_U_real, Delta_U_imag = Delta_U[:, :, 0], Delta_U[:, :, 1]

        zz, xx = y[:, :, 0], y[:, :, 1]

        ld = self.args.pml_active

        with torch.no_grad():
            dh = self.args.dh
            # 物理区域边界: nx/nz - 0.5 + pml_active
            x_boundary = (X_dim - 2 * ld - 0.5 + ld)  # (nx - 0.5 + ld)
            z_boundary = (Z_dim - ld - 0.5 + ld) if self.args.boundary_type == 'free_surface' else (Z_dim - 2 * ld - 0.5 + ld)

            lx = F.relu(((ld - 0.5) * dh - xx) / ((ld - 0.5) * dh)) + \
                 F.relu((xx - x_boundary * dh) / ((ld - 0.5) * dh))

            if self.args.boundary_type == 'free_surface':
                lz = F.relu((zz - z_boundary * dh) / ((ld - 0.5) * dh))
            else:
                lz = F.relu(((ld - 0.5) * dh - zz) / ((ld - 0.5) * dh)) + \
                     F.relu((zz - z_boundary * dh) / ((ld - 0.5) * dh))

            pml_tmp1 = C ** 2 * lx ** 2 * lz ** 2
            pml_tmp2 = C ** 2 * lx ** 4
            pml_tmp3 = C ** 2 * lz ** 4
            pml_tmp4 = C * (lz ** 2 - lx ** 2)
            pml_tmp5 = C * (lx ** 2 + lz ** 2)

        # 计算一阶导数
        Delta_U_grad_real = torch.autograd.grad(Delta_U_real, y, grad_outputs=torch.ones_like(Delta_U_real), create_graph=True, retain_graph=True, only_inputs=True)[0]
        Delta_U_grad_imag = torch.autograd.grad(Delta_U_imag, y, grad_outputs=torch.ones_like(Delta_U_imag), create_graph=True, retain_graph=True, only_inputs=True)[0]

        Delta_Uz_real, Delta_Ux_real = Delta_U_grad_real[:, :, 0], Delta_U_grad_real[:, :, 1]
        Delta_Uz_imag, Delta_Ux_imag = Delta_U_grad_imag[:, :, 0], Delta_U_grad_imag[:, :, 1]

        # 修正的一阶导数 (带 PML)
        eu_zr = (1 + pml_tmp1) / (1 + pml_tmp3) * Delta_Uz_real - pml_tmp4 / (1 + pml_tmp3) * Delta_Uz_imag
        eu_xr = (1 + pml_tmp1) / (1 + pml_tmp2) * Delta_Ux_real + pml_tmp4 / (1 + pml_tmp2) * Delta_Ux_imag
        eu_zi = pml_tmp4 / (1 + pml_tmp3) * Delta_Uz_real + (1 + pml_tmp1) / (1 + pml_tmp3) * Delta_Uz_imag
        eu_xi = -pml_tmp4 / (1 + pml_tmp2) * Delta_Ux_real + (1 + pml_tmp1) / (1 + pml_tmp2) * Delta_Ux_imag

        # 计算二阶导数
        Delta_Uzz_real = torch.autograd.grad(eu_zr, y, grad_outputs=torch.ones_like(eu_zr), create_graph=True, retain_graph=True, only_inputs=True)[0][:, :, 0]
        Delta_Uxx_real = torch.autograd.grad(eu_xr, y, grad_outputs=torch.ones_like(eu_xr), create_graph=True, retain_graph=True, only_inputs=True)[0][:, :, 1]
        Delta_Uzz_imag = torch.autograd.grad(eu_zi, y, grad_outputs=torch.ones_like(eu_zi), create_graph=True, retain_graph=True, only_inputs=True)[0][:, :, 0]
        Delta_Uxx_imag = torch.autograd.grad(eu_xi, y, grad_outputs=torch.ones_like(eu_xi), create_graph=True, retain_graph=True, only_inputs=True)[0][:, :, 1]

        # --- 5. 组合 PDE 残差 ---
        ur_r = (1 - pml_tmp1) * omega ** 2 * (kr * (Delta_U_real + U0_real) - ki * (Delta_U_imag + U0_imag))
        ui_r = pml_tmp5 * omega ** 2 * (kr * (Delta_U_imag + U0_imag) + ki * (Delta_U_real + U0_real))
        u0r_r = (1 - pml_tmp1) * omega ** 2 * (-k0r * U0_real + k0i * U0_imag)
        u0i_r = pml_tmp5 * omega ** 2 * (-k0r * U0_imag - k0i * U0_real)

        ur_i = (-pml_tmp5) * omega ** 2 * (kr * (Delta_U_real + U0_real) - ki * (Delta_U_imag + U0_imag))
        ui_i = (1 - pml_tmp1) * omega ** 2 * (kr * (Delta_U_imag + U0_imag) + ki * (Delta_U_real + U0_real))
        u0r_i = (-pml_tmp5) * omega ** 2 * (-k0r * U0_real + k0i * U0_imag)
        u0i_i = (1 - pml_tmp1) * omega ** 2 * (-k0r * U0_imag - k0i * U0_real)

        residual_real = Delta_Uzz_real + Delta_Uxx_real + ur_r + ui_r + u0r_r + u0i_r
        residual_imag = Delta_Uzz_imag + Delta_Uxx_imag + ur_i + ui_i + u0r_i + u0i_i

        return torch.mean(residual_real ** 2 + residual_imag ** 2)
    
    def loss_Reg(self, vel, y, UU0, source_coord):
        """震源区域正则化损失"""
        z_coord, x_coord = y[:, 0], y[:, 1]
        source_z, source_x = source_coord[:, 0], source_coord[:, 1]
        
        inside_distance = 100 - torch.sqrt((z_coord - source_z) ** 2 + (x_coord - source_x) ** 2)
        coe = F.relu(inside_distance) / (inside_distance + 1e-15)

        pred = self.forward(vel, y, UU0)
        N_reg = torch.clamp(torch.count_nonzero(coe), min=1.0).to(vel.device)

        return torch.sum(coe * (pred[:, 0] ** 2 + pred[:, 1] ** 2)) / N_reg

    def loss_op(self, model0, vel, y, UU0):
        """模型间操作损失 (如知识蒸馏或微调约束)"""
        with torch.no_grad():
            pred0 = model0(vel, y, UU0)
        pred_ft = self.forward(vel, y, UU0)
        return torch.sum((pred0 - pred_ft) ** 2)
        
    def get_ortho_loss(self, T, weight):
        """
        计算基底正交性损失。
        通过归一化 Gram 矩阵，使 Trunk 输出在序列维度上互相正交。
        """
        B_v, N, p = T.shape
        gram = torch.bmm(T.transpose(-2, -1), T)
        
        diag = torch.diagonal(gram, dim1=-2, dim2=-1).unsqueeze(-1) + 1e-8
        gram_normalized = gram / torch.sqrt(diag @ diag.transpose(-2, -1))
        
        gram_matrix = torch.bmm(T.transpose(1, 2), T) / N
        eye = torch.eye(p, device=T.device).unsqueeze(0).expand(B_v, -1, -1)
        
        loss = torch.mean((gram_matrix - eye) ** 2)
        return loss * weight
    
    def get_trunk_output(self, vel, y):
        """独立提取 Trunk 网络的基底输出"""
        # 修复：使用实际网格尺寸而不是硬编码 72
        Z_dim = vel.shape[2]
        X_dim = vel.shape[3]
        y_norm = 2 * (y - 0) / (self.args.dh * X_dim) - 1  # 使用实际x维度
        z_enc = self.pos_encoder(y_norm[:, :, 0:1])
        x_enc = self.pos_encoder(y_norm[:, :, 1:2])

        y_encoded = torch.cat([z_enc, x_enc], dim=-1)
        physical_context = get_local_physical_features(vel, y, eps=1e-3)
        y_encoded = torch.cat([y_encoded, physical_context], dim=-1)

        return self.trunk(y_encoded)

    def generate_structure_aware_y_ran(self, vel, num_pts=20000, max_z=None, max_x=None):
        """
        结构感知自适应采样点生成。
        根据速度场空间梯度的高低，自适应分配采样点（50% 结构点，50% 表层点）。

        Args:
            vel: 速度场 [B, C, Z, X]
            num_pts: 采样点数量
            max_z: z方向最大坐标（默认使用vel的实际尺寸）
            max_x: x方向最大坐标（默认使用vel的实际尺寸）

        Returns:
            y_ran: 采样点坐标 [B, num_pts, 2]，requires_grad=True
        """
        # 修复：如果未提供max_z和max_x，使用vel的实际尺寸
        if max_z is None:
            max_z = float(vel.shape[2]) * self.args.dh  # 转换为实际坐标
        if max_x is None:
            max_x = float(vel.shape[3]) * self.args.dh

        B_v = vel.shape[0]
        device = vel.device

        # 计算网格步长
        dz = max_z / vel.shape[2]  # z方向网格步长
        dx = max_x / vel.shape[3]  # x方向网格步长

        with torch.no_grad():
            # 计算空间梯度幅度
            grad_z = vel[:, :, 2:, 1:-1] - vel[:, :, :-2, 1:-1]
            grad_x = vel[:, :, 1:-1, 2:] - vel[:, :, 1:-1, :-2]
            vel_grad_mag = torch.sqrt(grad_z**2 + grad_x**2 + 1e-8)
            vel_grad_mag = F.pad(vel_grad_mag, (1, 1, 1, 1), mode='replicate').squeeze(1)

            y_ran_list = []
            for b in range(B_v):
                prob_dist = vel_grad_mag[b].view(-1)
                prob_dist = prob_dist / (prob_dist.sum() + 1e-8)

                # 修改采样策略：50% 结构点，50% 表层点
                num_structure = int(num_pts * 0.5)
                num_surface = num_pts - num_structure

                # --- 1. 抽取结构边界点（50%）---
                if num_structure > 0:
                    sampled_indices = torch.multinomial(prob_dist, num_samples=num_structure, replacement=True)
                    z_idx = sampled_indices // vel.shape[3]
                    x_idx = sampled_indices % vel.shape[3]

                    z_coords = z_idx.float() * dz + (torch.rand(num_structure, device=device) * dz)
                    x_coords = x_idx.float() * dx + (torch.rand(num_structure, device=device) * dx)
                    y_struct = torch.stack([z_coords, x_coords], dim=1)
                else:
                    y_struct = torch.empty((0, 2), device=device)

                # --- 2. 抽取表层点（50%）---
                # 表层定义：z < 2 个网格点的深度范围
                if num_surface > 0:
                    # 表层深度范围：[0, 2*dz]
                    surface_depth = 2.0 * dz

                    # 在表层范围内随机采样 z 坐标
                    z_surf = torch.rand(num_surface, device=device) * surface_depth

                    # x 坐标在整个范围内均匀采样
                    x_surf = torch.rand(num_surface, device=device) * max_x

                    y_surf = torch.stack([z_surf, x_surf], dim=1)
                else:
                    y_surf = torch.empty((0, 2), device=device)

                # 合并结构点和表层点
                y_ran_list.append(torch.cat([y_struct, y_surf], dim=0))

        y_ran = torch.stack(y_ran_list, dim=0)
        return y_ran.requires_grad_(True)

    def envelope_barrier_loss(self, vel, y, UU0, u_fno, lambda_env=1.0):
        """计算波场包络的流形屏障惩罚损失，消除高频相位错位的影响"""
        u_pred = self.forward(vel, y, UU0)
        
        env_pred = torch.sqrt(u_pred[..., 0]**2 + u_pred[..., 1]**2 + 1e-8)
        env_fno = torch.sqrt(u_fno[..., 0]**2 + u_fno[..., 1]**2 + 1e-8)
        
        loss_env = torch.abs(env_pred - env_fno)
        return torch.mean(loss_env)

        
    def loss(self, vel, y, UU0, labels, a, b, c, data_norm_coe=1., pde_norm_coe=1., freq_batch=None, y_ran=None):
        """
        核心损失函数计算接口。
        优化：只做一次 forward pass，同时服务于 BC loss 和 PDE loss。

        Args:
            freq_batch: 每个样本对应的频率值 [B_v]。若为 None 则使用默认值。
            y_ran: 预计算的自适应采样点 [B_v, N_ran, 2]。若提供则拼接到 y 后；若为 None 则不生成。
        """
        batch_size_v = vel.shape[0]
        nz, nx = vel.shape[2], vel.shape[3]

        # 1. 提取标签坐标 (根据给定的 y)
        batch_idx = torch.arange(batch_size_v, device=labels.device)[:, None]
        z_coord = (y[:, :, 0] / self.args.dh).long().clamp(0, nz - 1)
        x_coord = (y[:, :, 1] / self.args.dh).long().clamp(0, nx - 1)
        labels = labels[batch_idx, :, z_coord, x_coord]  # [B_v, 2, B_pts] -> [B_v, B_pts, 2]

        # 2. 拼接自适应采样点 (仅当显式提供 y_ran 时)
        n_y = y.shape[1]
        if y_ran is not None and y_ran.shape[1] > 0:
            y_combined = torch.cat([y, y_ran], dim=1)
        else:
            y_combined = y

        y_combined.requires_grad_(True)

        # 3. 只做一次 forward pass (服务于 BC loss 和 PDE loss)
        Delta_U = self.forward(vel, y_combined, UU0)

        # 4. BC loss: 使用前 n_y 个点的预测
        pred_y = Delta_U[:, :n_y, :]
        loss_u = self.loss_function(pred_y, labels) / data_norm_coe

        # 5. PDE loss: 使用完整输出计算物理残差
        loss_f_combined = self._compute_pde_residual(vel, y_combined, UU0, Delta_U, freq_batch=freq_batch) / pde_norm_coe

        loss_r = 0.0  # 占位

        # 6. 根据权重加权求和
        loss_val = (a * loss_u) + b * loss_f_combined

        return loss_val, loss_f_combined, loss_u, loss_r

    def compute_loss(self, Delta_U, vel, y, UU0, labels, y_combined,
                     a, b, c, data_norm_coe=1., pde_norm_coe=1., freq_batch=None):
        """
        在 DDP forward 之后计算损失 (不包含 forward 调用)。
        供 DDP 训练使用：先通过 DDP wrapper 调用 forward()，再调用此方法计算 loss。

        Args:
            Delta_U: model.forward() 的输出 [B_v, B_pts, 2]
            vel: 速度场 [B_v, 1, Z, X]
            y: 数据坐标点 [B_v, B_data_pts, 2]
            UU0: 背景波场 [B_v, 2, Z, X]
            labels: 标签波场 [B_v, 2, Z, X]
            y_combined: 拼接后的坐标 [B_v, B_data_pts + B_ran_pts, 2]，requires_grad=True
            a, b, c: 损失权重
            data_norm_coe: 数据损失归一化系数
            pde_norm_coe: PDE 损失归一化系数
            freq_batch: 频率值 [B_v]
        Returns:
            (total_loss, loss_f, loss_u, loss_r) — 与 loss() 返回格式一致
        """
        batch_size_v = vel.shape[0]
        nz, nx = vel.shape[2], vel.shape[3]
        n_y = y.shape[1]

        # 1. 提取标签值
        batch_idx = torch.arange(batch_size_v, device=labels.device)[:, None]
        z_coord = (y[:, :, 0] / self.args.dh).long().clamp(0, nz - 1)
        x_coord = (y[:, :, 1] / self.args.dh).long().clamp(0, nx - 1)
        labels_extracted = labels[batch_idx, :, z_coord, x_coord]

        # 2. 数据拟合损失
        pred_y = Delta_U[:, :n_y, :]
        loss_u = self.loss_function(pred_y, labels_extracted) / data_norm_coe

        # 3. PDE 物理残差损失
        loss_f = self._compute_pde_residual(vel, y_combined, UU0, Delta_U, freq_batch=freq_batch) / pde_norm_coe

        loss_r = 0.0

        # 4. 加权求和
        loss_val = (a * loss_u) + b * loss_f

        return loss_val, loss_f, loss_u, loss_r