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
        self.device = args.device
        self.feat_dim = 256  # 特征维度，必须能被注意力头数整除
        
        # --- 超参数 ---
        input_shape_branch1 = args.input_shape_branch1
        input_shape_branch2 = args.input_shape_branch2
        self.b2 = args.batch_size
        
        # --- 编码器与特征提取 ---
        self.pos_encoder = PositionalEncoding(embed_dim=4)
        self.fencoder = FourierFeatureEncoder(input_dim=2, mapping_size=self.feat_dim)
        
        self.pos_scale = nn.Parameter(torch.tensor(0.1))  # 位置编码缩放因子
        self.pos_encoding = self._generate_global_pos_encoding()  # 全局位置编码

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
        self.combinedlayer1 = GaussianWeightedLayer(self.feat_dim)
        self.combinedlayer2 = GaussianWeightedLayer(self.feat_dim)
        self.attengate = AttenGate(use_softmax=True)
        
        self.block_feature_encoder = BlockFeatureEncoder(self.feat_dim, self.feat_dim, grid_size=20)
        self.smooth_feature_encoder = SmoothBlockEncoder(self.feat_dim, self.feat_dim, grid_size=20)

        # --- 主干网络 (Trunk) 与输出层 ---
        self.trunk = FiLMTrunk(input_dim=16, width=self.feat_dim)
        self.final_layer = nn.Linear(self.feat_dim, 2)  # 输出实部和虚部
        
        # --- 损失函数组件 ---
        self.loss_function = nn.MSELoss(reduction='mean')
        self.loss_function_point = nn.MSELoss(reduction='none')
        
        # 动态损失权重参数
        self.log_var_data = nn.Parameter(torch.zeros(1))
        self.log_var_pde = nn.Parameter(torch.zeros(1))

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

    def _generate_global_pos_encoding(self):
        """生成全局位置编码，适配 (b1, b2, feat_dim) 维度"""
        max_b2 = 6400
        position = torch.arange(max_b2).unsqueeze(1)  # [max_b2, 1]
        div_term = torch.exp(
            torch.arange(0, self.feat_dim, 2, dtype=torch.float) 
            * (-torch.log(torch.tensor(10000.0)) / self.feat_dim)
        )
        pe = torch.zeros(max_b2, self.feat_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_b2, feat_dim]
        return nn.Parameter(pe, requires_grad=False)
    
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
        y_normalized = 2 * (y - 0) / (40 * x_dim - 0) - 1
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
        
    def loss_PDE_Scatter_pml(self, vel, y, UU0):
        """
        计算包含 PML 吸收边界条件的散射场 Helmholtz 方程物理残差损失。
        """
        y.requires_grad_(True)

        batch_size_v = vel.shape[0]
        batch_size_pts = y.shape[1]
        y_sample = y.expand(batch_size_v, -1, -1)
        
        Z_dim = vel.shape[2]
        X_dim = vel.shape[3]
        SPATIAL_SCALE = 40.0 

        # --- 1. 坐标归一化与 Grid 构造 ---
        z_pixel = y_sample[:, :, 0] / SPATIAL_SCALE
        x_pixel = y_sample[:, :, 1] / SPATIAL_SCALE
        z_norm = 2 * (z_pixel / (Z_dim - 1)) - 1
        x_norm = 2 * (x_pixel / (X_dim - 1)) - 1
        
        grid = torch.stack([x_norm, z_norm], dim=-1).unsqueeze(1)  # [B_v, 1, B_pts, 2]

        # --- 2. 可微双线性插值采样 ---
        # 采样速度场 c
        c_sampled = F.grid_sample(vel[:, :1, :, :], grid, mode='bilinear', padding_mode='border', align_corners=True)
        c = c_sampled.view(batch_size_v, batch_size_pts)
        
        # 采样背景场 U0
        U0_sampled = F.grid_sample(UU0, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(2)
        U0_real = U0_sampled[:, 0, :]
        U0_imag = U0_sampled[:, 1, :]
        
        # --- 3. 物理常数与衰减因子准备 ---
        c0 = torch.ones_like(c) * 1.5
        f, f0 = 5, 10
        omega = 2 * np.pi * f * 1e-3
        k = (1 / c) ** 2
        k0 = (1 / c0) ** 2
        
        Q = 15
        alpha = 1 / Q
        rhot = (1 - alpha / np.pi * np.log(f / 50) - 1j * alpha / 2) ** 2
        
        kr, ki = k * np.real(rhot), k * np.imag(rhot)
        k0r, k0i = k0 * np.real(rhot), k0 * np.imag(rhot)

        a0 = 1.79
        C = a0 * f0 / f

        # --- 4. 一阶与二阶导数计算 (Autograd) ---
        Delta_U = self.forward(vel, y, UU0)
        Delta_U_real, Delta_U_imag = Delta_U[:, :, 0], Delta_U[:, :, 1]
        
        zz, xx = y[:, :, 0], y[:, :, 1]
        ld = (Z_dim - 70) / 2
        
        lx = F.relu(((ld - 0.5) * 40 - xx) / ((ld - 0.5) * 40)) + F.relu((xx - (69.5 + ld) * 40) / ((ld - 0.5) * 40))
        lz = F.relu(((ld - 0.5) * 40 - zz) / ((ld - 0.5) * 40)) + F.relu((zz - (69.5 + ld) * 40) / ((ld - 0.5) * 40))
        
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
        y_norm = 2 * (y - 0) / (40 * 72) - 1 
        z_enc = self.pos_encoder(y_norm[:, :, 0:1])
        x_enc = self.pos_encoder(y_norm[:, :, 1:2])
        
        y_encoded = torch.cat([z_enc, x_enc], dim=-1)
        physical_context = get_local_physical_features(vel, y, eps=1e-3)
        y_encoded = torch.cat([y_encoded, physical_context], dim=-1)
        
        return self.trunk(y_encoded)

    def generate_structure_aware_y_ran(self, vel, num_pts=20000, max_z=72.0, max_x=72.0):
        """
        结构感知自适应采样点生成。
        根据速度场空间梯度的高低，自适应分配采样点（70% 结构点，30% 均匀点）。
        """
        B_v = vel.shape[0]
        device = vel.device
        
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
                
                num_structure = int(num_pts * 0.7)
                num_uniform = num_pts - num_structure
                
                # --- 抽取结构边界点 ---
                if num_structure > 0:
                    sampled_indices = torch.multinomial(prob_dist, num_samples=num_structure, replacement=True)
                    z_idx = sampled_indices // vel.shape[3]
                    x_idx = sampled_indices % vel.shape[3]
                    
                    dz, dx = max_z / vel.shape[2], max_x / vel.shape[3]
                    z_coords = z_idx.float() * dz + (torch.rand(num_structure, device=device) * dz)
                    x_coords = x_idx.float() * dx + (torch.rand(num_structure, device=device) * dx)
                    y_struct = torch.stack([z_coords, x_coords], dim=1)
                else:
                    y_struct = torch.empty((0, 2), device=device)
                    
                # --- 抽取全局均匀分布点 ---
                z_uni = torch.rand(num_uniform, device=device) * max_z
                x_uni = torch.rand(num_uniform, device=device) * max_x
                y_uni = torch.stack([z_uni, x_uni], dim=1)
                
                y_ran_list.append(torch.cat([y_struct, y_uni], dim=0))
                
        y_ran = torch.stack(y_ran_list, dim=0)
        return y_ran.requires_grad_(True)

    def envelope_barrier_loss(self, vel, y, UU0, u_fno, lambda_env=1.0):
        """计算波场包络的流形屏障惩罚损失，消除高频相位错位的影响"""
        u_pred = self.forward(vel, y, UU0)
        
        env_pred = torch.sqrt(u_pred[..., 0]**2 + u_pred[..., 1]**2 + 1e-8)
        env_fno = torch.sqrt(u_fno[..., 0]**2 + u_fno[..., 1]**2 + 1e-8)
        
        loss_env = torch.abs(env_pred - env_fno)
        return torch.mean(loss_env)

        
    def loss(self, vel, y, UU0, labels, a, b, c, data_norm_coe=1., pde_norm_coe=1.):
        """
        核心损失函数计算接口
        """
        batch_size_v = vel.shape[0]
        nz, nx = vel.shape[2], vel.shape[3]
        
        # 1. 提取标签坐标 (根据给定的 y)
        batch_idx = torch.arange(batch_size_v, device=labels.device)[:, None]
        z_coord = (y[:, :, 0] / 40.0).long().clamp(0, nz - 1)
        x_coord = (y[:, :, 1] / 40.0).long().clamp(0, nx - 1)
        labels = labels[batch_idx, :, z_coord, x_coord] 
        
        # 2. 生成自适应物理采样点
        y_ran = self.generate_structure_aware_y_ran(vel, num_pts=900, max_z=nz, max_x=nx)
        
        # 3. 计算各项基础损失
        loss_u = self.loss_BC(vel, y, UU0, labels) / data_norm_coe
        loss_f = self.loss_PDE_Scatter_pml(vel, y, UU0) / pde_norm_coe
        loss_f_ran = self.loss_PDE_Scatter_pml(vel, y_ran, UU0) / pde_norm_coe
        
        loss_r = 0.0 # 占位
        
        # 4. 根据权重加权求和
        loss_val = (a * loss_u) + b * (loss_f + loss_f_ran)

        return loss_val, loss_f + loss_f_ran, loss_u, loss_r