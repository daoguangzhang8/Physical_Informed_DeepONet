from Labconfig import *
from model.utils import *
from model.dataloader import *
from model.net_module import *


class Pi_DeepONet(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_shape_trunk, input_shape_branch1, input_shape_branch2 = args.input_shape_trunk, args.input_shape_branch1, args.input_shape_branch2
        self.device = args.device
        encoded_dim = 4
        self.feat_dim = 256  # 必须能被 num_heads 整除（72%4=0）
        self.num_heads = 1   # 多头注意力头数
        self.depth = 6     # Transformer 层数
        self.pos_encoder = PositionalEncoding(encoded_dim)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_tokens = 3  # branch1, branch2, trunk
        self.b2 = args.batch_size
        self.pos_encoding = self._generate_global_pos_encoding()  # 全局位置编码
        self.pos_scale = nn.Parameter(torch.tensor(0.1))  # 位置编码缩放因子
        self.fencoder = FourierFeatureEncoder(2,256)
        self.attengate = AttenGate(use_softmax=True)
        self.combinedlayer1 = GaussianWeightedLayer(self.feat_dim)
        self.combinedlayer2 = GaussianWeightedLayer(self.feat_dim)
        self.channel_attention1 = ChannelAttention(self.feat_dim, reduction=8)
        self.channel_attention2 = ChannelAttention(self.feat_dim, reduction=8)
        self.data_norm_coe = 1.
        self.pde_norm_coe = 1.
        self.pde_real_k = 0.
        self.pde_imag_k = 0.

        self.log_var_data = nn.Parameter(torch.zeros(1))
        self.log_var_pde = nn.Parameter(torch.zeros(1))
        
        self.b2 = args.batch_size

        self.branch1 = nn.Sequential(
            FNO2d(input_shape_branch1[1], self.feat_dim, 16, 16, 32),
        )

        self.branch2 = nn.Sequential(
            FNO2d(input_shape_branch2[1], self.feat_dim, 16, 16, 32),
        )
        self.block_feature_encoder = BlockFeatureEncoder(self.feat_dim, self.feat_dim, grid_size=20)
        self.smooth_feature_encoder = SmoothBlockEncoder(self.feat_dim, self.feat_dim, grid_size=20)

        self.trunk = FiLMTrunk(16, self.feat_dim)
        
        self.final_layer = nn.Linear(self.feat_dim, 2)
        self._init_weights()
         
        self.loss_function = nn.MSELoss(reduction='mean')
        # self.loss_function = nn.L1Loss(reduction='sum')
        self.loss_function_point = nn.MSELoss(reduction='none')

    def _init_weights(self):
        """初始化权重（保持不变）"""
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
        """生成适配 (b1, b2, feat_dim) 维度的位置编码"""
        # position 对应 token 序列长度维度 b2（而非 num_tokens）
        # 生成 0 到 (b2-1) 的位置索引，形状为 [b2, 1]
        position = torch.arange(6400).unsqueeze(1)  # [max_b2, 1]，max_b2 是 b2 的最大可能值

        # 计算频率衰减因子
        div_term = torch.exp(
            torch.arange(0, self.feat_dim, 2, dtype=torch.float) 
            * (-torch.log(torch.tensor(10000.0)) / self.feat_dim)
        )  # [feat_dim//2]

        # 初始化位置编码：[max_b2, feat_dim]（与 token 的 b2、feat_dim 维度对齐）
        pe = torch.zeros(6400, self.feat_dim)

        # 偶数维度用正弦编码，奇数维度用余弦编码
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_b2, feat_dim//2] → 广播到 [max_b2, feat_dim]
        pe[:, 1::2] = torch.cos(position * div_term)  # 同上

        # 扩展维度以匹配 token 形状 (b1, b2, feat_dim)
        # 最终形状：[1, max_b2, feat_dim]，其中 1 对应 b1（可广播），max_b2 对应 b2
        pe = pe.unsqueeze(0)  # [1, max_b2, feat_dim]

        # 注册为不可训练参数（确保设备一致性）
        return nn.Parameter(pe, requires_grad=False)
    
    def forward(self, vel, y, UU0):
        # 基础维度获取
        nn = vel.shape[-1]
        b1 = vel.shape[0]        # Batch size (B_v)
        b_pts = y.shape[1]      # Query points (B_pts)
        w = generate_weight(vel, type="energy")
        # 1. 坐标预处理与 Trunk 提取 (Query)
        # y_normlized: [B_v, B_pts, 2]
        y_normlized = 2 * (y - 0) / (40 * nn - 0) - 1
        z_normlized = y_normlized[:, :, 0].unsqueeze(-1)
        x_normlized = y_normlized[:, :, 1].unsqueeze(-1)
        
        z_encoded = self.pos_encoder(z_normlized)
        x_encoded = self.pos_encoder(x_normlized)
        y_encoded = torch.cat([z_encoded, x_encoded], dim=2) # [B_v, B_pts, 16]
        y_fencoded = self.fencoder(y_normlized)


        # print('physical_context', physical_context.shape)# vel_feature = vel_std * 1.5 + vel_mean
        # vel_feature = vel_feature.unsqueeze(1).expand(-1,b_pts,-1)
        # y_encoded = torch.cat([y_encoded, physical_context],dim=-1)
        # print('y_encoded', y_encoded.shape)
        # T 作为 Query: [B_v, B_pts, feat_dim]
        
        # q, r = torch.linalg.qr(T_raw)

        # 归一化：满足函数范数 ||t|| = 1
        # 在连续意义下 \int t^2 dx = 1，离散对应为 sum(t^2) * dx = 1
        # 这里简单乘以 sqrt(N) 确保基底数值量级稳定
        # t_ortho = q * (b_pts ** 0.5)
        # T_fused = self.Fourier_base(y_normlized)
        
        # 2. Branch 特征提取与 Tokenization (Memory/Key-Value)
        # 原始 FNO 输出: [B_v, feat_dim*2, H, W]
        B1_raw = self.branch1(vel)
        B2_raw = self.branch2(UU0)
        B1_raw = self.channel_attention1(B1_raw)
        B2_raw = self.channel_attention1(B2_raw)
        B1_feat = self.combinedlayer1(vel, y[0], B1_raw)
        B2_feat = self.combinedlayer2(vel, y[0], B2_raw, False)
        
        B = self.attengate(B1_feat, B2_feat)
        B_encoded = self.smooth_feature_encoder(B1_raw + B2_raw, y_normlized)
        
        T_raw = self.trunk(y_encoded, B_encoded)
        
        out = self.final_layer(B * T_raw)
        
        outputs = out

        # outputs = self.global_avg_pool(outputs).squeeze(-1).squeeze(-1)
        return outputs
                        
    def loss_BC(self, vel, y, UU0, labels):
        pred = self.forward(vel, y, UU0)
        # print('pred',pred.shape)
        loss_u = self.loss_function(pred, labels)

        return loss_u
    def dynamic_barrier_loss(self, error, r0=8, lambda_aux=1.0):
        """
        带动态自适应系数的流形屏障惩罚函数。
        在安全区 (r0) 内部，牵引力系数连续衰减，在圆心处严格为 0。
        
        Args:
            error: 外部传入的绝对误差张量 (shape: [Batch, ...])
            r0: 流形安全区半径 (死区阈值)
            lambda_barrier: 越界惩罚系数 (越界时产生巨大的拉回梯度)
            lambda_aux: 安全区边界处的最大辅助系数
            
        Returns:
            total_loss: 标量 Tensor
        """
        # 1. 构造动态系数: 在 0 到 r0 之间线性增长，超过 r0 截断为 1.0
        # 注意：防止除以 0，给分母加一个极小值
        x = torch.clamp(error / (r0 + 1e-8), min=0.0, max=1.0)
        
        # 2. 核心数学修正：使用严格下凸函数 x^p
        # 当 x=1 时，值为 1；当 x 稍微小于 1 时，值迅速下降
        dynamic_coeff = lambda_aux * (x ** 2)
        
        # 3. 动态辅助损失
        # 此时内部 Loss 相当于 error^(p+2)，极其平缓的盆底！
        aux_loss = dynamic_coeff * error
        
        # 4. 组合总损失
        total_loss = aux_loss
        
        return total_loss
        
    def loss_PDE_Scatter_pml(self, vel, y, UU0):
        y.requires_grad_(True)

        batch_size_v = vel.shape[0]   # B_v
        batch_size_pts = y.shape[1]   # B_pts
        y_sample = y.expand(batch_size_v, -1, -1)
        
        Z_dim = vel.shape[2]
        X_dim = vel.shape[3]
        SPATIAL_SCALE = 40.0 

        # --- 2. 坐标归一化 ---
        # 此时 y 的形状是 [B_v, B_pts, 2]
        
        # 提取 z 和 x 坐标
        z_pixel = y_sample[:, :, 0] / SPATIAL_SCALE
        x_pixel = y_sample[:, :, 1] / SPATIAL_SCALE
        
        # 归一化到 [-1, 1]
        z_norm = 2 * (z_pixel / (Z_dim - 1)) - 1
        x_norm = 2 * (x_pixel / (X_dim - 1)) - 1
        
        y_norm = torch.cat([z_norm.unsqueeze(-1),x_norm.unsqueeze(-1)],dim=-1)
        
        # 构造 grid: grid_sample 要求 [N, H_out, W_out, 2]
        # 这里 N=B_v, H_out=1, W_out=B_pts
        grid = torch.stack([x_norm, z_norm], dim=-1) # [B_v, B_pts, 2]
        grid = grid.unsqueeze(1) # 形状变为 [B_v, 1, B_pts, 2]

        # --- 3. 可微双线性插值采样 ---
        
        # 3.1 采样 c (速度)
        # vel 形状：[B_v, 1, Z_dim, X_dim]
        c_sampled = F.grid_sample(
            input=vel[:, :1, :, :], 
            grid=grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ) # 返回 [B_v, 1, 1, B_pts]
        
        c = c_sampled.view(batch_size_v, batch_size_pts) # 变为 [B_v, B_pts]
        
        # 3.2 采样 U0_real 和 U0_imag
        # UU0 形状：[B_v, 2, Z_dim, X_dim]
        U0_sampled = F.grid_sample(
            input=UU0,
            grid=grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ) # 返回 [B_v, 2, 1, B_pts]
        
        U0_sampled = U0_sampled.squeeze(2) # 变为 [B_v, 2, B_pts]
        
        U0_real = U0_sampled[:, 0, :] # [B_v, B_pts]
        U0_imag = U0_sampled[:, 1, :] # [B_v, B_pts]
        
        # print('c', c.shape)
        # c0 = vel[:, 0, 6, 40].unsqueeze(1).expand(-1,batch_size) #shape->[B_v, B]
        # c0 = vel[:, 0, 1, 35].unsqueeze(1).expand(-1,batch_size)
        c0 = torch.ones_like(c) * 1.5
        # print(c0[:10])
        # print('c0',U0_real.shape)
        # print('c0',c0[2,:,0])
        f = 5
        f0 = 10
        omega = 2 * np.pi * f * 1e-3
        k = (1 / c) ** 2
        k0 = (1 / c0) ** 2
        
        Q = 15
        alpha = 1 / Q
        rhot = (1 - alpha / np.pi * np.log(f / 50) - 1j * alpha / 2) ** 2
        
        kr = k * np.real(rhot)
        ki = k * np.imag(rhot)
        k0r = k0 * np.real(rhot)
        k0i = k0 * np.imag(rhot)

        delta_k = k - k0
        a0 = 1.79
        C = a0 * f0 / f
        # y_pde = y.unsqueeze(0).expand(batch_size_v, -1, -1)  # 扩展为 [B_v, B, 2]，适配PDE维度
        # y_pde.requires_grad_(True)  # 仅对扩展后的 y_pde 开启梯度，用于计算PDE残差的导数
        Delta_U = self.forward(vel, y, UU0)
        Delta_U_real = Delta_U[:, :, 0]
        Delta_U_imag = Delta_U[:, :, 1]
        # y_pde = y.unsqueeze(0).expand(batch_size_v, -1, -1)  # 扩展为 [B_v, B, 2]，适配PDE维度
        # y_pde.requires_grad_(True)  # 仅对扩展后的 y_pde 开启梯度，用于计算PDE残差的导数
        zz = y[:, :, 0]
        xx = y[:, :, 1]
        ld = (Z_dim - 70)/2
        # print(ld)
        # ld = 0
        lx = F.relu(((ld - 0.5) * 40 - xx) / ((ld - 0.5) * 40)) + F.relu((xx - (69.5 + ld) * 40) / ((ld - 0.5) * 40))
        lz = F.relu(((ld - 0.5) * 40 - zz) / ((ld - 0.5) * 40)) + F.relu((zz - (69.5 + ld) * 40) / ((ld - 0.5) * 40))
        
        pml_tmp1 = C ** 2 * lx ** 2 * lz ** 2
        pml_tmp2 = C ** 2 * lx ** 4
        pml_tmp3 = C ** 2 * lz ** 4
        pml_tmp4 = C * (lz ** 2 - lx ** 2)
        pml_tmp5 = C * (lx ** 2 + lz ** 2)
        
        Delta_U_grad_real = torch.autograd.grad(
            outputs=Delta_U_real,
            inputs=y,
            grad_outputs=torch.ones_like(Delta_U_real),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        Delta_U_grad_imag = torch.autograd.grad(
            outputs=Delta_U_imag,
            inputs=y,
            grad_outputs=torch.ones_like(Delta_U_imag),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        # print('pml_tmp1',pml_tmp1.shape)
        # print('Delta_U_grad_imag',Delta_U_grad_imag)
        Delta_Uz_real = Delta_U_grad_real[:, :, 0]
        # print('Delta_Uz_real', Delta_Uz_real[:10])
        Delta_Ux_real = Delta_U_grad_real[:, :, 1]
        Delta_Uz_imag = Delta_U_grad_imag[:, :, 0]
        Delta_Ux_imag = Delta_U_grad_imag[:, :, 1]
        # print('pml_tmp1',pml_tmp1.shape)
        # print('Delta_Uz_real',Delta_Uz_real.shape)
        eu_zr = (1+pml_tmp1)/(1+pml_tmp3) * Delta_Uz_real - pml_tmp4/(1+pml_tmp3) * Delta_Uz_imag
        eu_xr = (1+pml_tmp1)/(1+pml_tmp2) * Delta_Ux_real + pml_tmp4/(1+pml_tmp2) * Delta_Ux_imag
        eu_zi = pml_tmp4/(1+pml_tmp3) * Delta_Uz_real + (1+pml_tmp1)/(1+pml_tmp3) * Delta_Uz_imag
        eu_xi = -pml_tmp4/(1+pml_tmp2) * Delta_Ux_real + (1+pml_tmp1)/(1+pml_tmp2) * Delta_Ux_imag
        
        Delta_Uzz_real = torch.autograd.grad(
            outputs=eu_zr,
            inputs=y,
            grad_outputs=torch.ones_like(eu_zr),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0][:, :, 0]
        
        Delta_Uxx_real = torch.autograd.grad(
            outputs=eu_xr,
            inputs=y,
            grad_outputs=torch.ones_like(eu_xr),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0][:, :, 1]
        
        Delta_Uzz_imag = torch.autograd.grad(
            outputs=eu_zi,
            inputs=y,
            grad_outputs=torch.ones_like(eu_zi),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0][:, :, 0]
        Delta_Uxx_imag = torch.autograd.grad(
            outputs=eu_xi,
            inputs=y,
            grad_outputs=torch.ones_like(eu_xi),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0][:, :, 1]
        # print('Delta_U_real',Delta_U_real.shape)
        # print('U0_real', U0_real.shape)
        # print('kr', kr.shape)
        ur_r = (1 - pml_tmp1) * omega ** 2 * (kr * (Delta_U_real + U0_real) - ki * (Delta_U_imag + U0_imag))
        ui_r = pml_tmp5 * omega ** 2 * (kr * (Delta_U_imag + U0_imag) + ki * (Delta_U_real + U0_real))
        u0r_r = (1 - pml_tmp1) * omega ** 2 * (-k0r * U0_real + k0i * U0_imag)
        u0i_r = pml_tmp5 * omega ** 2 * (-k0r * U0_imag - k0i * U0_real)

        ur_i = (-pml_tmp5) * omega ** 2 * ( kr * (Delta_U_real + U0_real) - ki * (Delta_U_imag + U0_imag))
        ui_i = (1 - pml_tmp1) * omega ** 2 * (kr * (Delta_U_imag + U0_imag) + ki * (Delta_U_real + U0_real))
        u0r_i = (-pml_tmp5) * omega ** 2 * (-k0r * U0_real + k0i * U0_imag)
        u0i_i = (1 - pml_tmp1) * omega ** 2 * (-k0r * U0_imag - k0i * U0_real)

        residual_real = Delta_Uzz_real + Delta_Uxx_real + ur_r + ui_r + u0r_r + u0i_r
        residual_imag = Delta_Uzz_imag + Delta_Uxx_imag + ur_i + ui_i + u0r_i + u0i_i
        # residual_real = residual_real_k + self.pde_real_k
        # residual_imag = residual_imag_k + self.pde_imag_k

        # with torch.no_grad():
        #     self.pde_real_k = self.pde_real_k + residual_real_k.detach()
        #     self.pde_imag_k = self.pde_imag_k + residual_imag_k.detach()
        # print(torch.mean(residual_real ** 2 + residual_imag ** 2).shape)
        source_loc = torch.tensor([6, 40], dtype=torch.float32).unsqueeze(0).to(y_norm.device)
        source_loc = (source_loc/Z_dim) * 2 - 1
        weight = get_helmholtz_spatial_weights(y_norm, c, omega, source_loc, alpha=0.1, sigma=0.5)
        return torch.mean(1 * residual_real ** 2 + 1 * residual_imag ** 2)
    
    def loss_Reg(self, vel, y, UU0, source_coord):
        z_coord = y[:, 0]
        x_coord = y[:, 1]
        source_z = source_coord[:, 0]
        source_x = source_coord[:, 1]
        inside_distance = 100 - torch.sqrt((z_coord - source_z) ** 2 + (x_coord - source_x) ** 2)
        coe = F.relu(inside_distance) / (inside_distance + 1e-15)

        pred = self.forward(vel, y, UU0)
        N_reg = torch.count_nonzero(coe)
        N_reg = torch.clamp(N_reg, min=1.0).to(vel.device)

        return torch.sum(coe * pred[:,0] ** 2 + coe * pred[:, 1] ** 2) / N_reg
    def loss_op(self, model0, vel, y, UU0):
        with torch.no_grad():
            pred0 = model0(vel, y, UU0)
        pred_ft = self.forward(vel, y, UU0)
        return torch.sum((pred0-pred_ft) ** 2)
        
    def get_ortho_loss(self, T, weight):
        """
        计算基底正交性损失
        T 形状: [B_v, B_pts, 256] 或 [B_pts, 256]
        """
        
        B_v, N, p = T.shape
        
        # 计算 Gram 矩阵: T^T * T 
        # [B_v, 256, N] @ [B_v, N, 256] -> [B_v, 256, 256]
        gram = torch.bmm(T.transpose(-2, -1), T)
        
        # 关键：归一化 Gram 矩阵
        # 因为 T 是网络输出，其模长可能随训练剧烈变化，归一化能让 Loss 更稳定
        # 我们希望列向量的 self-inner product 趋近于 1
        diag = torch.diagonal(gram, dim1=-2, dim2=-1).unsqueeze(-1) + 1e-8
        gram_normalized = gram / torch.sqrt(diag @ diag.transpose(-2, -1))
        
        gram_matrix = torch.bmm(T.transpose(1, 2), T) / N
        # 构造单位阵
        eye = torch.eye(p, device=T.device).unsqueeze(0).expand(B_v, -1, -1)
        
        # 计算 Frobenius 范数的平方
        loss = torch.mean((gram_matrix - eye) ** 2)
        return loss * weight
    
    def get_trunk_output(self, vel, y):
        """
        专门提取 Trunk Net 的基底输出
        y: [B_pts, 2] 或 [B_v, B_pts, 2]
        """
        nn = vel.shape[-1]
        b1 = vel.shape[0]        # Batch size (B_v)
        b_pts = y.shape[1]      # Query points (B_pts)
        # 2. 坐标编码 (按照你原有的编码逻辑)
        # 假设你的编码逻辑在 self.pos_encoder
        y_norm = 2 * (y - 0) / (40 * 72) - 1 
        z_enc = self.pos_encoder(y_norm[:, :, 0:1])
        x_enc = self.pos_encoder(y_norm[:, :, 1:2])
        y_encoded = torch.cat([z_enc, x_enc], dim=-1)
        physical_context = get_local_physical_features(vel, y, eps=1e-3)

        # print('physical_context', physical_context.shape)# vel_feature = vel_std * 1.5 + vel_mean
        # vel_feature = vel_feature.unsqueeze(1).expand(-1,b_pts,-1)
        y_encoded = torch.cat([y_encoded, physical_context],dim=-1)
        # vel_mean = torch.mean(vel, dim=[2, 3]).view(b1,-1) # [b1, 1, 1, 1]
        # vel_std = torch.std(vel, dim=[2, 3]).view(b1,-1)  # [b1, 1, 1, 1]
        
        # vel_feature = vel_std * 1.5 + vel_mean
        # vel_feature = vel_feature.unsqueeze(1).expand(-1,b_pts,-1)
        # y_encoded = torch.cat([y_encoded, vel_feature],dim=-1)
        # 3. 通过 Trunk Net
        T = self.trunk(y_encoded) # [B_pts, 256]
        return T
    def generate_structure_aware_y_ran(self, vel, num_pts=20000, max_z=72.0, max_x=72.0):
        """
        完全解耦的自由点生成器：仅依赖当前速度场的物理结构，不依赖任何历史 Loss。
        """
        B_v = vel.shape[0]
        device = vel.device
        
        # 1. 快速计算当前速度场的空间梯度幅度
        grad_z = vel[:, :, 2:, 1:-1] - vel[:, :, :-2, 1:-1]
        grad_x = vel[:, :, 1:-1, 2:] - vel[:, :, 1:-1, :-2]
        vel_grad_mag = torch.sqrt(grad_z**2 + grad_x**2 + 1e-8)
        vel_grad_mag = F.pad(vel_grad_mag, (1, 1, 1, 1), mode='replicate').squeeze(1) # [B_v, Z, X]
        
        y_ran_list = []
        for b in range(B_v):
            # 2. 将梯度图展平作为采样概率分布
            prob_dist = vel_grad_mag[b].view(-1)
            prob_dist = prob_dist / (prob_dist.sum() + 1e-8)
            
            # 混合策略：70% 的点集中在速度界面（高梯度区），30% 的点全局均匀分布保底
            num_structure = int(num_pts * 0.7)
            num_uniform = num_pts - num_structure
            
            # --- 抽取结构点 ---
            # 按照梯度概率抽取一维网格索引
            if num_structure > 0:
                sampled_indices = torch.multinomial(prob_dist, num_samples=num_structure, replacement=True)
                # 转换回二维网格坐标
                z_idx = sampled_indices // vel.shape[3]
                x_idx = sampled_indices % vel.shape[3]
                
                # 映射到物理坐标并加上网格内的随机扰动
                dz = max_z / vel.shape[2]
                dx = max_x / vel.shape[3]
                z_coords = z_idx.float() * dz + (torch.rand(num_structure, device=device) * dz)
                x_coords = x_idx.float() * dx + (torch.rand(num_structure, device=device) * dx)
                y_struct = torch.stack([z_coords, x_coords], dim=1)
            else:
                y_struct = torch.empty((0, 2), device=device)
                
            # --- 抽取均匀点 ---
            z_uni = torch.rand(num_uniform, device=device) * max_z
            x_uni = torch.rand(num_uniform, device=device) * max_x
            y_uni = torch.stack([z_uni, x_uni], dim=1)
            
            # 合并当前速度模型的自由点
            y_ran_b = torch.cat([y_struct, y_uni], dim=0)
            y_ran_list.append(y_ran_b)
            
        y_ran = torch.stack(y_ran_list, dim=0) # [B_v, num_pts, 2]
        return y_ran.requires_grad_(True)
    def envelope_barrier_loss(self, vel, y, UU0, u_fno, lambda_env=1.0):
        """
        计算波场包络的流形屏障惩罚损失 (Envelope Barrier Loss)。
        利用 FNO 锁定宏观能量分布，消除高频相位错位带来的局部非凸性。
        
        Args:
            u_pred: DeepONet 预测的波场，形状 [B_v, B_pts, 2] (最后一维为实部和虚部)
            u_fno: FNO 预测的引导波场，形状 [B_v, B_pts, 2]
            r0: 流形容忍半径 (死区阈值)。
                - 只有当包络误差大于 r0 时，才会产生拉回梯度。
                - 在 r0 内部，Loss 为 0，完全交给 PDE 去雕刻高频相位。
                - 如果设为 0，则退化为纯粹的包络 MSE 引导。
            lambda_env: 损失权重系数。
            
        Returns:
            loss_env: 标量 Tensor
        """
        u_pred = self.forward(vel, y, UU0)
        # 1. 提取实部和虚部
        u_pred_real = u_pred[..., 0]
        u_pred_imag = u_pred[..., 1]
        
        u_fno_real = u_fno[..., 0]
        u_fno_imag = u_fno[..., 1]
        
        # 2. 计算波场包络 (振幅)
        # ⚠️ 极度重要：必须加上 1e-8，否则当波场能量为 0 时，sqrt 的导数会变成无穷大 (NaN)
        env_pred = torch.sqrt(u_pred_real**2 + u_pred_imag**2 + 1e-8)
        env_fno = torch.sqrt(u_fno_real**2 + u_fno_imag**2 + 1e-8)
        
        # 3. 计算逐点的包络绝对误差
        # env_error shape: [B_v, B_pts]
        loss_env = torch.abs(env_pred - env_fno)
        
        # 4. 应用流形屏障 (Barrier / Dead-zone)
        # F.relu 会将所有小于 0 的值截断为 0，完美实现流形内部无 FNO 梯度的构想
        
        return torch.mean(loss_env)
        
    def loss(self, vel, y, UU0, labels, a, b, c, data_norm_coe=1., pde_norm_coe=1.):
        
        batch_size_v = vel.shape[0]
        nz = vel.shape[2]
        nx = vel.shape[3]
        
        batch_idx = torch.arange(batch_size_v, device=labels.device)[:, None] # [B_v, 1]
        z_coord = (y[:, :, 0] / 40.0).long().clamp(0, nz - 1) # [B_v, B_pts]
        x_coord = (y[:, :, 1] / 40.0).long().clamp(0, nx - 1) # [B_v, B_pts]
        
        y_ran = self.generate_structure_aware_y_ran(vel, num_pts=900, max_z=nz, max_x=nx).to(self.device)
        # 2. 执行采样
        # 通过 [batch_idx, :, z_coord, x_coord] 采样
        # 注意：PyTorch 高级索引中，非相邻的索引张量（batch_idx 和 z_coord）
        # 会将索引维度放在最前面，中间保留未索引的维度 (Channel=2)
        sampled_labels = labels[batch_idx, :, z_coord, x_coord] 
        
        # 此时 sampled_labels 的形状通常是 [B_v, B_pts, 2]
        # 如果你的 labels 通道维在第 1 维，采样后的形状就是 [B_v, B_pts, 2]
        # 如果采样后维度顺序不对，可以用 .permute() 调整
        labels = sampled_labels # 已经是 [B_v, B_pts, 2]
        # print('labels',labels.shape)
        # -------------------------- 1. 计算基础损失 --------------------------
        # 数据损失（BC损失）
        loss_u = self.loss_BC(vel, y, UU0, labels)/ data_norm_coe
        # error = self.loss_BC(vel, y, UU0, labels)
        # loss_u = self.dynamic_barrier_loss(error, r0=8, lambda_aux=1.0)/ data_norm_coe
        # PDE损失
        # loss_env = self.envelope_barrier_loss(vel, y, UU0, labels)/ data_norm_coe
        
        # loss_f = self.Sobolev_loss(vel, y, UU0)
        loss_f = self.loss_PDE_Scatter_pml(vel, y, UU0)/ pde_norm_coe
        
        loss_f_ran = self.loss_PDE_Scatter_pml(vel, y_ran, UU0)/ pde_norm_coe
        # T_basis = self.get_trunk_output(vel, y)
        # loss_ortho = self.get_ortho_loss(T_basis, 1e2) # 初始权重建议 1e-3
        
        # print('loss_o', loss_ortho)
        # 正则化损失（暂为0）
        loss_r = 0.0
        # precision_data = torch.exp(-self.log_var_data)
        # precision_pde = torch.exp(-self.log_var_pde)
        
        # balanced_loss = (precision_data * loss_u + self.log_var_data) + \
        #                 (precision_pde * loss_f + self.log_var_pde) + loss_ortho
        # 融合动态权重与原有系数a、b（a和b可作为基础比例系数）
        loss_val = 1 * a * (loss_u) + b * 1 * (loss_f + loss_f_ran)

        return loss_val, loss_f + loss_f_ran, loss_u, loss_r

