from Labconfig import *
from model.utils import *

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
class Sin(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        embed = []
        for i in range(self.embed_dim):
            embed.append(torch.sin(2**(i - self.embed_dim + 1) * 6 * np.pi * x))
            embed.append(torch.cos(2**(i - self.embed_dim + 1) * 6 * np.pi * x))
        return torch.cat(embed, dim=-1)
        
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            Sin(), # 使用 GELU 替代 Sin 以获得更平滑的非线性
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.act = Sin()

    def forward(self, x):
        return self.act(x + self.net(x))
        
class AttenGate(nn.Module):
    def __init__(self, feat_dim=None, use_softmax=True):
        """
        平衡两个分支网络和一个主干网络的输出
        
        Args:
            feat_dim: 特征维度（若输入为高维特征图，可忽略，默认None）
            use_softmax: 是否用softmax归一化权重（否则用sigmoid+归一化）
        """
        super().__init__()
        self.use_softmax = use_softmax
        
        # 为三个网络（branch1, branch2, trunk）分配可学习权重
        # 若输入为特征图（含空间维度），权重为标量；若为向量，可适配维度
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # 初始权重均等

    def forward(self, branch1_out, branch2_out):
        """
        融合三个网络的输出
        
        Args:
            branch1_out: 分支网络1的输出（形状如 [B, C, H, W] 或 [B, C]）
            branch2_out: 分支网络2的输出（同上，需与branch1_out形状一致）
            trunk_out: 主干网络的输出（同上，需与前两者形状一致）
        
        Returns:
            fused_out: 融合后的输出（与输入形状一致）
        """
        # 1. 权重归一化
        if self.use_softmax:
            # softmax归一化：权重之和为1，强调相对重要性
            w1, w2, w3 = F.softmax(self.weights, dim=0)
        else:
            # sigmoid + 归一化：允许权重灵活调整，避免一方被完全压制
            w1, w2, w3 = F.sigmoid(self.weights)
            # w1, w2, w3 = w1 / (w1 + w2 + w3), w2 / (w1 + w2 + w3), w3 / (w1 + w2 + w3)
        
        # 2. 加权融合（广播机制适配高维输入）
        fused_out = w1 * branch1_out + w2 * branch2_out + w3 * branch2_out * branch1_out
        
        return fused_out

class PositionalEembedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 生成位置编码（不直接注册为缓冲区，而是作为参数）
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # 将位置编码设为参数（而非缓冲区），方便后续移动设备
        self.pe = nn.Parameter(pe, requires_grad=False)  # 仍不参与训练

    def forward(self, x):
        # 关键：将位置编码移至与输入x相同的设备
        x = x + self.pe[:x.size(0)].to(x.device)  # 确保pe与x在同一设备
        return x

class GaussianWeightedLayer(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.compress= nn.Sequential(
            nn.Linear((feat_dim + 0) * 16, feat_dim * 8),
            Sin(),
            nn.Linear(feat_dim * 8, feat_dim * 4),
            Sin(),
            nn.Linear(feat_dim * 4, feat_dim),
        )
    def forward(self, speed_maps, centers, B_out, cre=True):
        b1, feat_dim, H, W = speed_maps.shape  # (b1, feat_dim, H, W    
        b2, coord_dim = centers.shape   # (b2, 2)
        
        batch_size, feat_dim, H, W = B_out.shape
        patch_size = (H//4, W//4)
        patches = F.avg_pool2d(B_out, kernel_size=patch_size, stride=patch_size)
        B_global = patches.reshape(B_out.shape[0], -1)
        B_global = self.compress(B_global) 
        
        # B_global = self.cross_attention(B_out).squeeze(-1).squeeze(-1)
        # print('B_global', B_global.shpe)
        
        # -------------------------- 1. 全局平均池化（简化维度操作）--------------------------
        # global_avg = self.global_avg_pool(speed_maps)  # (b1, feat_dim, 1, 1)
        # global_avg = global_avg.squeeze(-1).squeeze(-1)  # (b1, feat_dim) → 直接挤压，无需中间unsqueeze
        # -------------------------- 2. 向量化坐标映射（避免循环，直接计算索引）--------------------------
        # 假设centers是归一化坐标（如[-1,1]），映射到特征图尺寸[0, H-1]和[0, W-1]
        # 若原代码的"除以40"是特定映射，可替换为实际坐标转换逻辑（此处保留原意并优化）
        z_indices = (centers[..., 0] / 40).long().clamp(0, H - 1).detach()
        x_indices = (centers[..., 1] / 40).long().clamp(0, W - 1).detach()
        
        # 2. 构造 Batch 索引
        # shape: (b1, 1)
        b1_idx = torch.arange(b1, device=speed_maps.device).view(-1, 1)
        
        # 3. 高效索引 (关键修改)
        # PyTorch 规则：当高级索引 (b1_idx, z_indices, x_indices) 被切片 (:) 隔开时，
        # 被索引的维度会移动到最前面。
        # speed_maps shape: [B1, C, H, W]
        # b1_idx (dim 0) 和 z/x_indices (dim 2,3) 广播后的 shape 是 [B1, B2]
        # 因此结果 shape 会变成: [B1, B2, C] (索引维度在前，特征维度 C 被挤到最后)
        
        # (B1, C, H, W) -> [索引操作] -> (B1, B2, C)
        extracted_feat = speed_maps[b1_idx, :, z_indices, x_indices]

        # -------------------------- 4. 合并结果（利用广播避免显式expand）--------------------------
        # global_avg.unsqueeze(1) → (b1, 1, feat_dim)，与center_values(b1, b2, feat_dim)自动广播
        # print('global_avg', global_avg.shape)
        # print('center_values', center_values.shape)
            
        combined = B_global.unsqueeze(1) + extracted_feat
        # print('com', combined.shape)
        if not cre:
            combined = combined - extracted_feat
        
        return combined

class SpectralConv2d(nn.Module):
    """2D傅里叶卷积层（适配DeepONet的通道/尺寸）"""
    def __init__(self, in_channels, out_channels, modes1=12, modes2=12):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # x轴傅里叶模式数（可调，建议8-16）
        self.modes2 = modes2  # y轴傅里叶模式数

        # 权重初始化缩放因子（防止梯度爆炸）
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        """复数矩阵乘法（傅里叶域卷积）"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        # 1. 实数域→复数域（傅里叶变换）
        x_ft = torch.fft.rfft2(x)

        # 2. 傅里叶域卷积（仅保留前modes个模式）
        out_ft = torch.zeros(batchsize, self.out_channels, size_x, size_y//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 3. 复数域→实数域（逆傅里叶变换）
        x = torch.fft.irfft2(out_ft, s=(size_x, size_y))
        return x

def QR_orthogonalization(T_raw):
    """
    针对输入形状 [B_v, B_pts, 256] 设计的 QR-DeepONet 变换
    T_raw: Trunk Net 的原始输出
    """
    B_v, N, p = T_raw.shape  # B_v=Batch Size, N=采样点数, p=256(基底数)

    # 1. 克隆以保护原始计算图，避免原地修改错误
    T_prime = T_raw.clone()

    # 2. 三角化操作 (Triangular Operation)
    # 我们只对每个 Batch 的前 p 行（即前 256 行）进行处理
    # 目标：构造一个 p x p 的单位下三角矩阵
    
    # 提取前 p 行为操作对象
    T_sub = T_prime[:, :p, :p]
    
    # 将上三角部分（不含对角线）置零
    # torch.tril 保留下三角，diagonal=-1 表示不包含对角线
    T_sub = torch.tril(T_sub, diagonal=-1)
    
    # 将对角线元素强制设为 1
    # 构造一个形状为 [p, p] 的单位阵，并扩展到 [B_v, p, p]
    eye = torch.eye(p, device=T_raw.device).unsqueeze(0).expand(B_v, -1, -1)
    T_sub = T_sub + eye
    
    # 写回原矩阵
    T_prime[:, :p, :p] = T_sub

    # 3. 执行批处理 QR 分解
    # PyTorch 的 linalg.qr 支持 [..., N, p] 形状，会在最后两个维度上操作
    # Q 形状: [B_v, B_pts, 256]
    Q, _ = torch.linalg.qr(T_prime)
    
    return Q

class FNO2d(nn.Module):
    """2D傅里叶神经算子完整模型"""
    def __init__(self, in_channels, out_channels, modes1=10, modes2=10, width=32):
        super(FNO2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width  # 隐藏层通道数

        # 1. 输入投影（将输入映射到高维特征空间）
        self.fc0 = nn.Linear(in_channels, self.width)

        # 2. 傅里叶卷积层（核心特征提取）
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # 3. 残差连接的线性层
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)

        # 4. 输出投影（映射到输出维度）
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
        # 激活函数
        self.activation = nn.GELU()

    def forward(self, x):
        # x shape: [batch, in_channels, H, W]
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        # 1. 输入投影：[B, C, H, W] → [B, H, W, C] → [B, H, W, width] → [B, width, H, W]
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # 2. 傅里叶卷积 + 残差连接（4层）
        x1 = self.conv1(x)
        x1 = self.w1(x1)
        x = x + self.activation(x1)  # 残差

        x2 = self.conv2(x)
        x2 = self.w2(x2)
        x = x + self.activation(x2)

        x3 = self.conv3(x)
        x3 = self.w3(x3)
        x = x + self.activation(x3)
        
        x4 = self.conv4(x)
        x4 = self.w4(x4)
        x = x + self.activation(x4)

        # 3. 输出投影：[B, width, H, W] → [B, H, W, width] → [B, H, W, out_channels] → [B, out_channels, H, W]
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = self.activation(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        return x

class FourierFeatureEncoder(nn.Module):
    """
    傅里叶特征编码器 (位置编码)
    将原始坐标 y 映射到高维空间，以帮助网络学习高频信息。
    """
    def __init__(self, input_dim: int, mapping_size: int, scale: float = 1.0):
        super().__init__()
        
        # B 矩阵：[input_dim, mapping_size] (例如 [2, 256])
        self.B = nn.Parameter(
            scale * torch.randn(input_dim, mapping_size),
            requires_grad=False # B 矩阵通常是固定的，不参与训练
        )
        self.output_dim = mapping_size * 2
        
    def forward(self, x):
        """
        x: 坐标输入，形状 [B_v, B_pts, input_dim] 或 [B_pts, input_dim]
        """
        # 1. 矩阵乘法: (y @ B) -> [..., mapping_size]
        x_B = x @ self.B 
        
        # 2. 傅里叶变换: [sin(y @ B), cos(y @ B)]
        return torch.cat([torch.sin(x_B), torch.cos(x_B)], dim=-1) 

class StandardCrossAttention(nn.Module):
    """
    功能完善且支持二阶导数的 Cross Attention 模块。
    用于将 Trunk (Query) 和 Branch (Key/Value) 特征进行融合。
    """
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # --- 1. Cross Attention 模块 ---
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout, # MHA 内部的 Dropout
            batch_first=False # 保持 (Seq_len, Batch, Feature) 格式
        )
        
        # --- 2. 前馈网络 (Feed-Forward Network, FFN) ---
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # --- 3. 归一化和 Dropout ---
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout) # 在残差连接中的 Dropout
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_ffn = nn.Dropout(dropout) # 在 FFN 残差连接中的 Dropout
        
        # 激活函数
        self.activation = nn.GELU() 

    def forward(self, query, memory):
        """
        :param query: Trunk 特征 (Q)，形状 [B_pts, B_v, feat_dim]
        :param memory: Branch 特征 (K, V)，形状 [S, B_v, feat_dim] (S 是 Branch Token 长度)
        """
        # 1. Attention
        attn_output, _ = self.attn(
            query=query, 
            key=memory, 
            value=memory,
            attn_mask=None, 
            key_padding_mask=None 
        )
        
        # 2. Add & Norm
        output_attn = query + self.dropout_attn(attn_output)
        output_attn = self.norm1(output_attn)
        
        # 3. FFN
        ffn_output = self.linear2(self.dropout1(self.activation(self.linear1(output_attn))))
        
        # 4. Add & Norm
        output_ffn = output_attn + self.dropout_ffn(ffn_output)
        output_ffn = self.norm2(output_ffn)

        return output_ffn

class StandardEncoderLayer(nn.Module):
    """
    支持二阶导数计算的标准 Transformer Encoder Layer。
    包含 Multihead Self-Attention (MSA) 和 Feed-Forward Network (FFN)。
    适用于需要计算高阶导数的 PINN/DeepONet Token序列处理。
    """
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # --- 1. Multihead Self-Attention (MSA) ---
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=False 
        )
        
        # --- 2. 前馈网络 (FFN) ---
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # --- 3. 归一化和 Dropout ---
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout) 
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_ffn = nn.Dropout(dropout) 
        
        # 激活函数
        self.activation = nn.GELU() 

    def forward(self, src: torch.Tensor):
        # Self-Attention
        attn_output, _ = self.self_attn(
            query=src, 
            key=src, 
            value=src,
            attn_mask=None,
            key_padding_mask=None
        )
        
        # Add & Norm 
        output_attn = src + self.dropout_attn(attn_output)
        output_attn = self.norm1(output_attn) 
        
        # FFN
        ffn_output = self.linear2(self.dropout1(self.activation(self.linear1(output_attn))))
        
        # Add & Norm 
        output_ffn = output_attn + self.dropout_ffn(ffn_output)
        output_ffn = self.norm2(output_ffn) 

        return output_ffn

class Tokenizer(nn.Module):
    def __init__(self, channels, target_size=8):
        super().__init__()
        # 采用卷积降采样来保留局部信息，而不是简单的插值
        self.downsample = nn.Conv2d(channels, channels, kernel_size=3, stride=int(70/target_size), padding=1, groups=channels)
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.target_size = target_size

    def forward(self, x):
        # x 形状: [B, C, H, W]
        x = self.downsample(x) # [B, C, target_size, target_size]
        x = self.conv1x1(x)    # [B, C, target_size, target_size]
        
        # 序列化: (C, H, W) -> (H*W, C)
        tokens = x.flatten(start_dim=2).permute(0, 2, 1) # [B, target_size^2, C]
        return tokens

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, conditioning_dim):
        super().__init__()
        self.modulate = nn.Linear(conditioning_dim, input_dim * 2)

    def forward(self, x, condition):
        params = self.modulate(condition) # [B, 1, input_dim * 2]
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return (1 + gamma) * x + beta

class FiLMTrunk(nn.Module):
    def __init__(self,input_dim=16, width=128, branch_feat_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.film1 = FiLMLayer(256, branch_feat_dim)
        
        self.fc2 = nn.Linear(256, 256)
        self.film2 = FiLMLayer(256, branch_feat_dim)
        
        self.fc3 = nn.Linear(256, 256)
        self.film3 = FiLMLayer(256, branch_feat_dim)
        
        self.fc4 = nn.Linear(256, width)
        self.tanh = Sin()

    def forward(self, y_trunk, branch_condition):
        x = self.tanh(self.fc1(y_trunk))
        x = self.film1(x, branch_condition)
        
        x = self.tanh(self.fc2(x))
        x = self.film2(x, branch_condition)

        x = self.tanh(self.fc3(x))
        x = self.film3(x, branch_condition)
        
        x = self.fc4(x)
        return x # 输出 128 维基底

class BlockFeatureEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        
        self.block_mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, out_channels),
            nn.LayerNorm(out_channels),
        )

    def forward(self, feat_field, y_norm):
        B, C, w, _ = feat_field.shape
        N_pts = y_norm.shape[1]
                
        block_feats = self.pool(feat_field) 
        block_feats_flat = block_feats.view(B, C, -1).permute(0, 2, 1)
        
        with torch.no_grad():
            margin = 1.0 - (1.0 / w) 
            y_rescaled = y_norm.detach() / margin
            eps = 1e-6
            y_shifted = (y_rescaled + 1.0) / 2.0
            
            grid_coords = torch.floor(y_shifted * (self.grid_size - eps)).long()
            grid_coords = torch.clamp(grid_coords, 0, self.grid_size - 1)
            idx = grid_coords[:, :, 1] * self.grid_size + grid_coords[:, :, 0]

        batch_idx = torch.arange(B, device=y_norm.device).view(B, 1).expand(-1, N_pts)
        selected_feats = block_feats_flat[batch_idx, idx] # [B, N_pts, C]
        
        condition = self.block_mlp(selected_feats)
        return condition

class SmoothBlockEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        
        self.block_mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, out_channels),
            nn.LayerNorm(out_channels),
        )

    def forward(self, feat_field, y_norm):
        B, C, _, _ = feat_field.shape
        
        low_res_feat = self.pool(feat_field).permute(0,1,3,2)
        grid = y_norm.detach().unsqueeze(2) # [B, N, 1, 2]
        
        smooth_feat = F.grid_sample(
            low_res_feat, 
            grid, 
            mode='bilinear',
            padding_mode='border', 
            align_corners=True
        )
        
        smooth_feat = smooth_feat.squeeze(-1).permute(0, 2, 1)
        return self.block_mlp(smooth_feat)