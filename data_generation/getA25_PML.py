"""
25 点广义差分格式 - 频域波动方程求解器
基于 Chen (2011) 的 25 点优化差分格式
严格遵循 MATLAB 原版实现
"""

import numpy as np
from scipy.sparse import spdiags, eye, kron
import math as mm


def getA25_PML(n_pml, nz, nx, freq, modelf0, h, mv):
    """
    25 点有限差分格式 + PML 吸收边界 (2D)
    严格遵循 MATLAB getA25_PML.m 实现

    参数:
        n_pml: PML 层数矩阵 [[顶部, 左侧], [底部, 右侧]]
                     注意: n_pml[0,0]=顶部PML(z方向), n_pml[1,0]=底部PML(z方向)
                           n_pml[0,1]=左侧PML(x方向), n_pml[1,1]=右侧PML(x方向)
        nz, nx: 扩展后的模型尺寸
        freq: 频率 (Hz)
        modelf0: 参考频率
        h: 网格间距 (标量，z和x方向相同)
        mv: 慢度模型 (nz x nx)

    返回:
        A: 系统矩阵 (稀疏矩阵)
        DEL: 拉普拉斯算子 (稀疏矩阵)
        B: 质量矩阵 (稀疏矩阵)
        C: PML 坐标变换矩阵 (稀疏矩阵)
    """

    N = nz * nx
    omega = 1e-3 * 2 * mm.pi * freq

    # 品质因子衰减 (与 getA9_PML 一致)
    Q = 75
    alpha = 1 / Q
    rhot = (1 - alpha / np.pi * np.log(freq / 50) - 1j * alpha / 2) ** 2
    mv = mv * rhot

    # 优化系数 (Chen 2011, eq 3.12)
    gamma1 = 0.2880195
    gamma2 = 0.12362650
    gamma3 = 0.29554904
    gamma4 = 0.014872050
    gamma5 = 0.27793280

    b1 = 0.363276
    b2 = 0.434392
    b3 = 0.0165948
    b4 = 0.1699204
    b5 = 0.000825248
    b6 = 0.0075106
    b7 = 0.00753368

    # PML 常数
    a0_pml = 1.79
    beta = 2 * mm.pi * a0_pml * modelf0 / freq

    # PML 边界厚度 (与 getA9_PML 索引一致)
    top_pml = n_pml[0, 0]     # 顶部 PML (z方向起始) - 自由表面时为0
    bottom_pml = n_pml[1, 0]  # 底部 PML (z方向结束)
    left_pml = n_pml[0, 1]    # 左侧 PML (x方向起始)
    right_pml = n_pml[1, 1]   # 右侧 PML (x方向结束)

    # 归一化因子 (避免除零，与 getA9_PML 一致)
    max_pml_z = max(n_pml[:, 0])
    max_pml_x = max(n_pml[:, 1])

    # 坐标向量
    xc = np.arange(1, nx + 1).reshape(-1, 1)  # 列向量
    zc = np.arange(1, nz + 1).reshape(-1, 1)  # 列向量

    # PML 衰减函数 (与 getA9_PML 一致)
    def pmlz(x):
        """z 方向 PML 衰减"""
        # 顶部 PML: x <= top_pml (自由表面时 top_pml=0，此项为0)
        top_term = 1 - 1j * beta * ((top_pml - x + 0.5) / max(max_pml_z, 1))**2 * (top_pml - x + 0.5 > 0)
        # 底部 PML: x > nz - bottom_pml
        bottom_term = 1 - 1j * beta * ((x - 0.5 - (nz - bottom_pml)) / max(max_pml_z, 1))**2 * (x - 0.5 - (nz - bottom_pml) > 0)
        return top_term + bottom_term - 1  # 减去1因为两项都包含1

    def pmlx(x):
        """x 方向 PML 衰减"""
        # 左侧 PML: x <= left_pml
        left_term = 1 - 1j * beta * ((left_pml - x + 0.5) / max(max_pml_x, 1))**2 * (left_pml - x + 0.5 > 0)
        # 右侧 PML: x > nx - right_pml
        right_term = 1 - 1j * beta * ((x - 0.5 - (nx - right_pml)) / max(max_pml_x, 1))**2 * (x - 0.5 - (nx - right_pml) > 0)
        return left_term + right_term - 1  # 减去1因为两项都包含1

    # 差分算子 (eq 3.1, 3.2)
    def Lh(pmlN, pmlP, n, h):
        """1h 步长差分算子"""
        return spdiags(
            [np.roll(pmlN.flatten(), -1), -pmlN.flatten() - pmlP.flatten(), np.roll(pmlP.flatten(), 1)],
            [-1, 0, 1],
            n, n
        ) / h**2

    def L2h(pmlN, pmlP, n, h):
        """2h 步长差分算子"""
        return spdiags(
            [np.roll(pmlN.flatten(), -2), -pmlN.flatten() - pmlP.flatten(), np.roll(pmlP.flatten(), 2)],
            [-2, 0, 2],
            n, n
        ) / (2 * h)**2

    # 构建差分矩阵
    Dxx = Lh(1 / pmlx(xc - 0.5), 1 / pmlx(xc + 0.5), nx, h)
    Dxx2 = L2h(1 / pmlx(xc - 1), 1 / pmlx(xc + 1), nx, h)
    Dzz = Lh(1 / pmlz(zc - 0.5), 1 / pmlz(zc + 0.5), nz, h)
    Dzz2 = L2h(1 / pmlz(zc - 1), 1 / pmlz(zc + 1), nz, h)

    # 辅助函数: 创建对角矩阵
    def diags_sp(v, k, n):
        """创建稀疏对角矩阵"""
        return spdiags(v.flatten(), k, n, n)

    # DEL 拉普拉斯算子 (after eq 3.2 of the paper)
    DEL = (
        gamma1 * (kron(Dxx, diags_sp(pmlz(zc), 0, nz)) + kron(diags_sp(pmlx(xc), 0, nx), Dzz)) +
        gamma2 / 2 * (kron(Dxx, diags_sp(np.roll(pmlz(zc + 1), 1), 1, nz)) + kron(diags_sp(np.roll(pmlx(xc + 1), 1), 1, nx), Dzz)) +
        gamma2 / 2 * (kron(Dxx, diags_sp(np.roll(pmlz(zc - 1), -1), -1, nz)) + kron(diags_sp(np.roll(pmlx(xc - 1), -1), -1, nx), Dzz)) +
        gamma3 * (kron(Dxx2, diags_sp(pmlz(zc), 0, nz)) + kron(diags_sp(pmlx(xc), 0, nx), Dzz2)) +
        gamma4 / 2 * (kron(Dxx2, diags_sp(np.roll(pmlz(zc + 2), 2), 2, nz)) + kron(diags_sp(np.roll(pmlx(xc + 2), 2), 2, nx), Dzz2)) +
        gamma4 / 2 * (kron(Dxx2, diags_sp(np.roll(pmlz(zc - 2), -2), -2, nz)) + kron(diags_sp(np.roll(pmlx(xc - 2), -2), -2, nx), Dzz2)) +
        gamma5 / 2 * (kron(Dxx2, diags_sp(np.roll(pmlz(zc + 1), 1), 1, nz)) + kron(diags_sp(np.roll(pmlx(xc + 1), 1), 1, nx), Dzz2)) +
        gamma5 / 2 * (kron(Dxx2, diags_sp(np.roll(pmlz(zc - 1), -1), -1, nz)) + kron(diags_sp(np.roll(pmlx(xc - 1), -1), -1, nx), Dzz2))
    )

    # PML 坐标变换矩阵 C
    gridZ, gridX = np.meshgrid(zc.flatten(), xc.flatten(), indexing='ij')
    C = spdiags((pmlz(gridZ) * pmlx(gridX)).flatten(), 0, N, N)

    # 位移矩阵 D (antilumped mass strategy, eq 2.3 of the paper)
    ex = np.ones((nx, 1))
    ez = np.ones((nz, 1))

    def D(xs, zs):
        """创建位移矩阵"""
        return kron(spdiags(ex.flatten(), xs, nx, nx), spdiags(ez.flatten(), zs, nz, nz))

    Ih0 = 0.25 * (D(-1, 0) + D(1, 0) + D(0, -1) + D(0, 1))
    I2h0 = 0.25 * (D(-2, 0) + D(0, -2) + D(2, 0) + D(0, 2))
    Ih45 = 0.25 * (D(-1, 1) + D(1, -1) + D(-1, -1) + D(1, 1))
    I2h45 = 0.25 * (D(-2, -2) + D(2, -2) + D(2, 2) + D(-2, 2))
    I2ht1 = 0.25 * (D(2, -1) + D(1, 2) + D(-2, 1) + D(-1, -2))
    I2ht2 = 0.25 * (D(1, -2) + D(2, 1) + D(-1, 2) + D(-2, -1))
    # B 质量矩阵
    B = b1 * eye(N, N, format='csr') + b2 * Ih0 + b3 * I2h0 + b4 * Ih45 + b5 * I2h45 + b6 * I2ht1 + b7 * I2ht2
    # 系统矩阵 A
    A = DEL + omega**2 * C @ spdiags(mv.flatten(), 0, N, N) @ B
    return A, DEL, B, C


if __name__ == "__main__":
    print("25 点差分格式测试")

    # 测试参数 (与 modeling.py 调用方式一致)
    n_pml = np.array([[10, 10], [10, 10]])
    nz, nx = 90, 90
    h = 20
    modelf0 = 10
    mv = np.ones((nz, nx)) * 0.5

    # 测试
    A, DEL, B, C = getA25_PML(n_pml, nz, nx, 15, modelf0, h, mv)
    print(f"A shape: {A.shape}, nnz: {A.nnz}")
    print(f"DEL shape: {DEL.shape}, nnz: {DEL.nnz}")
    print(f"B shape: {B.shape}, nnz: {B.nnz}")
    print(f"C shape: {C.shape}, nnz: {C.nnz}")
