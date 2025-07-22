from torch.nn.modules import padding
import triton
import triton.language as tl
import torch

class PytorchFLA2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 兼容3D和4D张量
        Nq, d = Q.shape[-2:]
        Nk = K.shape[-2]
        Bq, Bk = 16, 16
        Tq = triton.cdiv(Nq, Bq)
        Tk = triton.cdiv(Nk, Bk)
        dims = Q.shape[:-2]

        O = torch.zeros_like(Q)
        L = torch.zeros(dims + (Nq,), device=Q.device)

        for i in range(Tq):
            qs = i * Bq
            qe = min((i + 1) * Bq, Nq)
            Qi = Q[:, qs:qe, :]  # [B, Bq, d]
            Oi = torch.zeros(dims + (qe - qs, d), device=Q.device)
            mi = torch.full(dims + (qe - qs,), float('-inf'), device=Q.device)
            li = torch.zeros(dims + (qe - qs,), device=Q.device)
            for j in range(Tk):
                ks = j * Bk
                ke = min((j + 1) * Bk, Nk)
                Kj = K[..., ks:ke, :]  # [B, Bk, d]
                Vj = V[..., ks:ke, :]  # [B, Bk, d]
                # 计算注意力分数
                Sij = torch.einsum('...id,...jd->...ij', Qi, Kj) / (d ** 0.5)  # [B, Bq, Bk]
                if is_causal:
                    mask = torch.triu(torch.ones(Sij.shape[-2], Sij.shape[-1], device=Sij.device, dtype=torch.bool), diagonal=1)
                    Sij = Sij.masked_fill(mask, -1e6)
                if is_causal:
                    print("Not implement causal version!")
                mj = Sij.max(dim=2).values  # [B, Bq]
                m_new = torch.maximum(mi, mj)
                exp_mi_mnew = torch.exp(mi - m_new)
                exp_Sij_mnew = torch.exp(Sij - m_new[:, None])
                pij = exp_Sij_mnew  # [B, Bq, Bk]
                lij = pij.sum(dim=2)  # [B, Bq]
                # 累加
                Oi = Oi * exp_mi_mnew.unsqueeze(-1) + torch.einsum('...ij,...jd->...id', pij, Vj)
                li = li * exp_mi_mnew + lij
                mi = m_new
            O[..., qs:qe, :] = Oi / li.unsqueeze(-1)
            L[..., qs:qe] = mi + torch.log(li)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O


    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError
class TritonFLA2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Nq, d = Q.shape[-2:]
        Nk = K.shape[-2]
        Bq, Bk = 16, 16
        ctx.Bq, ctx.Bk = Bq, Bk
        ctx.is_causal = is_causal
        Tq = triton.cdiv(Nq, Bq)
        B = Q.shape[0]

        scale = 1 / (d ** 0.5)
        O = torch.zeros_like(Q)
        L = torch.zeros(B, Nq, device=Q.device)

        # 不能用变量传递 constexpr，必须用常量
        if is_causal:
            flash_fwd_kernel[(Tq, B,)](
                Q, K, V,
                O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                Nq, Nk,
                scale,
                D=d,
                Q_TILE_SIZE=16,
                K_TILE_SIZE=16,
                is_causal=True,
            )
        else:
            flash_fwd_kernel[(Tq, B,)](
                Q, K, V,
                O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                Nq, Nk,
                scale,
                D=d,
                Q_TILE_SIZE=16,
                K_TILE_SIZE=16,
                is_causal=False,
            )

        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 程序索引
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 创建块指针
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    O_block = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    for i in range(Tk):
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Sij = tl.dot(Q_block, tl.trans(K_block)) * scale
        # 实现 causal mask
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] < k_idx[None, :]
            Sij = tl.where(mask, Sij - 1e6, Sij)
        mj = tl.max(Sij, axis=1)
        m_new = tl.maximum(mi, mj)
        exp_mi_mnew = tl.exp(mi - m_new)
        exp_Sij_mnew = tl.exp(Sij - m_new[:, None])
        pij = exp_Sij_mnew
        lij = tl.sum(pij, axis=1)
        # O_block严格按论文公式归并
        O_block = O_block * exp_mi_mnew[:, None] + tl.dot(pij, V_block)
        li = li * exp_mi_mnew + lij
        mi = m_new
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # 归一化输出
    tl.store(O_block_ptr, O_block / li[:, None], boundary_check=(0, 1))
    tl.store(L_block_ptr, mi + tl.log(li), boundary_check=(0,))



