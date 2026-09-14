# -*- coding: utf-8 -*-
"""
main.py —— 《Hello Agents》3.1.2 节 Transformer 的演示程序

transformer.py 负责"定义模型"，本文件负责"把模型跑起来、把内部数字打出来看"。
四个演示逐层递进，对应教材 3.1.2 的四段正文：

    演示 1  张量形状追踪      → 看懂多头注意力里 (batch, seq, d_model) 怎么变成
                                (batch, heads, seq, d_k) 再变回来
    演示 2  因果掩码实验      → 亲眼看 Softmax 之前把"未来"填成 -1e9 会怎样
    演示 3  位置编码实验      → 验证"同一个词放在不同位置，输入向量就不同"
    演示 4  完整前向传播      → 组装出完整 Transformer，跑一次 logits

运行：python main.py        （需先 pip install -r requirements.txt）
"""

import math

import torch

from transformer import (
    MultiHeadAttention,
    PositionalEncoding,
    Transformer,
    make_causal_mask,
    make_padding_mask,
)


# -----------------------------------------------------------------------------
# 打印辅助：把矩阵画成 ASCII 表格，方便在终端里"看见"注意力权重
# -----------------------------------------------------------------------------

def print_matrix(title, matrix, fmt="{:<8.4f}", row_prefix="pos", col_prefix="pos", legend=None):
    """把二维矩阵按表格打印。

    row_prefix / col_prefix 控制行、列标签的前缀：
      注意力矩阵     → 行=query位置(pos)，列=key位置(pos)
      位置编码矩阵   → 行=位置(pos)，列=特征维度(dim)
      相似度矩阵     → 行/列都是位置(pos)
    """
    m = matrix.detach().cpu()
    rows, cols = m.shape
    print(f"\n{title}   形状 = {tuple(m.shape)}")
    print(" " * 8 + "".join((f"{col_prefix}{c}").ljust(8) for c in range(cols)))
    for r in range(rows):
        cells = "".join(fmt.format(float(m[r, c])) for c in range(cols))
        print(f"  {(row_prefix + str(r)).ljust(6)}" + cells)
    if legend:
        print(" " * 8 + "".join("↑".ljust(8) for _ in range(cols)))
        print(" " * 8 + legend)


# -----------------------------------------------------------------------------
# 演示 1：张量形状追踪 —— 多头注意力是怎么"分身"又"合体"的
# -----------------------------------------------------------------------------

def demo_shapes():
    print("=" * 78)
    print("演示 1 · 多头注意力：张量形状追踪")
    print("=" * 78)
    print("""
教材图 3.5 的关键动作是：把 d_model 维的向量沿维度切成 num_heads 份，
让 h 个"专家"各看一个子空间。下面用 batch=2, seq_len=5, d_model=8, num_heads=2 走一遍。
""")
    torch.manual_seed(0)
    batch, seq_len, d_model, num_heads = 2, 5, 8, 2
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)

    print(f"  输入 x                : {tuple(x.shape)}   (batch, seq_len, d_model)")
    q = mha.W_q(x)
    print(f"  过 Linear 之后 W_q(x) : {tuple(q.shape)}   d_model→d_model，形状不变")
    q_split = mha.split_heads(q)
    print(f"  split_heads 之后      : {tuple(q_split.shape)}   (batch, num_heads, seq_len, d_k)")
    print(f"    ↳ 其中 d_k = d_model / num_heads = {d_model} / {num_heads} = {mha.d_k}")
    combined = mha.combine_heads(q_split)
    print(f"  combine_heads 之后    : {tuple(combined.shape)}   又拼回 (batch, seq_len, d_model)")

    out = mha(x, x, x, mask=None)
    print(f"  forward 输出          : {tuple(out.shape)}   与输入同形，可以继续往上堆")
    print("""
一句话理解：split_heads 把"一条 8 维向量"看成"2 条 4 维向量"，
所以形状多了 num_heads 这一维；combine_heads 再把它拼回去。
transpose(1, 2) 是为了把 num_heads 提到 seq_len 前面，方便后面做批量矩阵乘法。
""")


# -----------------------------------------------------------------------------
# 演示 2：因果掩码 —— 让模型"看不到未来"
# -----------------------------------------------------------------------------

def demo_causal_mask():
    print("=" * 78)
    print("演示 2 · 因果掩码：把'未来'的注意力清零")
    print("=" * 78)
    print("""
教材 3.1.3 节讲的 Masked Self-Attention，核心就三行：
    1) 先算得分 scores = QKᵀ / √d_k
    2) 把"未来位置"的得分填成 -1e9（masked_fill）
    3) 再 Softmax —— 极小值经 Softmax 后趋近于 0
下面用 batch=1, seq_len=5, d_model=8, num_heads=1，把掩码前后的权重矩阵打出来对比。

⚠️ 先明确一件事：这里的 x 是 torch.randn 生成的【随机数】，这 5 个"词"没有任何语义。
   所以下面两张表是"5 个随机向量互相算出来的注意力分布"，不是某句话的理解结果。
   我们要看的是【掩码带来的结构变化】（右上三角变 0），不是这些数字具体多大。
   另外因为模型是随机初始化的、没训练过，权重会挤在 1/5 = 0.2 附近 —— 这是"均匀分布"的特征。
""")
    torch.manual_seed(42)
    seq_len, d_model, num_heads = 5, 8, 1
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(1, seq_len, d_model)

    # 手动复现 scaled_dot_product_attention 的前三步，方便把中间结果打印出来
    Q = mha.split_heads(mha.W_q(x))
    K = mha.split_heads(mha.W_k(x))
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(mha.d_k)

    causal = make_causal_mask(seq_len)          # (1, 1, seq_len, seq_len)，下三角为 True
    probs_before = torch.softmax(scores, dim=-1)
    scores_masked = scores.masked_fill(causal == 0, -1e9)
    probs_after = torch.softmax(scores_masked, dim=-1)

    h0_before = probs_before[0, 0]
    h0_after = probs_after[0, 0]

    print_matrix("① 未加掩码的注意力权重（softmax 之后）", h0_before,
                 legend="每行 = 一个词对全句的注意力分布（行和 = 1）")
    print_matrix("② 加因果掩码后的注意力权重", h0_after,
                 legend="每行 = 一个词对全句的注意力分布（行和 = 1）")

    print(f"""
观察 ②：右上三角全是 0.0000，行和仍然是 1.0 ——
  · 第 0 行只能看自己：      注意力全部集中在 pos0；
  · 第 4 行能看 pos0~pos4：  这正是自回归生成时的信息边界。
再看一个硬证据：把掩码矩阵本身画出来（1=可见，0=屏蔽）。
""")
    grid = "\n".join(
        "        " + "  ".join("1" if causal[0, 0, r, c] else "0" for c in range(seq_len))
        for r in range(seq_len)
    )
    print(grid)
    print("""
第 r 行有 r+1 个 1，越往下看得越多——这就是"下三角"名字的由来。
生成时每步只能依赖已生成的词，掩码把训练与推理的行为对齐了。
""")


# -----------------------------------------------------------------------------
# 演示 3：位置编码 —— 为什么"同一个词在不同位置"会得到不同向量
# -----------------------------------------------------------------------------

def demo_positional_encoding():
    print("=" * 78)
    print("演示 3 · 位置编码：给顺序'盖章'")
    print("=" * 78)
    print("""
注意力机制本身对顺序无感（"agent learns" 与 "learns agent" 等价）。
位置编码用固定的 sin/cos 公式给每个位置配一个独一无二的"坐标"。
""")
    d_model, max_len = 8, 6
    pe_module = PositionalEncoding(d_model=d_model, dropout=0.0, max_len=max_len)

    n_params = sum(p.numel() for p in pe_module.parameters())
    pe = pe_module.pe[0]        # (max_len, d_model)
    print(f"  位置编码矩阵 pe          : {tuple(pe_module.pe.shape)}  (1, max_len, d_model)")
    print(f"  pe 里的可学习参数个数     : {n_params}   ← 0 说明它是公式算出来的常量，不是学出来的")

    print_matrix("前 4 个位置的编码向量（偶数维 sin、奇数维 cos）", pe[:4],
                 fmt="{:<8.3f}", col_prefix="dim", legend="每列 = 位置向量的一个维度")

    print("""
验证"同词不同位 → 不同输入向量"：
两个位置放同一个词（词嵌入完全相同），加上位置编码后，喂给注意力的向量就不再相同。
""")
    # 两个位置放同一个词："零向量"代表完全相同的词嵌入，只有位置不同
    same = torch.zeros(1, 2, d_model)
    enc_out = pe_module(same)                   # 等价于直接取 pe 的前 2 行
    diff = (enc_out[0, 0] - enc_out[0, 1]).abs().mean().item()
    print(f"  位置 0 的向量 : {[round(v, 3) for v in enc_out[0, 0].tolist()]}")
    print(f"  位置 1 的向量 : {[round(v, 3) for v in enc_out[0, 1].tolist()]}")
    print(f"  两者平均绝对差 : {diff:.4f}  ← 非 0，说明位置信息确实被'加'了进去")

    print("""
再补一个很有意思的性质：位置编码携带的是"相对距离"。
把任意两个位置的编码向量做余弦相似度，会发现"隔 k 个位置"的相似度几乎不随起点变化。
""")
    normed = pe / pe.norm(dim=-1, keepdim=True)
    sim = normed @ normed.t()                   # 余弦相似度矩阵
    print_matrix("位置编码的余弦相似度矩阵", sim, fmt="{:<8.3f}",
                 legend="第 r 行 = 位置 r 与各位置的余弦相似度")
    adjacent = [round(float(sim[k, k + 1]), 3) for k in range(sim.shape[0] - 1)]
    print(f"  相邻位置（隔 1 位）的相似度：{adjacent}")
    print(f"  → 无论起点是 0 还是 3，'隔 1 位'的相似度都是 {adjacent[0]}——")
    print("    这正是原论文想要的效果：模型能从编码里直接读出'相对位置'，而不只是绝对坐标。\n")


# -----------------------------------------------------------------------------
# 演示 4：完整前向传播 —— 把拼好的模型跑一遍
# -----------------------------------------------------------------------------

def demo_forward_pass():
    print("=" * 78)
    print("演示 4 · 完整 Transformer 前向传播")
    print("=" * 78)
    print("""
把 Encoder / Decoder 各堆 2 层，装一个小 Transformer 跑一次（超参照比例缩小，纯 CPU 秒级完成）。
场景类比：源句 7 个词 → 目标句 6 个词。
""")
    torch.manual_seed(7)
    src_vocab, tgt_vocab = 50, 50
    batch, src_len, tgt_len = 2, 7, 6

    model = Transformer(
        src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab,
        d_model=32, num_heads=4, num_layers=2, d_ff=64, dropout=0.0,
    )
    model.eval()

    src = torch.randint(1, src_vocab, (batch, src_len))
    tgt = torch.randint(1, tgt_vocab, (batch, tgt_len))
    # 源端只需填充掩码；目标端要"填充掩码 & 因果掩码"叠加
    src_mask = make_padding_mask(src)
    tgt_mask = make_padding_mask(tgt) & make_causal_mask(tgt_len)

    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
        logits = model.forward(src, tgt, src_mask, tgt_mask)

    total = sum(p.numel() for p in model.parameters())
    print(f"  src          : {tuple(src.shape)}")
    print(f"  tgt          : {tuple(tgt.shape)}")
    print(f"  src_mask     : {tuple(src_mask.shape)}   (batch, 1, 1, src_len)")
    print(f"  tgt_mask     : {tuple(tgt_mask.shape)}   (batch, 1, tgt_len, tgt_len)")
    print(f"  编码器输出    : {tuple(encoder_output.shape)}   (batch, src_len, d_model)")
    print(f"  logits       : {tuple(logits.shape)}   (batch, tgt_len, tgt_vocab)")
    print(f"  模型参数量    : {total:,}")

    print("""
解码器输出的是每个位置对全词表的打分（logits）。
把它做 argmax 或 softmax 采样，就得到"下一个词"的预测 —— 这就是大模型生成文字的数学起点。
""")
    next_ids = logits.argmax(dim=-1)
    print(f"  贪心解码得到的 token id： {next_ids[0].tolist()}")

    print("""
说明：由于模型是随机初始化的、也没有经过训练，这里预测的 id 本身没有语义，
      演示的重点是"数据在形状正确的管道里流动"，而不是预测质量。
""")


def main():
    print("\n" + "★" * 78)
    print("  Hello Agents · 3.1.2 Transformer 架构解析 —— 代码演示")
    print("★" * 78)
    demo_shapes()
    demo_causal_mask()
    demo_positional_encoding()
    demo_forward_pass()
    print("=" * 78)
    print("全部演示结束 ✅  完整实现见 transformer.py，逐行讲解见 代码讲解.md")
    print("=" * 78 + "\n")


if __name__ == '__main__':
    main()
