# -*- coding: utf-8 -*-
"""
transformer.py —— 《Hello Agents》3.1.2 节 Transformer 架构的完整可运行实现

教材采用"自顶向下"的写法：先搭好 Encoder / Decoder 的骨架，再把 MultiHeadAttention、
PositionWiseFeedForward、PositionalEncoding 三个模块像拼图一样逐一填进去。

本文件在忠实还原教材代码的基础上，补全了教材为了控制篇幅而没有给出的部分：
  * Encoder / Decoder 的 N 层堆叠（教材只给了单个 EncoderLayer / DecoderLayer）
  * 完整的 Transformer 主类（词嵌入 + 位置编码 + 堆叠 + 输出线性层）
  * 两把"钥匙"——填充掩码（padding mask）与因果掩码（causal mask）的生成函数
    （3.1.3 节讲 Decoder-Only 时才会展开，但 Encoder-Decoder 要想真正跑起来必须先有它）

代码顺序与教材 3.1.2 的小节编号一一对应，方便对照阅读：

    (1) Encoder-Decoder 整体结构   → EncoderLayer / DecoderLayer
    (2) 从自注意力到多头注意力     → MultiHeadAttention
    (3) 前馈神经网络               → PositionWiseFeedForward
    (4) 残差连接与层归一化         → 已内联在 EncoderLayer / DecoderLayer 中
    3.1.2.5 位置编码               → PositionalEncoding
    补全部分                      → Encoder / Decoder / Transformer / 掩码函数

运行环境：Python 3.8+，PyTorch 1.13+（无需 GPU、无需联网）。
"""

import math

import torch
import torch.nn as nn


# =============================================================================
# (1) Encoder-Decoder 整体结构
# -----------------------------------------------------------------------------
# 教材图 3.4：左侧是 N 层编码器堆叠，右侧是 N 层解码器堆叠。
# 下面先给出两种"层"（Layer），它们是堆叠的最小重复单元。
# =============================================================================

class EncoderLayer(nn.Module):
    """
    编码器核心层（教材原样代码）。

    一层 = 多头自注意力 + 逐位置前馈网络，每个子模块都被 Add & Norm 包裹。
    注意：这里的 `x + self.dropout(...)` 就是"残差连接(Add)"，
    `self.norm1(...)` 就是"层归一化(Norm)"，两者合称 Add & Norm。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)      # 教材为占位符，此处已实现
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)  # 同上
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. 多头自注意力：Q、K、V 全部来自同一个 x（这就是"自"注意力）
        attn_output = self.self_attn(x, x, x, mask)
        # Add & Norm：残差 + Dropout + 层归一化
        x = self.norm1(x + self.dropout(attn_output))
        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        # 再次 Add & Norm
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    """
    解码器核心层（教材原样代码）。

    比编码器层多了一个"交叉注意力"子层：Q 来自解码器自己，K / V 来自编码器输出。
    这就是解码器能"咨询"编码器理解结果的地方。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)       # 带因果掩码的自注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads)      # 交叉注意力
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. 掩码多头自注意力（对自己）：tgt_mask 负责挡住"未来的词"
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 2. 交叉注意力（对编码器输出）：Q=x(解码器)，K=V=encoder_output(编码器)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# =============================================================================
# (2) 从自注意力到多头注意力
# -----------------------------------------------------------------------------
# 教材图 3.5：Q / K / V 各自过一层 Linear，再在维度上切成 h 份并行做
# 缩放点积注意力，最后 Concat 起来过一层 Linear 输出。
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块（教材原样代码）。

    用一个类比理解 Q / K / V：
      把注意力想成一次"开卷考试"——
        Q（Query 查询）= 你手里的问题；
        K（Key   键）  = 所有参考资料的标签/索引；
        V（Value 值）  = 参考资料里真正的内容。
      先用 Q 和每份资料的 K 做点积算"相关性得分"，
      再用 Softmax 把得分变成权重（总和为 1），最后按权重把 V 加权求和。
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # 每个头的维度
        # 定义 Q, K, V 和输出的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力：Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V"""
        # 1. 计算注意力得分 (QK^T)，除以 √d_k 做缩放
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 2. 应用掩码（如果提供）
        if mask is not None:
            # 将掩码中为 0 的位置设置为一个非常小的负数，这样 softmax 后会接近 0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        # 3. 计算注意力权重 (Softmax)，dim=-1 表示在"被关注的序列"这一维归一化
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # 4. 加权求和（权重 × V）
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # 将输入 x 的形状从 (batch_size, seq_length, d_model)
        # 变换为 (batch_size, num_heads, seq_length, d_k)
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # 将输入 x 的形状从 (batch_size, num_heads, seq_length, d_k)
        # 变回 (batch_size, seq_length, d_model)
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 1. 对 Q, K, V 进行线性变换，并把 head 维切出来
        #    一个 (batch, seq, d_model) 的矩阵被"掰"成了 num_heads 个 (batch, seq, d_k)
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        # 2. 计算缩放点积注意力（此时是 num_heads 个"小注意力"并行计算）
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # 3. 合并多头输出并进行最终的线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output


# =============================================================================
# (3) 前馈神经网络
# -----------------------------------------------------------------------------
# 逐位置（Position-wise）FFN：对序列中每个位置独立地做一次
# Linear(d_model→d_ff) → ReLU → Dropout → Linear(d_ff→d_model)。
# "先扩大再缩小"（通常 d_ff = 4 × d_model）被认为能学到更丰富的特征。
# =============================================================================

class PositionWiseFeedForward(nn.Module):
    """位置前馈网络模块（教材原样代码）。"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形状: (batch_size, seq_len, d_model)
        x = self.linear1(x)      # 升维 → (batch_size, seq_len, d_ff)
        x = self.relu(x)         # 非线性激活
        x = self.dropout(x)
        x = self.linear2(x)      # 降回 → (batch_size, seq_len, d_model)
        # 最终输出形状: (batch_size, seq_len, d_model)
        return x


# =============================================================================
# (4) 残差连接与层归一化
# -----------------------------------------------------------------------------
# 教材正文解释：Add 解决梯度消失，Norm 稳定每层输入分布、加速收敛。
# 这段逻辑没有独立成类，而是内联在 EncoderLayer / DecoderLayer 里：
#     x = self.norm1(x + self.dropout(attn_output))
# 上面这行同时包含了残差连接（x + ...）与层归一化（self.norm1）。
# =============================================================================


# =============================================================================
# 3.1.2.5 位置编码
# -----------------------------------------------------------------------------
# 注意力本身不区分词的先后顺序（"agent learns" 与 "learns agent" 对它等价），
# 因此要给每个词嵌入额外加上一个由 sin / cos 公式算出的、固定的位置向量。
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    为输入序列的词嵌入向量添加位置编码（教材原样代码）。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    偶数维度用 sin，奇数维度用 cos。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)                       # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe (positional encoding) 的大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 偶数维度使用 sin, 奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将 pe 注册为 buffer，这样它就不会被视为模型参数，但会随模型移动（例如 to(device)）
        self.register_buffer('pe', pe.unsqueeze(0))                         # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size(1) 是当前输入的序列长度
        # 将位置编码加到输入向量上（加法广播：实际只取前 seq_len 个位置）
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =============================================================================
# 以下为教材未展开、本项目补全的部分，目的是让整套代码"可以真正跑起来"。
# =============================================================================

class Encoder(nn.Module):
    """把 N 个 EncoderLayer 串联起来 —— 对应图 3.4 左侧的 "N×" 框。"""

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)   # 堆叠结束后的收尾归一化（原论文/post-LN 常见做法）

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """把 N 个 DecoderLayer 串联起来 —— 对应图 3.4 右侧的 "N×" 框。"""

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


def make_padding_mask(seq, pad_idx=0):
    """
    填充掩码：把 <pad> 占位符挡住，不让模型去关注无意义的填充位。

    seq 形状 (batch, seq_len)，返回 (batch, 1, 1, seq_len) 的布尔张量，
    True 表示"保留"（非 pad），False 表示"屏蔽"（pad）。
    中间两个 1 是为了对 num_heads 与 query 位置这两维做广播。
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(seq_len, device=None):
    """
    因果掩码（下三角掩码）：第 t 个位置只能看到 ≤ t 的位置，不能"偷看"未来。

    返回 (1, 1, seq_len, seq_len) 的布尔张量，下三角（含对角线）为 True。
    教材 3.1.3 节讲的 Masked Self-Attention 用的就是它。
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


class Transformer(nn.Module):
    """
    完整 Transformer（教材未给出，本项目补全）。

    数据流：源码 token ──Embedding──▶ ×√d_model ──+位置编码──▶ Encoder(N层)
            目标 token ──Embedding──▶ ×√d_model ──+位置编码──▶ Decoder(N层) ──▶ Linear ──▶ logits
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        # 两张词嵌入表：源语言、目标语言各一张（词表大小可以不同）
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # 位置编码模块（源端与目标端共用，因为公式与词表无关）
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        # 输出层：把 d_model 维的隐藏状态映射回目标词表大小的打分（logits）
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        """只跑编码器，方便复用（例如把编码结果缓存下来）。"""
        # 词嵌入乘以 √d_model：让嵌入与位置编码的数值量级相近（原论文做法）
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        """只跑解码器。"""
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src:     (batch, src_len) 源序列 token id
        tgt:     (batch, tgt_len) 目标序列 token id（训练时是"右移一位"的答案）
        返回:    (batch, tgt_len, tgt_vocab_size) 的 logits
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.fc_out(decoder_output)


def build_transformer(src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                      num_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
    """按"原论文超参"装配一个 Transformer（d_model=512, heads=8, layers=6, d_ff=2048）。

    直接调用 Transformer(...) 即可；这里额外提供一个便捷函数，
    方便演示脚本用"缩小版超参"实例化一个小模型。
    """
    return Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads,
                       num_layers, d_ff, dropout, max_len)


if __name__ == '__main__':
    # 直接运行本文件时，给出一个最简的自检：能否完成一次前向传播。
    torch.manual_seed(0)
    model = Transformer(src_vocab_size=20, tgt_vocab_size=20,
                        d_model=32, num_heads=4, num_layers=2, d_ff=64, dropout=0.0)
    src = torch.randint(1, 20, (2, 5))
    tgt = torch.randint(1, 20, (2, 6))
    src_mask = make_padding_mask(src)
    tgt_mask = make_padding_mask(tgt) & make_causal_mask(tgt.size(1))
    out = model(src, tgt, src_mask, tgt_mask)
    print("src.shape =", tuple(src.shape))
    print("tgt.shape =", tuple(tgt.shape))
    print("out.shape =", tuple(out.shape), "（应为 batch × tgt_len × tgt_vocab_size）")
    print("自检通过 ✅  请运行 python main.py 查看完整演示。")
