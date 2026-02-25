import torch
import torch.nn as nn
import torch.nn.functional as F 

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention) 模块
    
    功能：
    - 将输入的 Query、Key、Value 通过多个“注意力头”并行处理，再融合结果，以捕获不同子空间中的依赖关系，提升模型表达能力
    - 通过控制 Query、Key、Value 的来源，还可以实现自注意力机制 (Self-Attention) 和交叉注意力机制 (Cross-Attention)
          
    典型应用场景：Transformer 的编码器和解码器中
    """

    def __init__(self, d_model, num_heads, dropout = 0.1):
        """
        初始化多头注意力模块

        输入参数:
        - d_model (int): 模块的输入、输出和隐藏层维度。它的大小必须要跟 Query、Key、Value 的大小一致（考虑到该模块经常用在Transformer结构中，一致便于残差连接）
        - num_heads (int): 注意力头的数量（例如 8）。d_model 必须能被 num_heads 整除
        - dropout (float): Attention Dropout 层的丢弃率
        
        内部属性:
        - self.d_k (int): 每个注意力头的输入和输出的维度，等于 d_model // num_heads
        - W_q, W_k, W_v (nn.Linear): 分别用于对输入的 Query、Key、Value 进行线性变换，生成各头所需的表示
        - W_o (nn.Linear): 将多头输出拼接后的结果映射回原始 d_model 维度
        """

        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model                      # 模块的输入、输出和隐藏层维度
        self.num_heads = num_heads                  # 注意力头的数量
        self.dropout = dropout                      # 丢弃率
        assert self.d_model % self.num_heads == 0, 'd_model 必须能被 num_heads 整除'
        self.d_k = self.d_model // self.num_heads   # 每个注意力头的输入和输出的维度

        # 定义线性投影层：将输入映射到各自的参数空间
        # 让每一个token对应的 Query、Key、Value 先自身做充分的融合，这样就不必担心后续的分割操作会破坏语义，进而保证多头处理是有意义的
        self.W_q = nn.Linear(self.d_model, self.d_model) # Query映射
        self.W_k = nn.Linear(self.d_model, self.d_model) # Key映射
        self.W_v = nn.Linear(self.d_model, self.d_model) # Value映射

        # 输出投影层：将拼接后的多头结果映射回 d_model 维度
        self.W_o = nn.Linear(self.d_model, self.d_model)

        # Attention Dropout 层，削弱模块对某个头或者某个特定位置的高度依赖
        self.attn_dropout = nn.Dropout(dropout) 

    def forward(self, Q, K, V, mask = None):
        """
        前向传播

        输入参数:
        - Q: Query序列，[B, L_q, d_model]
        - K: Key  序列，[B, L_kv, d_model]
        - V: Value序列，[B, L_kv, d_model]
        - mask: 可选，用于屏蔽无效位置（如 padding 或 future tokens），其中 True 代表有效，False 代表无效

        注意：
        - 在自注意力（Self-Attention）中，Q = K = V = x，此时seq_len_q必定等于seq_len_kv
        - 在交叉注意力和多模态对齐的情况下seq_len_q不等于seq_len_kv是常有的事

        流程:
        1. 对 Q/K/V 分别做线性投影（仍保持 d_model 维度），即自我融合
        2. 拆分为多头 →         [B, H, L_q, d_k], [B, H, L_kv, d_k]
        3. 计算缩放点积注意力
        4. 合并多头输出结果 →   [B, L_q, d_model]
        5. 最终线性投影 →       [B, L_q, d_model]

        返回:
        - output: [B, L_q, d_model]
        """

        # 1. 线性投影（自我融合，维度不变）
        Q = self.W_q(Q) # [B, L_q, d_model]
        K = self.W_k(K) # [B, L_kv, d_model]
        V = self.W_v(V) # [B, L_kv, d_model]

        # 2. 拆分多头
        B, L_q, _ = Q.size()
        _, L_kv, _ = K.size()

        Q = Q.view(B, L_q, self.num_heads, self.d_k).transpose(1, 2) # [B, H, L_q, d_k]
        K = K.view(B, L_kv, self.num_heads, self.d_k).transpose(1, 2) # [B, H, L_kv, d_k]
        V = V.view(B, L_kv, self.num_heads, self.d_k).transpose(1, 2) # [B, H, L_kv, d_k]

        # 3. 计算缩放点积注意力
        # 3.1. 注意力分数：(Q @ K^T) / (d_k^0.5) [B, H, L_q, L_kv]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # 3.2. mask 操作：mask 中标记为 False 的位置对应的注意力分数用一个极小值来替换（Softmax 后几乎为 0）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == False, -1e9) # 不用 float('-inf') 是防止除 0 得到 NaN
        # 3.3. Softmax 归一化 [B, H, L_q, L_kv]
        attn_probs = F.softmax(attn_scores, dim = -1)
        # 3.4. Attention Dropout [B, H, L_q, L_kv]
        # 该操作会将 attn_probs 的最后一个维度中 self.dropout 比例的值置为 0，然后再令该维度的所有值除以 (1 - self.dropout)
        # 不要在Softmax之前做 Dropout 操作，会破坏 Softmax 的数值稳定性
        attn_probs = self.attn_dropout(attn_probs)
        # 3.5. 加权融合 [B, H, L_q, d_k]
        attn_output = torch.matmul(attn_probs, V)

        # 4. 合并多头输出结果
        B, H, L_q, d_k = attn_output.size()
        # transpose(): [B, H, L_q, d_k] → [B, L_q, H, d_k]
        # contiguous(): 确保内存连续，以便 view 正确工作
        # view(): 合并 H 和 d_k → d_model
        o_input = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)

        # 5. 最终线性投影 [B, L_q, d_model]
        output = self.W_o(o_input)

        return output 

if __name__ == "__main__":
    # ================================
    # 超参数设置（统一用于所有场景）
    # ================================
    batch_size = 2      # 批次大小：同时处理 2 个样本
    d_model = 512       # 模型维度：每个 token 用 512 维向量表示
    num_heads = 8       # 注意力头数：并行 8 个注意力机制

    # 创建多头注意力模块（可复用于所有场景）
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    print("=" * 60)
    print("【场景 1】普通注意力机制（General Attention）")
    print("→ Q 来自解码器，K/V 来自编码器（如 Transformer 解码器中的交叉注意力）")
    print("=" * 60)

    # 模拟解码器当前时刻的查询（例如目标语言的前缀）
    L_q = 3
    Q = torch.rand(batch_size, L_q, d_model)  # [B=2, L_q=3, d_model=512]

    # 模拟编码器输出的键值对（例如源语言的完整表示）
    L_kv = 5
    K = torch.rand(batch_size, L_kv, d_model)  # [B=2, L_kv=5, d_model=512]
    V = torch.rand(batch_size, L_kv, d_model)  # [B=2, L_kv=5, d_model=512]

    # 构建 padding mask：假设第 2 个样本的最后 2 个 token 是 padding（无效）
    # mask 形状需广播到 [B, H, L_q, L_kv]，但输入时只需 [B, 1, 1, L_kv] 或 [B, L_kv]
    # 先构建一个形状为 [L_kv] 的张量，其中的取值为 [0, 1, ..., L_kv - 1]。该张量表示 token 在序列中的索引
    src_seq_index = torch.arange(L_kv) # [L_kv]
    # 再构建形状为 [B] 的张量。该张量表示每个序列有效 token 的长度
    src_valid_len = torch.tensor([5, 3])  # [B] 第一个样本长度5（全有效），第二个长度3（后2个是padding）
    # 对 src_seq_index 和 src_valid_len 各自插入新的维度，使其形状分别变为 [1, L_kv] 和 [B, 1]，进而满足广播条件
    # 检验两个张量可否广播的方法：
    # 取出二者的形状元组，右对齐。如果元组长度不同则在短的元组左侧添加 1，直到相等
    # 逐一比较两个形状元组中的元素，如果二者相同或者其中一个为 1，则说明当前维度可以广播
    # 如果所有维度都可以广播，则说明这两个张量可以广播，否则不可广播，需要进行一定的维度调整
    padding_mask = src_seq_index.unsqueeze(0) < src_valid_len.unsqueeze(1)  # [B=2, L_kv=5]，其中 0 代表需要mask
    # 转换为与 attn_scores 兼容的形状：[B, 1, 1, L_kv] → 广播到所有 heads 和 queries
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L_kv]

    # 前向传播（使用 padding mask）
    output = mha(Q, K, V, mask=padding_mask)

    print(f"Q 形状: {Q.shape}  # 解码器查询（目标序列）")
    print(f"K/V 形状: {K.shape}  # 编码器输出（源序列）")
    print(f"Padding mask 形状: {padding_mask.shape}  # 屏蔽源序列中的 padding 位置")
    print(f"输出形状: {output.shape}  # [B, L_q, d_model]")
    print()

    # ======================================================================
    print("=" * 60)
    print("【场景 2】自注意力机制（Self-Attention）")
    print("→ Q = K = V = x，常见于 Transformer 编码器或解码器的自注意力层")
    print("→ 使用 padding mask 处理变长输入")
    print("=" * 60)

    L_x = 6
    x = torch.rand(batch_size, L_x, d_model)  # [B=2, L_x=6, d_model=512]

    # 假设两个样本的实际长度分别为 6 和 4（后 2 个是 padding）
    tgt_seq_index = torch.arange(L_x) # [L_x]
    tgt_valid_len = torch.tensor([6, 4]) # [B]
    padding_mask_self = tgt_seq_index.unsqueeze(0) < tgt_valid_len.unsqueeze(1)  # [B, L_x]，其中 0 代表需要mask
    padding_mask_self = padding_mask_self.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L_x]

    # 自注意力：Q=K=V=x
    output_self = mha(x, x, x, mask=padding_mask_self)

    print(f"输入 x 形状: {x.shape}  # 同时作为 Q/K/V")
    print(f"Padding mask 形状: {padding_mask_self.shape}  # 屏蔽自身序列中的 padding")
    print(f"输出形状: {output_self.shape}  # [B, L_x, d_model]")
    print()

    # ======================================================================
    print("=" * 60)
    print("【场景 3】带 Look-Ahead Mask 的自注意力（用于 Transformer 解码器训练）")
    print("→ 防止当前位置关注未来 token（因果掩码 / causal mask）")
    print("→ 同时结合 padding mask，处理变长序列")
    print("=" * 60)

    L_x = 5
    x = torch.rand(batch_size, L_x, d_model)  # [B=2, L=5, d_model=512]

    # 构建 look-ahead mask（上三角为 False，下三角含对角线为 True）
    look_ahead_mask = torch.tril(torch.ones(L_x, L_x)).bool()  # [L_x, L_x]，如果是上三角含对角线，则用torch.triu()
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L_x, L_x] → 可广播

    # 构建 padding mask（假设两个样本长度为 5 和 3）
    dec_seq_index = torch.arange(L_x) # [L_x]
    dec_valid_len = torch.tensor([5, 3]) # [B]
    padding_mask = dec_seq_index.unsqueeze(0) < dec_valid_len.unsqueeze(1)  # [B, L_x]
    # 扩展为 [B, 1, 1, L] 以匹配 attention scores 的最后一维
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L_x]

    # 合并两种 mask：只有当两个 mask 都为 True 时才保留
    # 所以需要将 look-ahead mask 与 padding_mask 做逻辑与操作
    combined_mask = look_ahead_mask & padding_mask # [B, 1, L_x, L_x]

    # 自注意力 + 因果掩码 + padding 掩码
    output_causal = mha(x, x, x, mask=combined_mask)

    print(f"解码器输入形状: {x.shape}  # [B=2, L=5, d_model=512]")
    print(f"Look-ahead mask 形状: {look_ahead_mask.shape}  # [1, 1, L_x, L_x]，下三角有效")
    print(f"Padding mask 形状: {padding_mask.shape}  # [B, 1, 1, L_x]")
    print(f"合并后 mask 形状: {combined_mask.shape}  # [B, 1, L_x, L_x]，确保不看未来且跳过 padding")
    print(f"输出形状: {output_causal.shape}  # [B, L_x, d_model]")
    print()
