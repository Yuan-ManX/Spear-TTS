import torch
from torch import nn, einsum
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version

from einops import rearrange, repeat


# 定义一个命名元组，用于配置高效注意力机制
"""
    EfficientAttentionConfig 是一个命名元组，用于配置高效注意力机制。

    字段:
        enable_flash (bool): 是否启用 Flash Attention。
        enable_math (bool): 是否启用数学计算方式的注意力机制。
        enable_mem_efficient (bool): 是否启用内存高效方式的注意力机制。
"""
Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def exists(val):
    """
    检查一个值是否存在（即不为 None）。

    Args:
        val: 需要检查的值。

    Returns:
        bool: 如果值不为 None，则返回 True；否则返回 False。
    """
    return val is not None


def once(fn):
    """
    装饰器函数，用于确保被装饰的函数只被调用一次。

    Args:
        fn (Callable): 需要被限制只调用一次的函数。

    Returns:
        Callable: 装饰后的函数。
    """
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

# 使用 once 装饰器装饰 print 函数，创建一个只调用一次的 print_once 函数
print_once = once(print)


class Attend(nn.Module):
    """
    Attend 模块实现了自注意力机制，支持因果掩码和 Flash Attention。

    Args:
        dropout (float, optional): Dropout 概率，应用于注意力计算后的输出。默认为 0。
        causal (bool, optional): 是否使用因果掩码，防止模型在预测时看到未来的信息。默认为 False。
        flash (bool, optional): 是否使用 Flash Attention 来加速计算。默认为 False。
    """
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        flash = False
    ):
        super().__init__()
        self.dropout = dropout
        # 定义注意力 Dropout 层，用于在注意力计算后应用 Dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        # 注册一个缓冲区用于存储掩码，不持久化到模型状态中
        self.register_buffer("mask", None, persistent=False)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # A100 GPU 使用 Flash Attention
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            # 非 A100 GPU 使用数学或内存高效注意力
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def get_mask(self, i, j, device):
        """
        生成掩码，用于遮蔽注意力机制中不需要的部分。

        Args:
            i (int): 当前时间步的索引。
            j (int): 目标时间步的索引。
            device (torch.device): 张量所在的设备。

        Returns:
            torch.Tensor: 掩码张量，形状为 [i, j]。
        """
        # 计算最大时间步长度
        n = max(i, j)

        if exists(self.mask) and self.mask.shape[-1] >= n:
            # 如果已有掩码且足够大，则复用
            mask = self.mask[:n, :n]
        else:
            # 否则，生成新的上三角布尔掩码
            mask = torch.ones((n, n), device = device, dtype = torch.bool).triu(1)
            self.register_buffer("mask", mask, persistent = False)

        # 返回当前时间步所需的掩码部分
        return mask[-i:, :]

    def flash_attn(self, q, k, v, mask = None):
        """
        使用 Flash Attention 计算注意力机制。

        Args:
            q (torch.Tensor): 查询张量，形状为 (batch_size, heads, q_len, head_dim)。
                - batch_size: 批次大小。
                - heads: 注意力头的数量。
                - q_len: 查询序列的长度。
                - head_dim: 每个注意力头的维度。

            k (torch.Tensor): 键张量，形状为 (batch_size, heads, k_len, head_dim)。
                - k_len: 键序列的长度。

            v (torch.Tensor): 值张量，形状为 (batch_size, heads, k_len, head_dim)。

            mask (Optional[torch.Tensor], optional): 注意力掩码，形状为 (batch_size, q_len)。默认为 None。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, heads, q_len, head_dim)。
        """
        # 获取张量形状信息
        _, heads, q_len, _, k_len, causal, is_cuda, device = *q.shape, k.shape[-2], self.causal, q.is_cuda, q.device

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        # 检查掩码是否存在并扩展到兼容的形状
        # 掩码的形状为 B L，因此需要扩展到 B H N L

        if exists(mask):
            # 调整形状以匹配注意力计算
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention
        # 检查是否有兼容的设备用于 Flash Attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # if q and k lengths differ (caching of key/values), and causal, manually construct causal attn mask as float, as not supported (flash attn 2 will support this eventually)
        # 如果 q 和 k 的长度不同（键/值缓存），并且是因果的，则手动构造因果掩码为浮点数，因为不支持（Flash Attention 2 最终会支持）
        row_is_entirely_masked = None

        if causal and q_len != k_len:
            causal_mask = self.get_mask(q_len, k_len, device = device)

            if exists(mask):
                mask = mask & ~causal_mask
            else:
                mask = ~causal_mask

            row_is_entirely_masked = ~mask.any(dim = -1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        """
        前向传播函数，执行自注意力机制。
        使用爱因斯坦求和约定（Einstein Notation）进行张量操作。

        Args:
            q (torch.Tensor): 查询张量，形状为 (batch_size, heads, seq_len_q, head_dim)。
                - batch_size: 批次大小。
                - heads: 注意力头的数量。
                - seq_len_q: 查询序列的长度。
                - head_dim: 每个注意力头的维度。

            k (torch.Tensor): 键张量，形状为 (batch_size, heads_k, seq_len_k, head_dim)。
                - heads_k: 键序列的注意力头数量。
                - seq_len_k: 键序列的长度。

            v (torch.Tensor): 值张量，形状为 (batch_size, heads_v, seq_len_v, head_dim)。
                - heads_v: 值序列的注意力头数量。
                - seq_len_v: 值序列的长度。

            mask (Optional[torch.Tensor], optional): 注意力掩码，形状为 (batch_size, seq_len_q, seq_len_k)。默认为 None。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, heads, seq_len_q, head_dim)。

        爱因斯坦求和约定（Einstein Notation）中的符号说明:
            b - 批次大小 (batch)
            h - 注意力头的数量 (heads)
            n, i, j - 序列长度 (基础序列长度, 源序列长度, 目标序列长度)
            d - 特征维度 (feature dimension)
        """
        # 获取查询张量的序列长度和设备信息
        n, device = q.shape[-2], q.device
        # 获取查询张量和键张量的注意力头数量
        heads, kv_heads = q.shape[1], k.shape[1]
        
        # 如果键和值张量的注意力头数量少于查询张量，则重复键和值张量以匹配查询的注意力头数量
        if kv_heads < heads:
            k, v = map(lambda t: repeat(t, 'b h ... -> b (g h) ...', g = heads // kv_heads), (k, v))

        # 计算缩放因子，缩放因子为查询特征维度的平方根的倒数
        scale = q.shape[-1] ** -0.5

        # 如果使用 Flash Attention，则调用 flash_attn 方法
        if self.flash:
            return self.flash_attn(q, k, v, mask = mask)

        # similarity
        # 计算相似度矩阵
        # 使用爱因斯坦求和约定进行批量矩阵乘法，并进行缩放
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask
        # 应用键填充掩码
        if exists(mask):
            # 调整掩码形状以匹配相似度矩阵
            mask = rearrange(mask, 'b j -> b 1 1 j')
            # 使用掩码遮蔽相似度矩阵中的无效位置
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask
        # 应用因果掩码
        if self.causal:
            # 获取当前时间步的因果掩码
            i, j = sim.shape[-2:]
            causal_mask = self.get_mask(i, j, device)
            # 使用因果掩码遮蔽相似度矩阵中的未来位置
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        # 计算注意力权重
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values
        # 使用爱因斯坦求和约定进行批量矩阵乘法，聚合值
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out
