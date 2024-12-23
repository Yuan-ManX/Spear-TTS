import math
from pathlib import Path
from functools import partial
from random import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor, nn, einsum, IntTensor, LongTensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset

from einops import rearrange, repeat, pack, reduce
from einops.layers.torch import Rearrange
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Union, Callable, Literal, Tuple, List

from audiolm_pytorch import FairseqVQWav2Vec, HubertWithKmeans
from audiolm_pytorch.data import get_dataloader
from rotary_embedding_torch import RotaryEmbedding
from x_clip.tokenizer import tokenizer

from attend import Attend
from distributed import all_gather


# 定义 FloatTensor 类型别名，可以是 CPU 上的 FloatTensor 或 CUDA 上的 FloatTensor
FloatTensor = Union[
    torch.FloatTensor, # CPU 上的 32 位浮点张量
    torch.cuda.FloatTensor # CUDA 上的 32 位浮点张量
]


def exists(val):
    """
    检查一个值是否存在（即不为 None）。

    Args:
        val: 需要检查的值。

    Returns:
        bool: 如果值不为 None，则返回 True；否则返回 False。
    """
    return val is not None


def default(val, d):
    """
    如果值存在（即不为 None），则返回该值；否则返回默认值。

    Args:
        val: 需要检查的值。
        d: 默认值。

    Returns:
        如果 val 存在，则返回 val；否则返回 d。
    """
    return val if exists(val) else d


def empty(t: Tensor):
    """
    检查一个张量是否为空（即元素数量为零）。

    Args:
        t (torch.Tensor): 需要检查的张量。

    Returns:
        bool: 如果张量为空，则返回 True；否则返回 False。
    """
    return t.numel() == 0


def l2norm(t):
    """
    对张量进行 L2 归一化。

    L2 归一化会将张量的每个向量（通常是最后一个维度）归一化为单位向量。

    Args:
        t (torch.Tensor): 需要归一化的张量。

    Returns:
        torch.Tensor: 归一化后的张量。
    """
    return F.normalize(t, dim = -1)


def set_eos_id(t: Tensor, eos_id: int, pad_id: int):
    """
    在张量的每个序列末尾添加 EOS（End of Sequence）标识符。

    Args:
        t (torch.Tensor): 输入张量，形状为 (batch_size, seq_length)。
        eos_id (int): EOS 的标识符 ID。
        pad_id (int): 填充符的标识符 ID。

    Returns:
        torch.Tensor: 在每个序列末尾添加了 EOS 标识符后的张量。
    """
    # 计算每个序列中第一个 pad_id 之前的位置，作为 EOS 的插入位置
    eos_indices = ((t == pad_id).cumsum(dim = -1) == 0).sum(dim = -1, keepdim = True).long()

    # 生成一个范围张量，用于索引 batch
    batch_range = torch.arange(t.shape[0], device = t.device, dtype = torch.long)
    # 重塑为 (batch_size, 1)
    batch_range = rearrange(batch_range, '... -> ... 1')

    # 在每个序列末尾添加一个填充符
    t = F.pad(t, (0, 1), value = pad_id)
    # 在指定的位置插入 EOS 标识符
    t[batch_range, eos_indices] = eos_id
    return t


def batch_unique_consecutive(t, pad_value = 0.):
    """
    对批次中的每个序列执行 unique_consecutive 操作，并填充以保持批次形状。

    Args:
        t (torch.Tensor): 输入张量，形状为 (batch_size, seq_length)。
        pad_value (float, optional): 填充值，默认为 0。

    Returns:
        torch.Tensor: 每个序列经过 unique_consecutive 处理后的张量，形状为 (batch_size, new_seq_length)。
    """
    # 对批次中的每个序列执行 unique_consecutive 操作
    unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim = 0)]
    # 对处理后的序列进行填充，以保持批次形状
    return pad_sequence(unique_arr, batch_first = True, padding_value = pad_value)


def mask_after_eos(target, eos_id, pad_id):
    """
    在 EOS 标识符之后的位置应用掩码，将其替换为填充符。

    Args:
        target (torch.Tensor): 目标张量，形状为 (batch_size, seq_length)。
        eos_id (int): EOS 的标识符 ID。
        pad_id (int): 填充符的标识符 ID。

    Returns:
        torch.Tensor: 在 EOS 之后的位置被掩码替换为填充符后的张量。
    """
    # 生成一个掩码，标记每个序列中 EOS 之后的位置
    mask = (target == eos_id).cumsum(dim = -1) > 0
    # 在掩码的末尾添加一个 False 值，以避免最后一个位置被掩码
    mask = F.pad(mask, (1, -1), value = False)
    # 使用掩码将 EOS 之后的位置替换为填充符
    return target.masked_fill(mask, pad_id)


def safe_div(num, den, eps = 1e-10):
    """
    安全地执行除法操作，避免除以零。

    Args:
        num (torch.Tensor): 分子张量。
        den (torch.Tensor): 分母张量。
        eps (float, optional): 一个极小值，默认为 1e-10。

    Returns:
        torch.Tensor: 除法结果。
    """
    return num / max(den, eps)


def find_first_true_index(bool_tensor, dim = -1):
    """
    查找布尔张量中每个序列第一个 True 值的位置。

    Args:
        bool_tensor (torch.Tensor): 布尔张量。
        dim (int, optional): 查找的维度，默认为最后一个维度。

    Returns:
        torch.Tensor: 每个序列中第一个 True 值的位置索引。
    """
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)


# freezing and unfreezing helpers

def set_requires_grad_(module: Module, requires_grad: bool):
    """
    设置模块中所有参数的 requires_grad 属性。

    Args:
        module (torch.nn.Module): 需要设置参数的模块。
        requires_grad (bool): 是否需要梯度。True 表示需要梯度，False 表示不需要梯度。
    """
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze(module: Module):
    """
    冻结模块，使其参数不参与梯度计算。

    Args:
        module (torch.nn.Module): 需要冻结的模块。
    """
    set_requires_grad_(module, False)


def unfreeze(module: Module):
    """
    解冻模块，使其参数参与梯度计算。

    Args:
        module (torch.nn.Module): 需要解冻的模块。
    """
    set_requires_grad_(module, True)


# sampling helpers

def eval_decorator(fn):
    """
    装饰器，用于在函数执行前后切换模型的训练模式。

    该装饰器在执行被装饰的函数之前将模型设置为评估模式（eval），执行完毕后恢复之前的训练模式（train）。

    Args:
        fn (callable): 被装饰的函数。

    Returns:
        callable: 装饰后的函数。
    """
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner


def log(t, eps = 1e-20):
    """
    对张量进行对数运算，并添加一个极小值以避免对数运算中的数值不稳定。

    Args:
        t (torch.Tensor): 输入张量。
        eps (float, optional): 极小值，默认为 1e-20。

    Returns:
        torch.Tensor: 对数运算后的张量。
    """
    return torch.log(t.clamp(min = eps))


def gumbel_noise(t):
    """
    生成与输入张量形状相同的 Gumbel 噪声。

    Gumbel 噪声常用于实现 Gumbel-Softmax 技巧，用于从离散分布中进行采样。

    Args:
        t (torch.Tensor): 输入张量。

    Returns:
        torch.Tensor: 与输入张量形状相同的 Gumbel 噪声。
    """
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature = 1., dim = -1):
    """
    使用 Gumbel-Softmax 技巧对输入张量进行采样。

    Args:
        t (torch.Tensor): 输入张量，通常是 logits。
        temperature (float, optional): 温度参数，用于控制采样的平滑程度。默认为 1.0。
        dim (int, optional): 采样的维度。默认为最后一个维度。

    Returns:
        torch.Tensor: 采样后的张量，形状与输入张量相同。
    """
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)


def top_p(logits, thres = 0.9):
    """
    对 logits 应用 top-p（核采样）方法，保留累计概率不超过阈值的 top tokens。

    Args:
        logits (torch.Tensor): 输入的 logits 张量。
        thres (float, optional): 概率阈值，默认为 0.9。

    Returns:
        torch.Tensor: 应用 top-p 后的 logits 张量。
    """
    # 对 logits 进行降序排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # 计算累计概率
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # 找到需要移除的 tokens
    sorted_indices_to_remove = F.pad(cum_probs > thres, (1, -1), value = 0)
    # 将需要移除的 tokens 的 logits 设为负无穷
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    # 将排序后的 logits 重新排列回原始顺序
    sorted_logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)
    # 返回应用 top-p 后的 logits 张量。
    return sorted_logits


def top_k(logits, thres = 0.1, k = None):
    """
    对 logits 应用 top-k 方法，保留前 k 个 logits。

    Args:
        logits (torch.Tensor): 输入的 logits 张量。
        thres (float, optional): 比例阈值，用于动态计算 k 值。默认为 0.1。
        k (int, optional): 要保留的 top tokens 数量。如果未提供，则根据 thres 计算。

    Returns:
        torch.Tensor: 应用 top-k 后的 logits 张量。
    """
    if not exists(k):
        # 根据比例阈值计算 k 值
        k = math.ceil(thres * logits.shape[-1])
    # 找到前 k 个 logits 和对应的索引
    val, ind = torch.topk(logits, k, dim = -1)
    # 创建一个全为负无穷的张量
    probs = torch.full_like(logits, float('-inf'))
    # 将前 k 个 logits 的值填入 probs 中
    probs.scatter_(-1, ind, val)
    # 返回应用 top-k 后的 logits 张量。
    return probs


# residual wrapper

class Residual(nn.Module):
    """
    Residual 模块，用于实现残差连接。

    Args:
        fn (callable): 要应用的函数或模块。
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        # 前向传播函数，将输入 x 通过函数 fn 处理后与原始输入 x 相加，实现残差连接。
        return self.fn(x, **kwargs) + x


# rmsnorm

class RMSNorm(nn.Module):
    """
    RMSNorm 模块，实现均方根归一化。

    Args:
        dim (int): 输入张量的维度。
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 前向传播函数，对输入张量进行归一化后乘以缩放因子和 gamma 参数。
        return F.normalize(x, dim = -1) * self.scale * self.gamma


# feedforward

class GEGLU(nn.Module):
    """
    GEGLU (Gated Linear Unit with Gaussian Error Linear Units) 模块。

    GEGLU 是门控线性单元的一种变体，结合了 GLU 和 GELU 激活函数。
    它将输入张量沿最后一个维度分成两部分：一部分用于门控，另一部分用于计算输出。

    Args:
        None

    Forward Args:
        x (torch.Tensor): 输入张量，形状为 (..., dim)。
    """
    def forward(self, x):
        """
        前向传播函数，实现 GEGLU 操作。

        Args:
            x (torch.Tensor): 输入张量，形状为 (..., dim)。

        Returns:
            torch.Tensor: GEGLU 激活后的输出张量。
        """
        # 将输入张量沿最后一个维度分成两部分
        x, gate = x.chunk(2, dim = -1)
        # 对门控部分应用 GELU 激活函数，并将其与输入张量的第一部分相乘
        return F.gelu(gate) * x


def FeedForward(dim, mult = 4, dropout = 0.):
    """
    创建一个前馈神经网络模块。

    该模块通常用于 Transformer 模型中，作为其前馈子层。
    它由 RMSNorm、线性层、GEGLU 激活函数、Dropout 和另一个线性层组成。

    Args:
        dim (int): 输入和输出的维度。
        mult (int, optional): 隐藏层维度的乘数因子，默认为 4。
        dropout (float, optional): Dropout 概率，默认为 0。

    Returns:
        nn.Sequential: 包含前馈网络各层的有序容器。
    """
    # 计算隐藏层的维度
    dim_inner = int(dim * mult * 2 / 3)
    # 返回一个 Sequential 模块，包含以下层：
    return nn.Sequential(
        RMSNorm(dim), # RMS 归一化层
        nn.Linear(dim, dim_inner * 2), # 第一个线性层，将维度从 dim 扩展到 dim_inner * 2
        GEGLU(), # GEGLU 激活函数
        nn.Dropout(dropout), # Dropout 层
        nn.Linear(dim_inner, dim) # 第二个线性层，将维度从 dim_inner 恢复到 dim
    )


# attention

class Attention(nn.Module):
    """
    多头自注意力机制模块，支持多种配置选项，如因果掩码、旋转位置编码、Flash Attention 等。
    
    参数说明:
    - dim: 输入特征的维度。
    - dim_head: 每个注意力头的维度，默认为64。
    - heads: 注意力头的数量，默认为8。
    - kv_heads: 键值对注意力头的数量，如果未指定，则默认与 heads 相同。
    - causal: 是否使用因果掩码，默认为 False。
    - dim_context: 上下文特征的维度，如果未指定，则默认与 dim 相同。
    - dropout: Dropout 概率，默认为0。
    - rotary_emb: 旋转位置编码实例，默认为 None。
    - flash: 是否使用 Flash Attention，默认为 False。
    - add_null_kv: 是否添加空键值对，默认为 False。
    """
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        kv_heads = None,
        causal = False,
        dim_context = None,
        dropout = 0.,
        rotary_emb: Optional[RotaryEmbedding] = None,
        flash = False,
        add_null_kv = False
    ):
        super().__init__()
        # 如果未指定上下文维度，则默认为输入维度
        dim_context = default(dim_context, dim)

        # 初始化多头参数
        self.heads = heads
        self.kv_heads = default(kv_heads, heads)
        assert (self.heads % self.kv_heads) == 0, 'number of key value heads must be divisible by query heads'

        # 计算缩放因子
        self.scale = dim_head ** -0.5
        # 查询向量的总维度
        dim_query_inner = heads * dim_head
        # 键值对向量的总维度
        dim_kv_inner = self.kv_heads * dim_head

        # 初始化旋转位置编码
        self.rotary_emb = rotary_emb

        # 初始化 Attend 模块，用于执行注意力机制
        self.attend = Attend(
            causal = causal,
            flash = flash,
            dropout = dropout
        )

        # 初始化归一化层
        self.norm = RMSNorm(dim)
        # 初始化 Dropout 层
        self.attn_dropout = nn.Dropout(dropout)

        # 定义查询向量的线性变换和重排
        self.to_q = nn.Sequential(
            nn.Linear(dim, dim_query_inner, bias = False),
            Rearrange('b n (h d) -> b h n d', h = self.heads)
        )

        # 定义键值对向量的线性变换和重排
        self.to_kv = nn.Sequential(
            nn.Linear(dim_context, dim_kv_inner * 2, bias = False),
            Rearrange('b n (kv h d) -> kv b h n d', kv = 2, h = self.kv_heads)
        )

        # 定义输出线性层，将注意力输出映射回原始维度
        self.to_out = nn.Linear(dim_query_inner, dim, bias = False)

        # 是否添加空键值对
        self.add_null_kv = add_null_kv
        if add_null_kv:
            # 初始化空键值对
            self.null_kv = nn.Parameter(torch.randn(2, self.kv_heads, 1, dim_head))

    def forward(
        self,
        x,
        context = None,
        mask = None,
        cache = None,
        return_cached_key_values = False
    ):
        """
        前向传播方法，执行多头自注意力机制。
        
        参数说明:
        - x: 输入张量，形状为 (批次大小, 序列长度, 特征维度)。
        - context: 上下文张量，如果未指定，则默认为输入张量 x。
        - mask: 注意力掩码，默认为 None。
        - cache: 缓存的键值对张量，默认为 None。
        - return_cached_key_values: 是否返回缓存的键值对，默认为 False。
        
        返回:
        - 如果 return_cached_key_values 为 False，则返回输出张量。
        - 否则，返回输出张量和新的缓存键值对。
        """
        # 检查是否提供了上下文
        has_context = exists(context)
        b = x.shape[0]

        # 对输入进行归一化
        x = self.norm(x)

        # 如果未提供上下文，则上下文默认为输入 x
        context = default(context, x)

        # 计算查询、键和值向量
        q, k, v = (self.to_q(x), *self.to_kv(context))

        # 如果提供了缓存，则将缓存的键值对与当前的键值对拼接
        if exists(cache):
            ck, cv = cache.unbind(dim = 1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # 缓存新的键值对
        new_cache = torch.stack((k, v), dim = 1)

        # 如果使用了旋转位置编码，则旋转查询和键
        if exists(self.rotary_emb):
            assert not has_context
            q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # 如果添加了空键值对，则将空键值对与当前的键值对拼接
        if self.add_null_kv:
            assert not exists(self.rotary_emb)
            nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = b), self.null_kv)
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

            # 如果提供了掩码，则在掩码前填充一个额外的维度
            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

        # 执行注意力机制
        out = self.attend(q, k, v, mask = mask)

        # 重排输出形状
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 通过线性层映射回原始维度
        out =  self.to_out(out)

        # 如果不需要返回缓存的键值对，则直接返回输出
        if not return_cached_key_values:
            return out

        # 否则，返回输出和新的缓存
        return out, new_cache


# transformer

class Transformer(nn.Module):
    """
    Transformer 模型类，实现了多头自注意力机制和前馈神经网络的多层堆叠。
    
    参数说明:
    - dim: 输入特征的维度。
    - depth: Transformer 层的数量。
    - dim_head: 每个注意力头的维度，默认为64。
    - heads: 注意力头的数量，默认为8。
    - kv_heads: 键值对注意力头的数量，如果未指定，则默认与 heads 相同。
    - causal: 是否使用因果掩码，默认为 False。
    - attn_dropout: 注意力层的 Dropout 概率，默认为0。
    - ff_mult: 前馈网络中间层的维度乘数，默认为4。
    - ff_dropout: 前馈层的 Dropout 概率，默认为0。
    - cross_attend: 是否使用交叉注意力，默认为 False。
    - attn_flash: 是否使用 Flash Attention，默认为 False。
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        kv_heads = None,
        causal = False,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        cross_attend = False,
        attn_flash = False
    ):
        super().__init__()

        # 初始化旋转位置编码
        rotary_emb = RotaryEmbedding(dim_head)

        # 初始化 Transformer 层列表
        self.layers = nn.ModuleList([])

        # 逐层构建 Transformer
        for _ in range(depth):
            # 构建自注意力层
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, kv_heads = kv_heads, dropout = attn_dropout, rotary_emb = rotary_emb, flash = attn_flash),
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash, add_null_kv = True) if cross_attend else None,
                # 构建前馈网络层
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        # 初始化最终的归一化层
        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        context_mask = None,
        cache = None,
        return_cache = False,
        return_hiddens = False,
        early_exit_at_layer = None,
        seq_start_pos = None
    ):
        """
        Transformer 的前向传播方法，执行多层自注意力和前馈网络的前向计算。
        
        参数说明:
        - x: 输入张量，形状为 (批次大小, 序列长度, 特征维度)。
        - mask: 自注意力掩码，默认为 None。
        - context: 上下文张量，用于交叉注意力，默认为 None。
        - context_mask: 交叉注意力掩码，默认为 None。
        - cache: 缓存的键值对张量，默认为 None。
        - return_cache: 是否返回缓存的键值对，默认为 False。
        - return_hiddens: 是否返回中间隐藏状态，默认为 False。
        - early_exit_at_layer: 提前退出的层数，默认为 None。
        - seq_start_pos: 序列起始位置，默认为 None。
        
        返回:
        - 如果 return_hiddens 和 return_cache 均为 False，则返回输出张量。
        - 如果 return_hiddens 为 True，则返回输出张量和中间隐藏状态。
        - 如果 return_cache 为 True，则返回输出张量和缓存的键值对。
        - 如果同时 return_hiddens 和 return_cache 为 True，则返回输出张量、中间隐藏状态和缓存的键值对。
        """
        # 检查是否提供了上下文
        has_context = exists(context)

        # 如果提供了 seq_start_pos，则生成掩码
        if exists(seq_start_pos):
            assert not exists(mask)
            seq_len = x.shape[-2]
            seq_arange = torch.arange(seq_len, device = x.device, dtype = torch.long)
            mask = seq_arange >= seq_start_pos[..., None]

        # 如果提供了缓存，则处理缓存
        if exists(cache):
            cached_length, seq_len = cache.shape[-2], x.shape[-2]
            assert seq_len > cached_length
            x = x[:, cached_length:]

        # 初始化缓存和隐藏状态列表
        new_cache = []
        hiddens = []

        # 如果提供了缓存，则使用缓存的键值对；否则使用空迭代器
        if exists(cache):
            iter_cache = iter(cache.unbind(dim = 1))
        else:
            iter_cache = iter([])

        # 逐层处理 Transformer
        for ind, (self_attn, maybe_cross_attn, ff) in enumerate(self.layers):
            layer = ind + 1

            # 保存输入作为残差连接
            residual = x
            # 执行自注意力机制
            attn_out, key_values = self_attn(x, mask = mask, cache = next(iter_cache, None), return_cached_key_values = True)
            x = attn_out + residual

            # 将键值对添加到缓存中
            new_cache.append(key_values)

            # 如果需要交叉注意力，则执行交叉注意力
            if exists(maybe_cross_attn):
                assert has_context
                x = maybe_cross_attn(x, context = context, mask = context_mask) + x

            # 执行前馈网络
            x = ff(x) + x
            # 将中间隐藏状态添加到列表中
            hiddens.append(x)

            # 如果设置了提前退出，则在指定层退出
            if exists(early_exit_at_layer) and early_exit_at_layer == layer:
                break
        
        # 如果设置了提前退出，则根据需要返回结果
        if exists(early_exit_at_layer):
            if return_cache:
                return x, torch.stack(new_cache, dim = 1)
            return x

        # 应用最终的归一化层
        out = self.final_norm(x)

        # 如果需要返回隐藏状态，则返回输出和隐藏状态
        if return_hiddens:
            assert not return_cache
            return out, torch.stack(hiddens)

        # 如果不需要返回缓存，则返回输出
        if not return_cache:
            return out

        # 如果需要返回缓存，则返回输出和缓存
        return out, torch.stack(new_cache, dim = 1)


# class
# 定义一个联合类型，表示语音或文本类型
SpeechOrTextLiteral = Union[
    Literal['speech'], # 表示语音类型
    Literal['text'] # 表示文本类型
]


# 定义一个联合类型，表示语义模型类型
SemanticModelType = Union[
    FairseqVQWav2Vec, # 假设这是基于 Fairseq 的 VQWav2Vec 模型
    HubertWithKmeans # 假设这是基于 Hubert 和 K-means 的模型
]


class TextToSemantic(Module):
    """
    TextToSemantic 类用于将文本输入转换为语义表示。该类集成了文本编码器（如 OpenAI 的 tokenizer 或自定义 tokenizer）、语义模型（如 wav2vec 模型）以及 Transformer 架构，
    以实现从文本到语义的映射，并支持条件指导（classifier-free guidance）和对齐正则化（alignment regularization）。
    
    参数说明:
    - dim: 输入特征的维度。
    - source_depth: 源 Transformer 的层数。
    - target_depth: 目标 Transformer 的层数。
    - num_text_token_ids: 文本 token 的数量。如果未指定，则需要使用 OpenAI 的 tokenizer 或自定义 tokenizer。
    - tokenizer_encode: 自定义 tokenizer 编码函数。如果未指定，则使用 OpenAI 的 tokenizer。
    - use_openai_tokenizer: 是否使用 OpenAI 的 tokenizer。如果为 True，则忽略 tokenizer_encode 和 num_text_token_ids 参数。
    - wav2vec: 语义模型实例（如基于 audiolm-pytorch 的 wav2vec 模型）。如果未指定，则需要指定 num_semantic_token_ids。
    - num_semantic_token_ids: 语义 token 的数量。如果未指定，则需要传入 wav2vec 模型。
    - dim_head: 每个注意力头的维度，默认为64。
    - heads: 注意力头的数量，默认为8。
    - target_kv_heads: 目标 Transformer 中键值对注意力头的数量，用于分组查询注意力以节省解码器推理时的内存。
    - attn_dropout: 注意力层的 Dropout 概率，默认为0。
    - ff_mult: 前馈网络中间层的维度乘数，默认为4。
    - ff_dropout: 前馈层的 Dropout 概率，默认为0。
    - semantic_pad_id: 语义 token 的填充 ID，默认为-1。
    - text_pad_id: 文本 token 的填充 ID，默认为0。
    - autoset_semantic_eos_id: 是否自动设置语义 token 的结束 ID，默认为 True。
    - autoset_text_eos_id: 是否自动设置文本 token 的结束 ID，默认为 True。
    - attn_flash: 是否使用 Flash Attention，默认为 False。
    - cond_drop_prob: 条件指导中条件被丢弃的概率，默认为0。
    - target_early_exit_layer: 目标 Transformer 中提前退出的层数，默认为 None。
    - detach_early_exit_embed: 是否在提前退出时分离嵌入，默认为 False。
    - align_reg_loss_weight: 对齐正则化损失的权重，默认为0.1。
    - align_reg_use_logsumexp_pool: 是否使用 logsumexp 池化进行对齐正则化，默认为 True。
    - align_reg_logsumexp_pool_temp: logsumexp 池化的温度参数，默认为0.1。
    """
    @beartype
    def __init__(
        self,
        dim,
        *,
        source_depth,
        target_depth,
        num_text_token_ids = None,
        tokenizer_encode: Optional[Callable] = None,
        use_openai_tokenizer = False,
        wav2vec: Optional[SemanticModelType] = None,
        num_semantic_token_ids = None,
        dim_head = 64,
        heads = 8,
        target_kv_heads = None,  # for grouped query attention, saving memory on decoder inference
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        semantic_pad_id = -1,
        text_pad_id = 0,
        autoset_semantic_eos_id = True,
        autoset_text_eos_id = True,
        attn_flash = False,
        cond_drop_prob = 0.,
        target_early_exit_layer = None,
        detach_early_exit_embed = False,
        align_reg_loss_weight = 0.1,
        align_reg_use_logsumexp_pool = True,
        align_reg_logsumexp_pool_temp = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.wav2vec = wav2vec

        # 如果提供了 wav2vec 模型，则冻结其参数
        if exists(self.wav2vec):
            freeze(self.wav2vec)

        self.tokenizer_encode = tokenizer_encode

        # 如果使用 OpenAI 的 tokenizer，则忽略 tokenizer_encode 和 num_text_token_ids 参数
        if use_openai_tokenizer:
            assert not exists(tokenizer_encode)
            assert not exists(num_text_token_ids)
            self.tokenizer_encode = tokenizer.tokenize
            num_text_token_ids = tokenizer.vocab_size
        else:
            assert exists(num_text_token_ids), 'num_text_token_ids not specified'

        # 如果提供了 wav2vec 模型，则语义 token 的数量为 codebook_size；否则需要指定 num_semantic_token_ids
        num_semantic_token_ids = wav2vec.codebook_size if exists(wav2vec) else num_semantic_token_ids
        assert exists(num_semantic_token_ids), 'you need to either pass in a wav2vec model from audiolm-pytorch, or specify the number of semantic token ids with num_semantic_token_ids'

        self.num_semantic_token_ids = num_semantic_token_ids
        self.num_text_token_ids = num_text_token_ids

        # padding id, for deriving attention mask automatically if not passed in
        # 填充 ID，用于在未提供掩码时自动生成注意力掩码
        self.semantic_pad_id = semantic_pad_id
        self.text_pad_id = text_pad_id

        self.pad_id = dict(
            speech = semantic_pad_id,
            text = text_pad_id
        )

        # eos id
        # 结束 ID
        self.autoset_eos_id = dict(
            speech = autoset_semantic_eos_id,
            text = autoset_text_eos_id
        )

        self.eos_id = dict(
            speech = num_semantic_token_ids,
            text = num_text_token_ids
        )

        # embedding
        # 嵌入层
        num_semantic_token_ids_with_eos = num_semantic_token_ids + int(autoset_semantic_eos_id)
        num_text_token_ids_with_eos = num_text_token_ids + int(autoset_text_eos_id)

        semantic_token_emb = nn.Embedding(num_semantic_token_ids_with_eos, dim)
        text_token_emb = nn.Embedding(num_text_token_ids_with_eos, dim)

        self.semantic_token_emb = semantic_token_emb

        self.token_emb = nn.ModuleDict(dict(
            speech = semantic_token_emb,
            text = text_token_emb
        ))

        # respective start tokens
        # 起始 token
        self.start_token = nn.ParameterDict(dict(
            speech = nn.Parameter(torch.randn(dim)),
            text = nn.Parameter(torch.randn(dim))
        ))

        # projection to logits
        # 投影到 logit
        to_semantic_logit = nn.Linear(dim, num_semantic_token_ids, bias = False)
        to_text_logit = nn.Linear(dim, num_text_token_ids, bias = False)

        to_semantic_logit.weight = semantic_token_emb.weight
        to_text_logit.weight = text_token_emb.weight

        self.to_logits = nn.ModuleDict(dict(
            speech = to_semantic_logit,
            text = to_text_logit
        ))

        # source and target attention layers
        # source Transformer
        self.source_transformer = Transformer(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            depth = source_depth,
            attn_dropout = attn_dropout,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout,
            causal = False,
            attn_flash = attn_flash
        )

        # target Transformer
        self.target_transformer = Transformer(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            kv_heads = target_kv_heads,
            depth = target_depth,
            attn_dropout = attn_dropout,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout,
            causal = True,
            cross_attend = True,
            attn_flash = attn_flash
        )

        # classifier free guidance - prob of dropping condition
        # CFG 
        assert 0 <= cond_drop_prob < 1
        self.cond_drop_prob = cond_drop_prob

        self.align_reg_loss_weight = align_reg_loss_weight # lambda for weight of regularization loss in https://arxiv.org/abs/2309.08773
        self.align_reg_use_logsumexp_pool = align_reg_use_logsumexp_pool
        self.align_reg_logsumexp_pool_temp = align_reg_logsumexp_pool_temp

        # for speculative decoding, to speed up text-to-speech decoding and make real-time TTS approach more feasible with spear-tts
        # using early exist strategy so one can train just the same model

        self.target_has_early_exit = exists(target_early_exit_layer)
        self.early_exit_layer = target_early_exit_layer

        if self.target_has_early_exit:
            assert 0 < target_early_exit_layer <= target_depth, f'the early exit layer for the speech transformer must be between 1 and {target_depth}'

            self.detach_early_exit_embed = detach_early_exit_embed

            self.to_early_exit_semantic_logits = nn.Sequential(
                Residual(FeedForward(dim)),
                RMSNorm(dim),
                nn.Linear(dim, num_semantic_token_ids_with_eos, bias = False)
            )

    @property
    def device(self):
        """
        获取模型当前所在的设备（CPU 或 GPU）。

        返回:
            torch.device: 模型所在的设备。
        """
        return next(self.parameters()).device

    def load(self, path, strict = True):
        # Return pkg so that if this function gets called from within a Trainer function call,
        # the trainer can also access the package loaded from the checkpoint.
        """
        从指定的路径加载模型参数。

        参数:
            path (str): 模型参数文件的路径。
            strict (bool): 是否严格加载模型状态字典，默认为 True。

        返回:
            dict: 加载的模型状态字典包。

        备注:
            返回包以便在 Trainer 函数调用中访问加载的检查点。
        """
        path = Path(path)
        assert path.exists()
        # 使用 CPU 映射位置加载模型参数
        pkg = torch.load(str(path), map_location = 'cpu')
        # 严格加载模型状态字典
        self.load_state_dict(pkg['model'], strict = strict)
        return pkg

    # a set of freezing / unfreezing utils
    # then rely on get_optimizer to filter out the parameters that do not require grad from being exposed to optimizer
    # 一系列冻结和解冻模型组件的工具方法
    # 这些方法依赖于 get_optimizer 来过滤不需要梯度的参数，防止它们被暴露给优化器

    def unfreeze_all(self):
        """
        解冻模型中的所有参数，使其能够进行梯度更新。
        """
        unfreeze(self)

    def freeze_encoder(self):
        """
        冻结源 Transformer 中的所有参数，使其不进行梯度更新。
        """
        freeze(self.source_transformer)

    def freeze_encoder_below_layer(self, layer: int):
        """
        冻结源 Transformer 中指定层以下的参数，使其不进行梯度更新。
        这在伪标签数据集上的最终训练阶段使用，以便逐步解冻模型。

        参数:
            layer (int): 冻结的层数上限。

        备注:
            对于文本到语义的最终训练，他们冻结编码器的一部分直到某一层。
        """
        unfreeze(self.source_transformer)

        for ind, module in enumerate(self.source_transformer.layers):
            current_layer = ind + 1

            if current_layer <= layer:
                freeze(module)

    def freeze_decoder(self):
        """
        冻结目标 Transformer 中的所有参数，使其不进行梯度更新。
        """
        freeze(self.target_transformer)

    def freeze_speech_emb(self):
        """
        冻结语音嵌入层中的所有参数，使其不进行梯度更新。
        """
        freeze(self.token_emb['speech'])
        self.start_token['speech'].requires_grad = False

    def freeze_text_emb(self):
        """
        冻结文本嵌入层中的所有参数，使其不进行梯度更新。
        """
        freeze(self.token_emb['text'])
        self.start_token['text'].requires_grad = False

    # sampling function

    @torch.no_grad()
    @eval_decorator
    @beartype
    def generate(
        self,
        source: Union[List[str], Tensor],
        *,
        source_type: SpeechOrTextLiteral,
        target_type: SpeechOrTextLiteral,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_fn_kwargs: dict = dict(),
        source_mask: Optional[Tensor] = None,
        max_length = 2048,
        beam_search_decode = False,
        spec_decode = False,
        spec_decode_gamma = 5,
        spec_decode_lenience = 1.,
        beam_size = 4,
        return_source = False,
        return_target_mask = False,
        cond_scale = 1.
    ):
        """
        生成目标序列的方法。

        参数说明:
            source: 输入源，可以是文本列表或张量。
            source_type: 输入源的类型，可以是 'speech' 或 'text'。
            target_type: 目标类型，可以是 'speech' 或 'text'。
            temperature: 生成过程中的温度参数，控制生成的多样性。
            filter_logits_fn: 用于过滤逻辑值的函数，默认为 top_k。
            filter_fn_kwargs: 过滤函数的关键字参数。
            source_mask: 输入源的掩码，默认为 None。
            max_length: 生成的最大长度。
            beam_search_decode: 是否使用束搜索解码。
            spec_decode: 是否使用推测解码。
            spec_decode_gamma: 推测解码的 gamma 参数。
            spec_decode_lenience: 推测解码的宽容度。
            beam_size: 束搜索的束大小。
            return_source: 是否返回输入源。
            return_target_mask: 是否返回目标掩码。
            cond_scale: 条件缩放因子，用于条件指导。

        返回:
            生成的目标序列。
        """
        assert cond_scale >= 1.
        assert not (cond_scale > 1 and self.cond_drop_prob == 0), 'you need to train with conditional drop probability greater than 0 to use classifier free guidance at inference, and it needs to be the right source to target pair'

        # 如果源是张量且类型为 'speech'，则使用 wav2vec 模型进行处理
        if is_bearable(source, FloatTensor) and source_type == 'speech':
            assert exists(self.wav2vec), 'wav2vec should be passed in, if generating with source as raw soundwave'

            with torch.no_grad():
                self.wav2vec.eval()
                source = source.to(self.device)
                source = self.wav2vec(source)

        # 如果源是字符串列表，则使用 tokenizer 进行编码
        if is_bearable(source, List[str]):
            assert exists(self.tokenizer_encode)
            source = self.tokenizer_encode(source)
            source = source.to(self.device)

        batch = source.shape[0]

        source_token_emb = self.token_emb[source_type]
        source_pad_id = self.pad_id[source_type]

        # all target modules and parameters
        # 所有目标模块和参数
        target_token_emb = self.token_emb[target_type]
        target_start_token = self.start_token[target_type]
        target_to_logit = self.to_logits[target_type]
        target_pad_id = self.pad_id[target_type]
        target_eos_id = self.eos_id[target_type]

        # auto set eos id
        # 自动设置 eos id
        if self.autoset_eos_id[source_type]:
            source_eos_id = self.eos_id[source_type]
            source = set_eos_id(source, source_eos_id, pad_id = source_pad_id)

        # if source mask is not passed in
        # automatically derive by the padding id of the modality
        # 如果未提供源掩码，则根据填充 ID 自动推导
        if not exists(source_mask) and source.dtype == torch.long:
            source_mask = source != source_pad_id
        
        # source embedding
        # 源嵌入
        source_emb = source_token_emb(source)

        source_emb = self.source_transformer(source_emb, mask = source_mask)

        # decode target
        # 解码目标
        target = torch.empty((batch, 0), dtype = torch.long, device = self.device)
        start_token = repeat(target_start_token, 'd -> b 1 d', b = batch)

        # loop to decode
        # 解码循环
        assert not (beam_search_decode and spec_decode), 'you must choose either beam decode or speculative decoding, but not both'

        if not beam_search_decode and not spec_decode:
            cache = None
            null_cache = None

            for _ in tqdm(range(max_length)):
                target_emb = target_token_emb(target)
                target_emb = torch.cat((start_token, target_emb), dim = 1)

                # target attention
                # 目标注意力
                attended_target_emb, cache = self.target_transformer(target_emb, context = source_emb, context_mask = source_mask, cache = cache, return_cache = True)

                # decoder logits
                # 解码器逻辑值
                logits = target_to_logit(attended_target_emb)
                logits = logits[:, -1]

                # handle classifier free guidance
                # 处理无分类器指导
                if cond_scale > 1.:
                    null_source_mask = source_mask.float().zero_().bool()

                    attended_null_target_emb, null_cache = self.target_transformer(target_emb, context = source_emb, context_mask = null_source_mask, cache = null_cache, return_cache = True)

                    null_logits = target_to_logit(attended_null_target_emb)
                    null_logits = null_logits[:, -1]

                    logits = null_logits + (logits - null_logits) * cond_scale

                # filter logits
                # 过滤逻辑值
                logits = filter_logits_fn(logits, **filter_fn_kwargs)

                sampled = gumbel_sample(logits, temperature = temperature)
                target, _ = pack((target, sampled), 'b *')

                if not self.autoset_eos_id[target_type]:
                    continue

                is_eos = target == target_eos_id
                all_eos = is_eos.any(dim = -1).all()

                if not all_eos:
                    continue

                target = mask_after_eos(target, target_eos_id, target_pad_id)
                break
        elif beam_search_decode:
            beam = [(target, 0.0, None, None)]

            batch_range = torch.arange(batch, device = self.device, dtype = torch.long)
            batch_range = rearrange(batch_range, 'b -> b 1')

            needs_classifier_free_guidance = cond_scale > 1.

            for _ in tqdm(range(max_length)):
                all_candidates = []
                
                for sentence, sentence_prob, sentence_cache, null_sentence_cache in beam:
                    target_emb = target_token_emb(sentence)
                    target_emb = torch.cat((start_token, target_emb), dim = 1)

                    # target attention

                    attended_target_emb, next_sentence_cache = self.target_transformer(target_emb, context = source_emb, context_mask = source_mask, cache = sentence_cache, return_cache = True)

                    # decoder logits

                    logits = target_to_logit(attended_target_emb)
                    logits = logits[:, -1]

                    # handle classifier free guidance

                    if needs_classifier_free_guidance:
                        null_source_mask = source_mask.float().zero_().bool()

                        attended_null_target_emb, next_null_sentence_cache = self.target_transformer(target_emb, context = source_emb, context_mask = null_source_mask, cache = null_sentence_cache, return_cache = True)

                        null_logits = target_to_logit(attended_null_target_emb)
                        null_logits = null_logits[:, -1]

                        logits = null_logits + (logits - null_logits) * cond_scale
                    else:
                        next_null_sentence_cache = next_sentence_cache[:, 0:0]

                    # log probs for ranking beams

                    log_probs = torch.log_softmax(logits / max(temperature, 1e-10), dim = -1)
                    topk_log_probs, topk_ids = log_probs.topk(beam_size, dim = -1)

                    for i in range(beam_size):
                        candidate = torch.cat([sentence, topk_ids[..., i:i + 1]], dim = -1)
                        candidate_prob = sentence_prob + topk_log_probs[..., i]
                        all_candidates.append((candidate, candidate_prob, next_sentence_cache, next_null_sentence_cache))

                # concat into shape (beam, batch, seq), (beam, batch)
                # 堆叠成形状 (beam, batch, seq), (beam, batch)
                candidates, candidate_probs, candidate_caches, candidate_null_caches = map(partial(torch.stack, dim = 1), zip(*all_candidates))

                # sort by candidate scores across beams
                # 根据候选分数排序
                sorted_indices = candidate_probs.sort(dim = 1, descending = True).indices

                sorted_candidates = candidates[batch_range, sorted_indices]
                sorted_candidate_probs = candidate_probs[batch_range, sorted_indices]
                sorted_candidate_caches = candidate_caches[batch_range, sorted_indices]
                sorted_candidate_null_caches = candidate_null_caches[batch_range, sorted_indices]

                # reconstitute ordered List[Tuple[Tensor, Tensor]]
                # 重新组成有序的列表 [Tuple[Tensor, Tensor]]
                ordered = list(zip(*map(partial(torch.unbind, dim = 1), (sorted_candidates, sorted_candidate_probs, sorted_candidate_caches, sorted_candidate_null_caches))))

                beam = ordered[:beam_size]

                # check if we've hit eos for all sequences
                # 检查所有序列是否已经到达结束符
                all_eos = all([((sentence == target_eos_id).any(dim = -1)).all() for sentence, _, _, _ in beam])

                if all_eos:
                    break

            target = beam[0][0]

            if exists(target_eos_id):
                target = mask_after_eos(target, target_eos_id, target_pad_id)

        elif spec_decode:
            # 推测解码分支
            assert self.target_has_early_exit, 'early exit layer must have been specified and trained in order to use speculative decoding (using the earlier layers of the target transformer as the small fast prediction network)'
            assert source_type == 'text' and target_type == 'speech', 'speculative decoding can only be employed for text-to-speech decoding'

            batch, prompt_seq_len, device = *target.shape, self.device

            cache = None
            small_cache = None

            num_steps = 0
            total_accepted = 0

            batch_range = torch.arange(batch, device = device, dtype = torch.long)[..., None]
            seq_lens = torch.full((batch,), prompt_seq_len, device = device, dtype = torch.long)

            while (seq_lens < max_length).any():

                # predict with smaller network
                # 使用小网络进行预测
                all_small_logits = []
                q_sampled_out = []

                for _ in range(spec_decode_gamma):
                    target_emb = target_token_emb(target)
                    target_emb = torch.cat((start_token, target_emb), dim = 1)

                    small_emb, small_cache = self.target_transformer(
                        target_emb,
                        cache = small_cache,
                        context = source_emb,
                        context_mask = source_mask,
                        return_cache = True,
                        early_exit_at_layer = self.early_exit_layer,
                        seq_start_pos = target.shape[-1] - seq_lens
                    )

                    small_logits = self.to_early_exit_semantic_logits(small_emb)
                    small_logits = small_logits[:, -1]

                    small_logits = filter_logits_fn(small_logits, **filter_fn_kwargs)
                    all_small_logits.append(small_logits)

                    sample = gumbel_sample(small_logits, temperature = temperature, dim = -1)
                    target = torch.cat((target, sample[..., None]), dim = -1)
                    seq_lens += 1

                    q_sampled_out.append(rearrange(sample, 'b -> b 1 1'))

                q_sampled_out = torch.cat(q_sampled_out, dim = -2)
                small_logits = torch.stack(all_small_logits, dim = -2)

                # verify with larger network
                # 使用大网络进行验证
                target_emb = target_token_emb(target)
                target_emb = torch.cat((start_token, target_emb), dim = 1)

                emb, cache = self.target_transformer(
                    target_emb,
                    cache = cache,
                    context = source_emb,
                    context_mask = source_mask,
                    return_cache = True,
                    seq_start_pos = target.shape[-1] - seq_lens
                )

                logits = target_to_logit(emb)
                logits = logits[..., -(spec_decode_gamma + 1):, :]
                logits = filter_logits_fn(logits, **filter_fn_kwargs)

                # prob and prob of small model (p(x) and q(x) in algorithm 1)
                # 计算概率和小网络的概率（算法1中的 p(x) 和 q(x)）
                prob = safe_div(logits, temperature).softmax(dim = -1)
                small_prob = safe_div(small_logits, temperature).softmax(dim = -1)

                p, prob_next = prob[:, :-1], prob[:, -1]

                p = p.gather(-1, q_sampled_out)
                q = small_prob.gather(-1, q_sampled_out) * spec_decode_lenience

                p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]

                r = random_uniform = torch.zeros_like(q).float().uniform_(0, 1)

                accepted = find_first_true_index(r > (p / q))

                total_accepted += accepted.float().mean()
                num_steps += 1

                num_rejected = spec_decode_gamma - accepted
                has_rejected = num_rejected > 0

                accepted = rearrange(accepted, 'b -> b 1')
                accepted.clamp_(max = spec_decode_gamma - 1)

                adjusted_prob = F.relu(prob[batch_range, accepted] - small_prob[batch_range, accepted])
                adjusted_prob = adjusted_prob / adjusted_prob.sum(dim = -1, keepdim = True)
                adjusted_prob = rearrange(adjusted_prob, 'b 1 d -> b d')

                prob_next = torch.where(
                    rearrange(has_rejected, '... -> ... 1'),
                    adjusted_prob,
                    prob_next
                )

                # do a bunch of slicing and align everything to the right, including kv caches
                # 进行切片并对齐所有内容，包括键值缓存
                max_num_rejected = num_rejected.amax()
                seq_arange = torch.arange(target.shape[-1], device = device, dtype = torch.long)

                seq_offset_indices = seq_arange + (max_num_rejected - num_rejected)[..., None]

                seq_lens -= num_rejected
                max_seq_len = seq_lens.amax()

                if batch > 1:
                    target = F.pad(target, (0, max_num_rejected), value = target_pad_id)
                    target = target[batch_range, seq_offset_indices]

                    cache = F.pad(cache, (0, 0, 0, max_num_rejected), value = target_pad_id)
                    small_cache = F.pad(small_cache, (0, 0, 0, max_num_rejected), value = target_pad_id)

                    cache = rearrange(cache, 'b ... n d -> b n ... d')
                    small_cache = rearrange(small_cache, 'b ... n d -> b n ... d')

                    cache = cache[batch_range, seq_offset_indices]
                    small_cache = small_cache[batch_range, seq_offset_indices]

                    cache = rearrange(cache, 'b n ... d -> b ... n d')
                    small_cache = rearrange(small_cache, 'b n ... d -> b ... n d')

                    if target.shape[-1] > max_seq_len:
                        left_index = target.shape[-1] - max_seq_len
                        target = target[:, left_index:]
                        cache = cache[..., left_index:, :]
                        small_cache = small_cache[..., left_index:, :]

                # sample the additional token, one of the tricks in the paper to better bound the worst case
                # 对额外的 token 进行采样，论文中的一种技巧，可以更好地限制最坏情况
                next_token = torch.multinomial(prob_next, 1)

                target = torch.cat((target, next_token), dim = -1)
                seq_lens += 1

                all_eos = (target == target_eos_id).any(dim = -1).all()

                if all_eos:
                    break

            # now left align
            # 现在进行左对齐
            max_seq_lens = seq_lens.amax()

            num_pad_left = target.shape[-1] - seq_lens
            max_pad_left = num_pad_left.amax()
            target = F.pad(target, (0, max_pad_left), value = target_pad_id)

            seq_len_range = torch.arange(min(max_length, max_seq_lens), device = device, dtype = torch.long)
            target = target[batch_range, seq_len_range + num_pad_left[..., None]]
            target = target[..., prompt_seq_len:]

            # mask out anything after eos
            # 掩码掉 eos 之后的所有内容
            if exists(target_eos_id):
                target = mask_after_eos(target, target_eos_id, target_pad_id)

        # whether to return the target mask
        # for variable lengthed generation output
        # needed for conditioning voicebox, NS2, etc
        # 是否返回目标掩码
        # 对于可变长度的生成输出
        # 对于 Voicebox, NS2 等的条件生成是必需的

        if return_target_mask:
            target_mask = target != target_pad_id

        # 4 different types of return cases
        # 四种不同的返回情况

        if not return_source:
            if not return_target_mask:
                return target

            return target, target_mask

        if not return_target_mask:
            return source, target

        return source, target, target_mask

    @beartype
    def forward(
        self,
        source: Union[List[str], Tensor],
        target: Union[List[str], Tensor],
        *,
        source_type: SpeechOrTextLiteral,
        target_type: SpeechOrTextLiteral,
        source_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
        return_loss = False,
        return_logits = False,
        cond_drop_prob: Optional[float] = None,
        should_sim_regularize = True,
        return_early_exit_loss = False
    ):
        """
        前向传播方法，执行从源到目标的转换，并计算损失（如果需要）。

        参数说明:
            source: 输入源，可以是文本列表或张量。
            target: 目标，可以是文本列表或张量。
            source_type: 输入源的类型，可以是 'speech' 或 'text'。
            target_type: 目标类型，可以是 'speech' 或 'text'。
            source_mask: 输入源的掩码，默认为 None。
            target_mask: 目标的掩码，默认为 None。
            return_loss: 是否返回损失，默认为 False。
            return_logits: 是否返回逻辑值，默认为 False。
            cond_drop_prob: 条件丢弃概率，默认为 None。
            should_sim_regularize: 是否进行相似性正则化，默认为 True。
            return_early_exit_loss: 是否返回提前退出损失，默认为 False。

        返回:
            如果 return_loss 为 False，则返回逻辑值张量。
            如果 return_loss 为 True，则返回损失张量和逻辑值张量。
        """
        # 设置条件丢弃概率，默认为类的 cond_drop_prob 属性
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        # 判断是否需要丢弃条件
        drop_cond = cond_drop_prob > 0 and random() < cond_drop_prob

        # 如果源是张量且类型为 'speech'，则使用 wav2vec 模型进行处理
        if is_bearable(source, FloatTensor) and source_type == 'speech':
            assert exists(self.wav2vec), 'wav2vec should be passed in, if generating with source as raw soundwave'

            with torch.no_grad():
                self.wav2vec.eval()
                source = self.wav2vec(source)

        # 如果源是字符串列表，则使用 tokenizer 进行编码
        if is_bearable(source, List[str]):
            assert exists(self.tokenizer_encode)
            source = self.tokenizer_encode(source)
            source = source.to(self.device)

        # 如果目标是字符串列表，则使用 tokenizer 进行编码
        if is_bearable(target, List[str]):
            assert exists(self.tokenizer_encode)
            target = self.tokenizer_encode(target)
            target = target.to(self.device)

        # 确保源和目标的批次大小相同
        assert source.shape[0] == target.shape[0]
        batch = source.shape[0]

        # 获取源和目标的嵌入层和填充 ID
        source_token_emb = self.token_emb[source_type]
        source_pad_id = self.pad_id[source_type]

        # all target modules and parameters
        
        target_token_emb = self.token_emb[target_type]
        target_start_token = self.start_token[target_type]
        target_to_logit = self.to_logits[target_type]
        target_pad_id = self.pad_id[target_type]

        # auto set eos id
        # 自动设置结束 ID
        if self.autoset_eos_id[source_type]:
            source_eos_id = self.eos_id[source_type]
            source = set_eos_id(source, source_eos_id, pad_id = source_pad_id)

        if self.autoset_eos_id[target_type] and return_loss:
            target_eos_id = self.eos_id[target_type]
            target = set_eos_id(target, target_eos_id, pad_id = target_pad_id)

        # if source/target mask is not passed in
        # automatically derive by the padding id of the modality
        # 如果未提供源和目标的掩码，则根据填充 ID 自动推导
        if not exists(source_mask) and source.dtype == torch.long:
            source_mask = source != source_pad_id

        if not exists(target_mask) and target.dtype == torch.long:
            target_mask = target != target_pad_id

            # attend to bos
            # 在目标掩码前填充一个额外的维度，以包含起始符
            target_mask = F.pad(target_mask, (1, 0), value = True)

        # embedding
        # 源嵌入
        source_emb = source_token_emb(source)
        # 目标嵌入，并添加起始符
        target_emb = target_token_emb(target)
        start_token = repeat(target_start_token, 'd -> b 1 d', b = batch)

        target_emb = torch.cat((start_token, target_emb), dim = 1)

        # source attention
        # 源注意力
        source_emb = self.source_transformer(source_emb, source_mask)

        # whether to drop condition, for CFG
        # 是否丢弃条件，用于条件指导（CFG）
        context_mask = source_mask
        if drop_cond:
            context_mask = torch.zeros_like(context_mask).bool()

        # target attention
        # 目标注意力
        target_emb, target_hiddens = self.target_transformer(
            target_emb,
            mask = target_mask,
            context = source_emb,
            context_mask = context_mask,
            return_hiddens = True
        )

        # decoder logits
        # 解码器逻辑值
        logits = target_to_logit(target_emb)

        # 如果不需要返回损失，则直接返回逻辑值
        if not return_loss:
            return logits

        assert (self.training and not empty(target)) or not self.training

        # 计算交叉熵损失
        logits = rearrange(logits[:, :-1], 'b n c -> b c n')

        loss = F.cross_entropy(
            logits,
            target,
            ignore_index = target_pad_id
        )

        # 如果需要返回提前退出损失，则计算提前退出损失
        if return_early_exit_loss:
            assert self.target_has_early_exit, 'you need to set the `target_early_exit_layer` in order to train a predictor on an earlier hidden dimension for speculative decoding'
            assert source_type == 'text' and target_type == 'speech'

            early_layer_index = self.early_exit_layer - 1
            early_embed = target_hiddens[early_layer_index]

            if self.detach_early_exit_embed:
                # a way to train the early exit head without affecting the main loss
                # 一种在不影响主损失的情况下训练提前退出头的方法
                early_embed = early_embed.detach()

            early_exit_logits = self.to_early_exit_semantic_logits(early_embed)
            early_exit_logits = rearrange(early_exit_logits[:, :-1], 'b n c -> b c n')

            early_exit_loss = F.cross_entropy(
                early_exit_logits,
                target,
                ignore_index = target_pad_id
            )

            loss = loss + early_exit_loss

        # 如果需要相似性正则化，并且源类型和目标类型不同，并且条件被丢弃，则计算对齐正则化损失
        if should_sim_regularize and source_type != target_type and drop_cond and self.align_reg_loss_weight > 0:
            # regularizer proposed in https://arxiv.org/abs/2309.08773, alternative to contrastive loss when unconditional
            # supposedly fixes CFG for encoder / decoder transformers
            # 在无条件情况下，提出的正则化器，替代对比损失
            # 据说可以解决编码器/解码器 Transformer 的 CFG 问题

            source_emb, batch_sizes = all_gather(source_emb, 0, None)
            target_emb, _           = all_gather(target_emb, 0, batch_sizes)

            mask_value = -torch.finfo(source_emb.dtype).max

            if exists(source_mask):
                source_emb = source_emb.masked_fill(~source_mask[..., None], mask_value)

            if exists(target_mask):
                target_emb = target_emb.masked_fill(~target_mask[..., None], mask_value)

            # they found that max pool worked best
            # also offer logsumexp pool (smooth max)
            # 他们发现最大池化效果最好
            # 同时提供 logsumexp 池化（平滑最大）

            batch, device = source_emb.shape[0], source_emb.device

            if self.align_reg_use_logsumexp_pool:
                temp = self.align_reg_logsumexp_pool_temp
                source_emb, target_emb = map(lambda t: t / temp, (source_emb, target_emb))
                source_emb = reduce(source_emb, 'b n d -> b d', torch.logsumexp)
                target_emb = reduce(target_emb, 'b n d -> b d', torch.logsumexp)
                source_emb, target_emb = map(lambda t: t * temp, (source_emb, target_emb))
            else:
                source_emb = reduce(source_emb, 'b n d -> b d', 'max')
                target_emb = reduce(target_emb, 'b n d -> b d', 'max')

            source_emb, target_emb = map(l2norm, (source_emb, target_emb))

            source_sim, target_sim = map(lambda t: einsum('i d, j d -> i j', t, t), (source_emb, target_emb))
            diag_mask = torch.eye(batch, device = device, dtype = torch.bool)

            align_reg_loss = F.mse_loss(source_sim[~diag_mask], target_sim[~diag_mask])
            loss = loss + align_reg_loss * self.align_reg_loss_weight

        # 如果不需要返回逻辑值，则返回损失
        if not return_logits:
            return loss

        return loss, logits


# pretraining modules

def get_mask_subset_prob(mask, prob, min_mask = 0):
    """
    根据给定的概率和最小掩码数量，从输入的掩码中选择一个子集掩码。

    参数说明:
        mask (Tensor): 输入的布尔掩码张量，形状为 (batch_size, sequence_length)。
        prob (float): 选择掩码的概率。
        min_mask (int, 可选): 每个样本中至少要掩码的数量，默认为0。

    返回:
        Tensor: 生成的子集掩码张量，形状与输入的 mask 相同。
    """
    # 展开掩码张量的形状，并获取设备信息
    batch, seq, device = *mask.shape, mask.device
    # 计算每个样本中需要掩码的数量，基于给定的概率和最小掩码数量
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    # 生成一个随机数张量，形状为 (batch_size, sequence_length)，值范围在 [0, 1)
    logits = torch.rand((batch, seq), device = device)
    # 将掩码为 False 的位置对应的随机数设为 -1，以便后续排序时忽略这些位置
    logits = logits.masked_fill(~mask, -1)
    
    # 对随机数进行排序，得到排序后的索引
    randperm = logits.argsort(dim = -1).float()
    
    # 计算每个样本中填充（padding）的数量，并从排序后的索引中减去填充的数量
    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    # 根据需要掩码的数量生成子集掩码
    subset_mask = randperm < num_to_mask
    # 将非掩码位置的子集掩码设置为 False，确保掩码位置的一致性
    subset_mask.masked_fill_(~mask, False)
    return subset_mask


class SpeechSpeechPretrainWrapper(nn.Module):
    """
    SpeechSpeechPretrainWrapper 类用于语音到语音的预训练包装器。
    该类在输入语音数据上应用删除掩码（masking），并使用 TextToSemantic 模型进行重建任务。
    支持两种重建方式：
        1. 重建整个序列。
        2. 仅按顺序输出被删除的语音片段。
    
    参数说明:
        model (TextToSemantic): 预训练的 TextToSemantic 模型实例。
        wav2vec (Optional[SemanticModelType]): 可选的 wav2vec 模型实例，用于处理原始音频波形。如果未指定，则使用 model 中的 wav2vec。
        deletion_prob (float): 删除掩码的概率，默认为0.6。
        reconstruct_seq (bool): 是否重建整个序列。如果为 False，则仅按顺序输出被删除的语音片段。
        mask_id (int, 可选): 用于表示掩码位置的 ID。如果提供，则 reconstruct_seq 必须为 True。
    """
    @beartype
    def __init__(
        self,
        model: TextToSemantic,
        wav2vec: Optional[SemanticModelType] = None,
        deletion_prob: float = 0.6,
        reconstruct_seq: bool = False,
        mask_id = None
    ):
        super().__init__()

        self.model = model
        self.wav2vec = default(wav2vec, model.wav2vec)

        self.deletion_prob = deletion_prob
        # 是否重建整个序列，或者仅按顺序输出被删除的语音片段
        self.reconstruct_seq = reconstruct_seq # whether to reconstruct the entire sequence, or just output the deleted ones in order
        self.mask_id = mask_id

    def forward(
        self,
        x,
        return_early_exit_loss = False
    ):
        """
        前向传播方法，应用删除掩码并进行重建任务。

        参数说明:
            x: 输入的语音数据张量。
            return_early_exit_loss: 是否返回提前退出损失，默认为 False。

        返回:
            loss: 重建任务的损失。
            logits: 模型输出的逻辑值。
        """
        is_raw_audio = x.dtype == torch.float

        # 如果输入是原始音频波形，则使用 wav2vec 模型进行处理
        if is_raw_audio:
            assert exists(self.wav2vec)
            
            with torch.no_grad():
                self.wav2vec.eval()
                x = self.wav2vec(x, flatten = False)

        batch = x.shape[0]

        # 初始化掩码张量，形状与输入 x 相同
        mask = torch.ones_like(x, dtype = torch.bool, device = self.model.device)

        # 如果提供了 mask_id，则需要重建整个序列
        if exists(self.mask_id):
            assert self.reconstruct_seq, 'reconstruct_seq must be true if mask id is provided'
            
            # 将填充位置对应的掩码设置为 False
            mask = mask.masked_fill(x == self.model.semantic_pad_id, False)
            # 生成删除掩码
            delete_mask = get_mask_subset_prob(mask, self.deletion_prob)
            # 应用删除掩码，并用 mask_id 替换被删除的位置
            source = x.masked_fill(delete_mask, self.mask_id)
        else:
            # 生成删除掩码
            delete_mask = get_mask_subset_prob(mask, self.deletion_prob)
            # 移除被删除的位置，重塑为 (batch_size, sequence_length)
            source = rearrange(x[~delete_mask], '(b n) -> b n', b = batch)

        # 如果需要重建整个序列，则目标为原始输入 x
        if self.reconstruct_seq:
            target = x
        else:
            # 否则，目标为被删除的位置，重塑为 (batch_size, sequence_length)
            target = rearrange(x[delete_mask], '(b n) -> b n', b = batch)

         # 调用 TextToSemantic 模型的 forward 方法进行重建
        loss, logits = self.model(
            source, target,
            source_type = 'speech',
            target_type = 'speech',
            return_loss = True,
            return_logits = True,
            return_early_exit_loss = return_early_exit_loss,
        )

        return loss, logits


# wrapper for backtranslation task

class SemanticToTextWrapper(nn.Module):
    """
    SemanticToTextWrapper 类用于将语义表示转换为文本表示。
    该类封装了 TextToSemantic 模型，并将其用于从语义 token 到文本 token 的转换任务。
    
    参数说明:
        model (TextToSemantic): 预训练的 TextToSemantic 模型实例。
    """
    @beartype
    def __init__(
        self,
        model: TextToSemantic
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        semantic_token_ids,
        grapheme_token_ids,
    ):
        """
        前向传播方法，将语义 token 转换为文本 token。

        参数说明:
            semantic_token_ids: 输入的语义 token ID 张量。
            grapheme_token_ids: 目标文本 token ID 张量。

        返回:
            loss: 模型在转换任务中的损失。
            logits: 模型输出的逻辑值，用于进一步处理或预测。
        """
        # 源数据为语义 token ID
        source = semantic_token_ids
        # 目标数据为文本 token ID
        target = grapheme_token_ids

        loss, logits = self.model(
            source, target,
            source_type='speech',  # 源类型为语音（语义）
            target_type='text',    # 目标类型为文本
            return_loss=True,      # 返回损失
            return_logits=True     # 返回逻辑值
        )

        return loss, logits


# wrapper for text to semantic task

class TextToSemanticWrapper(nn.Module):
    """
    TextToSemanticWrapper 类用于将文本表示转换为语义表示。
    该类封装了 TextToSemantic 模型，并将其用于从文本 token 到语义 token 的转换任务。
    
    参数说明:
        model (TextToSemantic): 预训练的 TextToSemantic 模型实例。
    """
    @beartype
    def __init__(
        self,
        model: TextToSemantic
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        grapheme_token_ids,
        semantic_token_ids,
        return_early_exit_loss = True
    ):
        """
        前向传播方法，将文本 token 转换为语义 token。

        参数说明:
            grapheme_token_ids: 输入的文本 token ID 张量。
            semantic_token_ids: 目标语义 token ID 张量。
            return_early_exit_loss: 是否返回提前退出损失，默认为 True。

        返回:
            loss: 模型在转换任务中的损失。
            logits: 模型输出的逻辑值，用于进一步处理或预测。
        """
        source = grapheme_token_ids  # 源数据为文本 token ID
        target = semantic_token_ids  # 目标数据为语义 token ID

        loss, logits = self.model(
            source, target,
            source_type='text',        # 源类型为文本
            target_type='speech',      # 目标类型为语音（语义）
            return_loss=True,          # 返回损失
            return_logits=True,        # 返回逻辑值
            return_early_exit_loss=return_early_exit_loss  # 返回提前退出损失
        )

        return loss, logits


# wrapper for generating the pseudo-labelled audio to text dataset

class SemanticToTextDatasetGenerator(nn.Module):
    """
    SemanticToTextDatasetGenerator 类用于生成语义到文本的数据集。
    该类使用预训练的 TextToSemantic 模型，将音频数据转换为语义表示，再进一步转换为文本表示，
    并将生成的音频语义 ID 和文本 ID 对保存到指定文件夹中。

    参数说明:
        model: 预训练的 TextToSemantic 模型实例。
        dataset (Dataset): 数据集实例，包含音频数据。
        folder (str, 可选): 保存生成数据的文件夹路径，默认为 './generated-audio-text-pairs'。
        batch_size (int, 可选): 数据加载的批量大小，默认为4。
        delimiter_id (int, 可选): 分隔符 ID，用于分隔音频语义 ID 和文本 ID，默认为 -1。
        audio_pad_id (int, 可选): 音频数据的填充 ID，如果未指定，则不进行填充处理。
        text_pad_id (int, 可选): 文本数据的填充 ID，默认为0。
    """
    @beartype
    def __init__(
        self,
        model,
        *,
        dataset: Dataset,
        folder = './generated-audio-text-pairs',
        batch_size = 4,
        delimiter_id: int = -1,
        audio_pad_id = None,
        text_pad_id = 0
    ):
        super().__init__()
        self.model = model

        self.dataset = dataset
        # 获取数据加载器
        self.dl = get_dataloader(dataset, batch_size = batch_size)
        # 分隔符 ID
        self.delimiter_id = delimiter_id

        # 音频数据的填充 ID
        self.audio_pad_id = audio_pad_id
        # 文本数据的填充 ID
        self.text_pad_id = text_pad_id

        # 文件夹路径
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok = True, parents = True)

    def forward(
        self,
        max_length = 2048,
        beam_search_decode = True,
        **generate_kwargs
    ):
        """
        前向传播方法，执行语义到文本的生成任务，并将结果保存到指定文件夹中。

        参数说明:
            max_length: 生成序列的最大长度。
            beam_search_decode: 是否使用束搜索解码。
            **generate_kwargs: 其他生成参数。

        返回:
            None
        """
        # 创建分隔符张量
        delimiter = torch.tensor([self.delimiter_id], device = self.model.device)

        counter = 0
        # 遍历数据加载器中的音频数据
        for audio, in self.dl:
            # 使用模型生成音频语义 ID 和文本 ID
            audio_semantic_ids, text_ids = self.model.generate(
                source = audio,
                source_type = 'speech',
                target_type = 'text',
                return_source = True,
                max_length = max_length,
                beam_search_decode = beam_search_decode,
                **generate_kwargs
            )

            # 遍历生成的音频语义 ID 和文本 ID 对
            for audio_semantic_id, text_id in zip(audio_semantic_ids, text_ids):

                # 如果存在音频填充 ID，则移除填充部分
                if exists(self.audio_pad_id):
                    audio_pad_mask = audio_semantic_id == self.audio_pad_id
                    audio_semantic_id = audio_semantic_id[~audio_pad_mask]

                # 如果存在文本填充 ID，则移除填充部分
                if exists(self.text_pad_id):
                    text_pad_mask = text_id == self.text_pad_id
                    text_id = text_id[~text_pad_mask]

                # 将音频语义 ID、分隔符和文本 ID 打包成一个序列
                row, _ = pack([audio_semantic_id, delimiter, text_id], '*')
                path = str(self.folder / f'{counter}.pt')

                # 保存生成的序列到指定路径
                torch.save(row, path)
                counter += 1
