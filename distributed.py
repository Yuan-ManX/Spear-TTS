import torch
from torch.autograd import Function
import torch.distributed as distributed

from einops import rearrange

from attend import exists


def all_gather_variable_dim(t, dim = 0, sizes = None):
    """
    在分布式环境中，对具有不同维度大小的张量进行 all_gather 操作。

    Args:
        t (torch.Tensor): 输入张量，形状为 (..., dim_size, ...)。
        dim (int, optional): 需要进行 gather 的维度。默认为 0。
        sizes (Optional[List[torch.Tensor]], optional): 如果已知每个进程在该维度的大小，可以传入 sizes 列表。默认为 None。

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor]]: 返回一个元组，包含 gather 后的张量和每个进程在该维度的大小列表。

    处理步骤:
        1. 获取当前设备、当前进程的 rank 和总进程数。
        2. 如果未提供 sizes，则收集每个进程在该维度的大小。
        3. 计算所有进程在该维度上的最大大小。
        4. 对输入张量进行填充，使其在该维度上达到最大大小。
        5. 对填充后的张量进行 all_gather 操作。
        6. 根据每个进程在该维度的大小，生成掩码以去除填充的冗余部分。
        7. 根据掩码提取有效的元素，拼接成最终的 gather 结果。
    """
    # 获取当前设备、当前进程的 rank 和总进程数
    device, rank, world_size = t.device, distributed.get_rank(), distributed.get_world_size()
    
    # 如果未提供 sizes，则收集每个进程在该维度的大小
    if not exists(sizes):
        # 获取当前张量在该维度的大小
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        # 为每个进程创建一个空的 size 张量
        sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
        # 收集所有进程的 size 信息
        distributed.all_gather(sizes, size)
        # 将收集到的 size 信息堆叠成一个张量
        sizes = torch.stack(sizes)

    # 计算所有进程在该维度上的最大大小
    max_size = sizes.amax().item()
    # 对输入张量进行填充，使其在该维度上达到最大大小
    padded_t = pad_dim_to(t, max_size, dim = dim)

    # 为每个进程创建一个空的张量，用于存储 gather 后的结果
    gathered_tensors = [torch.empty(padded_t.shape, device = device, dtype = padded_t.dtype) for i in range(world_size)]
    # 对填充后的张量进行 all_gather 操作
    distributed.all_gather(gathered_tensors, padded_t)

    # 将所有进程的 gather 结果拼接成一个张量
    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    # 生成一个序列张量，用于生成掩码
    seq = torch.arange(max_size, device = device)

    # 生成掩码，标记每个进程在该维度上的有效元素
    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    # 调整掩码形状以匹配 gather 结果
    mask = rearrange(mask, 'i j -> (i j)')

    # 生成索引张量，根据掩码提取有效的元素
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    # 根据索引提取有效的元素
    gathered_tensor = gathered_tensor.index_select(dim, indices)

    return gathered_tensor, sizes


class AllGather(Function):
    """
    AllGather 类是一个自定义的 Autograd 函数，用于在分布式环境中执行 all_gather 操作。

    该函数在前向传播时收集所有进程的输入张量，并在反向传播时将梯度分发回相应的进程。
    """
    @staticmethod
    def forward(ctx, x, dim, sizes):
        """
        前向传播函数，执行 all_gather 操作。

        Args:
            ctx (torch.autograd.function.Context): 上下文对象，用于在反向传播时存储信息。
            x (torch.Tensor): 输入张量，形状为 (..., dim_size, ...)。
            dim (int): 需要进行 gather 的维度。
            sizes (Optional[List[int]]): 如果已知每个进程在该维度的大小，可以传入 sizes 列表。

        Returns:
            Tuple[torch.Tensor, List[int]]: 返回一个元组，包含 gather 后的张量和每个进程在该维度的大小列表。
        """
        # 检查是否在分布式环境中初始化，并且总进程数大于 1
        is_dist = distributed.is_initialized() and distributed.get_world_size() > 1
        # 将分布式环境标志存储在上下文中
        ctx.is_dist = is_dist

        if not is_dist:
            # 如果不在分布式环境中，直接返回输入张量和 None
            return x, None

        # 如果不在分布式环境中，直接返回输入张量和 None
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        # 将每个进程在该维度的大小列表转换为列表并存储在上下文中
        ctx.batch_sizes = batch_sizes.tolist()

        # 将 gather 的维度存储在上下文中
        ctx.dim = dim
        # 返回 gather 后的张量和大小列表
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        """
        反向传播函数，将梯度分发回相应的进程。

        Args:
            ctx (torch.autograd.function.Context): 上下文对象，用于获取前向传播时存储的信息。
            grads (torch.Tensor): 反向传播传递回来的梯度张量。
            _ (torch.Tensor): 忽略的输入参数。

        Returns:
            Tuple[torch.Tensor, None, None]: 返回一个元组，包含分发回相应进程的梯度张量和两个 None 值。
        """
        if not ctx.is_dist:
            # 如果不在分布式环境中，直接返回梯度张量和两个 None
            return grads, None, None

        # 获取每个进程在该维度的大小和当前进程的 rank
        batch_sizes, rank = ctx.batch_sizes, distributed.get_rank()
        # 根据大小列表将梯度张量拆分回每个进程的梯度
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        # 返回当前进程的梯度张量和两个 None
        return grads_by_rank[rank], None, None


# 定义 all_gather 函数为 AllGather 类的 apply 方法的别名
all_gather = AllGather.apply
