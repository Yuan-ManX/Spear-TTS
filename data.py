from pathlib import Path
import torch
from torch.utils.data import Dataset
from beartype import beartype


# MockDataset 类用于生成一个模拟的数据集
class MockDataset(Dataset):
    """
    MockDataset 是一个模拟的数据集类，用于生成指定长度的随机张量数据。

    Args:
        length (int): 数据集的长度，即样本的数量。
    """
    def __init__(self, length: int):
        """
        初始化 MockDataset 实例。

        Args:
            length (int): 数据集的长度，即样本的数量。
        """
        # 设置数据集的长度
        self.length = length

    def __len__(self):
        """
        返回数据集的长度。

        Returns:
            int: 数据集的长度。
        """
        # 返回数据集的长度
        return self.length

    def __getitem__(self, ind):
        """
        根据索引获取数据集中的一个样本。

        Args:
            ind (int): 样本的索引。

        Returns:
            torch.Tensor: 形状为 (1024,) 的随机张量。
        """
        # 返回一个形状为 (1024,) 的随机张量
        return torch.randn(1024)


# generated audio-text dataset
# GeneratedAudioTextDataset 类用于处理生成的音频-文本数据集
class GeneratedAudioTextDataset(Dataset):
    """
    GeneratedAudioTextDataset 类用于加载和处理生成的音频-文本数据。

    数据集文件夹中包含多个 .pt 文件，每个文件包含一个音频-文本张量。
    张量中包含音频特征和文本特征，通过一个分隔符 ID 进行分隔。

    Args:
        folder (str): 数据集文件夹的路径。
        delimiter_id (int, optional): 分隔符 ID，用于分隔音频和文本。默认为 -1。
    """
    @beartype
    def __init__(
        self,
        folder: str,
        delimiter_id: int = -1
    ):
        """
        初始化 GeneratedAudioTextDataset 实例。

        Args:
            folder (str): 数据集文件夹的路径。
            delimiter_id (int, optional): 分隔符 ID，用于分隔音频和文本。默认为 -1。
        """
        # 将文件夹路径转换为 Path 对象
        self.folder = Path(folder)
        assert self.folder.exists() and self.folder.is_dir()
        # 获取文件夹中所有 .pt 文件的路径列表
        self.paths = list(self.folder.glob('*.pt'))
        # 设置分隔符 ID
        self.delimiter_id = delimiter_id

    def __len__(self):
        """
        返回数据集的长度，即 .pt 文件的数量。

        Returns:
            int: 数据集的长度。
        """
        return len(self.paths)

    def __getitem__(self, ind):
        """
        根据索引获取数据集中的一个样本。

        Args:
            ind (int): 样本的索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 返回一个元组，包含音频张量和文本张量。
        """
        # 获取当前索引对应的文件路径
        path = self.paths[ind]
        # 加载 .pt 文件中的张量数据
        tensor = torch.load(str(path))

        # 创建分隔符掩码，标记分隔符 ID 的位置
        delimiter_mask = tensor == self.delimiter_id
        assert delimiter_mask.any(), f'delimeter (<audio> <delimeter> <text>) not found'

        # 计算分隔符的位置
        # 通过累积求和找到第一个分隔符的位置
        ind = (delimiter_mask.cumsum(dim = -1) == 0).sum().item()

        # 返回音频张量和文本张量
        # 假设分隔符前的部分是音频，后面的部分是文本
        return tensor[:ind], tensor[(ind + 1):]
