from typing import Callable, Optional, Tuple, Union
import torch
from torch import Tensor
# from torch.nn import Parameter
from torch_geometric.utils import scatter, softmax
# from ...utils.num_nodes import maybe_num_nodes
# from ..inits import uniform
###该文件仅用于学习原理


def topk(x: Tensor, ratio: Optional[Union[float, int]], batch: Tensor, min_score: Optional[float] = None,
    tol: float = 1e-7,) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)

    elif ratio is not None:
        # 向量中每个元素表示子图中节点个数
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())
        # 节点累积向量，第k个元素表示前k-1个子图的总节点数
        cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        # 统一每张子图中节点数，并将其节点编码映射到相应的编码空间
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ), -60000.0)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)
        # 对每张子图中节点的分数按降序排
        _, perm = dense_x.sort(dim=-1, descending=True)
        # 将每张子图中的节点编码映射回最开始的编码空间，向量中可能存在相同编码。因为两个向量出现在不同子图中不影响选每张子图得分最高的前k个节点
        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
            k = torch.min(k, num_nodes)
        else:
            # 选取每张子图的前k个节点
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        if isinstance(ratio, int) and (k == ratio).all():
            # If all graphs have exactly `ratio` or more than `ratio` entries,
            # we can just pick the first entries in `perm` batch-wise:
            index = torch.arange(batch_size, device=x.device) * max_num_nodes
            index = index.view(-1, 1).repeat(1, ratio).view(-1)
            index += torch.arange(ratio, device=x.device).repeat(batch_size)
        else:
            # Otherwise, compute indices per graph:
            index = torch.cat([
                torch.arange(k[i], device=x.device) + i * max_num_nodes
                for i in range(batch_size)
            ], dim=0)

        perm = perm[index]

    else:
        raise ValueError("At least one of 'min_score' and 'ratio' parameters "
                         "must be specified")

    return perm
