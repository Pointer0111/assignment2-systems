import torch

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        """
        初始化分片优化器。params 是需要优化的参数集合（或参数组），
        可以为模型不同部分使用不同的超参数（如学习率）。
        这些参数会在所有 rank 之间进行分片。
        optimizer_cls 指定要包装的优化器类型（如 torch.optim.AdamW）。
        其余关键字参数会传递给 optimizer_cls 的构造函数。
        需要调用 torch.optim.Optimizer 的父类构造函数。
        """
        super().__init__(params, defaults={})
        self.optimizer = optimizer_cls(self.param_groups, **kwargs)
        # TODO: 实现参数分片逻辑

    def step(self, closure=None, **kwargs):
        """
        调用被包装优化器的 step() 方法，并在参数更新后与其他 rank 同步。
        """
        loss = self.optimizer.step(closure=closure, **kwargs)
        # TODO: 实现参数同步逻辑
        return loss

    def add_param_group(self, param_group):
        """
        向分片优化器添加参数组。
        该方法在分片优化器构造期间由父类构造函数调用，
        也可能在训练过程中调用（如逐步解冻模型层）。
        需要处理参数在各 rank 之间的分配。
        """
        self.optimizer.add_param_group(param_group)
        # TODO: 实现参数组分片逻辑