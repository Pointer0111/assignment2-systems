import torch
import torch.distributed as dist
from typing import Dict, List, Any, Optional, Callable

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
        # 首先收集所有参数
        all_params = []
        if hasattr(params, '__iter__'):
            for param_group in params:
                if isinstance(param_group, dict):
                    all_params.extend(param_group['params'])
                else:
                    all_params.append(param_group)
        else:
            all_params = list(params)
        
        # 获取分布式信息
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        
        # 将参数分片到当前rank
        self.all_params = all_params
        self.param_to_rank = {}
        self.local_params = []
        
        # 为每个参数分配rank
        for i, param in enumerate(all_params):
            assigned_rank = i % self.world_size
            self.param_to_rank[param] = assigned_rank
            if assigned_rank == self.rank:
                self.local_params.append(param)
        
        # 创建本地优化器，只优化分配给当前rank的参数
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.local_optimizer = optimizer_cls(self.local_params, **kwargs)
        
        # 调用父类构造函数 - 需要传入参数组格式
        # 我们传入所有参数，但实际只优化分配给当前rank的参数
        param_groups = [{'params': all_params}]
        super().__init__(param_groups, {})

    def step(self, closure=None, **kwargs):
        """
        调用被包装优化器的 step() 方法，并在参数更新后与其他 rank 同步。
        """
        # 执行本地优化器的step
        loss = None
        if closure is not None:
            loss = self.local_optimizer.step(closure, **kwargs)
        else:
            self.local_optimizer.step(**kwargs)
        
        # 同步所有参数
        if dist.is_initialized() and self.world_size > 1:
            self._synchronize_parameters()
        
        return loss
    
    def _synchronize_parameters(self):
        """在所有rank之间同步参数"""
        for param in self.all_params:
            responsible_rank = self.param_to_rank[param]
            # 从负责该参数的rank广播参数到所有其他rank
            dist.broadcast(param.data, src=responsible_rank)

    def add_param_group(self, param_group):
        """
        向分片优化器添加参数组。
        该方法在分片优化器构造期间由父类构造函数调用，
        也可能在训练过程中调用（如逐步解冻模型层）。
        需要处理参数在各 rank 之间的分配。
        """
        # 提取参数
        if isinstance(param_group, dict):
            new_params = param_group['params']
            param_group_dict = param_group
        else:
            new_params = param_group
            param_group_dict = {'params': new_params}
        
        # 为新参数分配rank
        local_new_params = []
        for param in new_params:
            # 使用当前总参数数量来分配rank，确保均匀分布
            assigned_rank = len(self.all_params) % self.world_size
            self.param_to_rank[param] = assigned_rank
            self.all_params.append(param)
            
            if assigned_rank == self.rank:
                local_new_params.append(param)
        
        # 如果有分配给当前rank的新参数，添加到本地优化器
        if local_new_params:
            local_param_group = param_group_dict.copy()
            local_param_group['params'] = local_new_params
            self.local_optimizer.add_param_group(local_param_group)
    
    def zero_grad(self, set_to_none: bool = True):
        """清零梯度"""
        self.local_optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        """返回优化器状态字典"""
        # 收集所有rank的状态
        local_state = self.local_optimizer.state_dict()
        
        if not dist.is_initialized() or self.world_size == 1:
            return local_state
        
        # 在分布式环境中，需要收集所有rank的状态
        # 这里简化实现，只返回本地状态
        # 实际应用中可能需要更复杂的状态合并逻辑
        return local_state
    
    def load_state_dict(self, state_dict):
        """加载优化器状态字典"""
        self.local_optimizer.load_state_dict(state_dict)
    
    @property
    def param_groups(self):
        """返回参数组"""
        return self.local_optimizer.param_groups
    
    @param_groups.setter
    def param_groups(self, value):
        """设置参数组"""
        self.local_optimizer.param_groups = value