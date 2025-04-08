class BaseRLAgent:
    """所有RL代理的基类，定义统一接口"""
    
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
    
    def act(self, state):
        """根据状态选择动作"""
        raise NotImplementedError
    
    def learn(self, experiences):
        """从经验中学习"""
        raise NotImplementedError
    
    def save(self, path):
        """保存模型"""
        raise NotImplementedError
    
    def load(self, path):
        """加载模型"""
        raise NotImplementedError