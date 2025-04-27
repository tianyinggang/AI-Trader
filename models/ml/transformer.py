import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TimeSeriesTransformer(nn.Module):
    """基于Transformer的时间序列预测模型"""
    def __init__(self, feature_dim, seq_len, num_heads=4, hidden_dim=128, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 增加位置编码以帮助模型理解序列中不同位置的关系
        self.positional_encoding = self._create_positional_encoding(seq_len, feature_dim)
        
        # 添加输入投影层，将feature_dim映射到更高维度，增强表达能力
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        
        # 更新Transformer编码器配置
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,  # 使用hidden_dim而不是feature_dim
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # 扩大feedforward层
            dropout=dropout,     # 添加dropout防止过拟合
            batch_first=True     # 设置batch_first更符合习惯
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        
        # 使用注意力机制聚合序列信息而不是简单展平
        self.attention = nn.Linear(hidden_dim, 1)
        
        # 更复杂的输出层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
        # 初始化参数
        self._init_weights()
    
    def _create_positional_encoding(self, seq_len, d_model):
        # 创建位置编码
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:d_model//2]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        # 注册为缓冲区，不参与梯度更新
        return nn.Parameter(pe, requires_grad=False)
    
    def _init_weights(self):
        # 初始化参数，提高收敛速度和性能
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # 输入形状: [batch_size, seq_len, feature_dim]
        batch_size = x.size(0)
        
        # 确保输入数据形状正确
        if x.dim() == 2:
            x = x.view(batch_size, self.seq_len, self.feature_dim)
        
        # 添加位置编码
        pos_enc = self.positional_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        if x.size(-1) == self.positional_encoding.size(-1):
            x = x + pos_enc.to(x.device)
        
        # 投影到更高维度
        x = self.input_projection(x)
        
        # 应用Transformer编码器
        x = self.transformer_encoder(x)
        
        # 使用注意力机制聚合序列信息
        attn_weights = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(x * attn_weights, dim=1)
        
        # 应用输出层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def train_epoch(self, X_train, y_train, optimizer, batch_size=32):
        """训练一个epoch"""
        self.train()  # 设置为训练模式
        total_loss = 0.0
        correct_preds = 0
        
        # 将numpy数组转换为torch张量(如果需要)
        if isinstance(X_train, np.ndarray):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if isinstance(y_train, np.ndarray):
            y_train = torch.tensor(y_train, dtype=torch.float32)
        
        # 创建数据加载器以处理批量
        indices = torch.randperm(len(X_train))
        num_batches = (len(X_train) + batch_size - 1) // batch_size
        
        criterion = nn.MSELoss()
        
        for i in range(num_batches):
            # 获取批次数据
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            batch_indices = indices[start_idx:end_idx]
            
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices].view(-1, 1)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self(batch_X)
            
            # 计算损失
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # 更新权重
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item() * len(batch_indices)
            
            # 计算方向准确率
            correct_preds += torch.sum((torch.sign(outputs) == torch.sign(batch_y)).float()).item()
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(X_train)
        accuracy = correct_preds / len(X_train)
        
        return avg_loss, accuracy
    
    def validate(self, X_val, y_val, batch_size=32):
        """在验证集上验证模型"""
        self.eval()  # 设置为评估模式
        total_loss = 0.0
        correct_preds = 0
        
        # 将numpy数组转换为torch张量(如果需要)
        if isinstance(X_val, np.ndarray):
            X_val = torch.tensor(X_val, dtype=torch.float32)
        if isinstance(y_val, np.ndarray):
            y_val = torch.tensor(y_val, dtype=torch.float32)
        
        # 创建数据加载器以处理批量
        num_batches = (len(X_val) + batch_size - 1) // batch_size
        
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for i in range(num_batches):
                # 获取批次数据
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_val))
                
                batch_X = X_val[start_idx:end_idx]
                batch_y = y_val[start_idx:end_idx].view(-1, 1)
                
                # 前向传播
                outputs = self(batch_X)
                
                # 计算损失
                loss = criterion(outputs, batch_y)
                
                # 累计损失
                total_loss += loss.item() * (end_idx - start_idx)
                
                # 计算方向准确率
                correct_preds += torch.sum((torch.sign(outputs) == torch.sign(batch_y)).float()).item()
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(X_val)
        accuracy = correct_preds / len(X_val)
        
        return avg_loss, accuracy
    
    def predict(self, X_test):
        """预测函数"""
        self.eval()  # 设置为评估模式
        
        # 将numpy数组转换为torch张量(如果需要)
        if isinstance(X_test, np.ndarray):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self(X_test)
            
        # 转换回numpy数组
        if isinstance(outputs, torch.Tensor):
            predictions = outputs.cpu().numpy().flatten()
        else:
            predictions = outputs
            
        return predictions