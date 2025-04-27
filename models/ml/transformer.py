import torch
import torch.nn as nn
import torch.optim as optim

class TimeSeriesTransformer(nn.Module):
    """基于Transformer的时间序列预测模型"""
    
    def __init__(self, feature_dim, seq_len, num_heads=4, hidden_dim=128, num_layers=2):
        super(TimeSeriesTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 定义Transformer编码器
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )

        # 定义全连接层用于输出
        self.fc = nn.Linear(seq_len * feature_dim, 1)

    def forward(self, x):
        """前向传播"""
        # x shape: (batch_size, seq_len, feature_dim)
        x = x.permute(1, 0, 2)  # 调整为 (seq_len, batch_size, feature_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # 调整回 (batch_size, seq_len, feature_dim)
        x = self.fc(x[:, -1, :])  # 取最后一个时间步的输出
        return x

    def train_model(self, X_train, y_train, epochs=100, lr=0.001):
        """训练模型"""
        self.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, X):
        """预测未来走势"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(X)
        return predictions