import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


# 定义TCN层
class TCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super(TCNLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size - 1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size - 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        return out


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Linear(input_dim, input_dim)
        self.input_dim = input_dim

    def forward(self, src):
        src = self.encoder(src) * np.sqrt(self.input_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


# 结合TCN和Transformer的两阶段模型
class TCNTransformer(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, transformer_input_dim, num_heads, transformer_layers,
                 dim_feedforward, dropout):
        super(TCNTransformer, self).__init__()
        self.tcn = TCNLayer(num_inputs, num_channels, kernel_size, dropout)
        self.transformer = TransformerModel(transformer_input_dim, num_heads, transformer_layers, dim_feedforward,
                                            dropout)
        self.decoder = nn.Linear(transformer_input_dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # TCN expects channel as the second dimension
        x = self.tcn(x)
        x = x.transpose(1, 2)  # Transformer expects sequence as the second dimension
        x = self.transformer(x)
        x = self.decoder(x)
        return x


# Hyperparameters
num_inputs = 1
num_channels = 64
kernel_size = 3
transformer_input_dim = 64
num_heads = 2
transformer_layers = 2
dim_feedforward = 256
dropout = 0.1

# 实例化模型
model = TCNTransformer(num_inputs, num_channels, kernel_size, transformer_input_dim, num_heads, transformer_layers, dim_feedforward, dropout)

# 生成随机数据
input_sequence = torch.rand(32, 10, 1)  # (batch_size, sequence_length, features)
target_sequence = torch.rand(32, 10, 1)  # 同上，这里是模拟数据

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
model.train()
for epoch in range(10):  # 10个epochs作为演示
    optimizer.zero_grad()
    output_sequence = model(input_sequence)
    loss = criterion(output_sequence, target_sequence)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')