import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        

class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=False):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, 
                          dropout=dropout if num_layers>1 else 0, 
                          batch_first=True, bidirectional=bidirectional)
        
        self.output_dim = hidden_size * (2 if bidirectional else 1)
    
    def forward(self, x):
        x, _ = self.gru(x)
        return x[:, -1, :]  # 取最後時間步的隱藏狀態
    

class StrategyModule(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(StrategyModule, self).__init__()
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        return self.fc(x)  # 輸出下一個配送點的 (X, Y)