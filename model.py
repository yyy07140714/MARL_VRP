# model.py
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, hidden_size)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        # 加權求和
        weighted = torch.sum(attention_weights * x, dim=1)
        return weighted
    

class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=2, dropout=0.2, bidirectional=True, num_heads=3):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, 
                          dropout=dropout if num_layers > 1 else 0, 
                          batch_first=True, bidirectional=bidirectional)
        
        self.output_dim = hidden_size * (2 if bidirectional else 1)
        self.mha = nn.MultiheadAttention(self.output_dim, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim))
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch_size, seq_length, input_size)
        """
        device = x.device
        self.to(device)
        x, _ = self.gru(x)
        attn_output, _ = self.mha(x, x, x)
        x = self.layer_norm(x + attn_output)
        ff_output = self.ff(x)
        x = self.layer_norm(x + ff_output)
        x = self.dropout(x)
        return x[:, -1, :]  # 取最後一個時間步的輸出


class MultiAgentGRUDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_agents):
        super(MultiAgentGRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_agents = num_agents

        # 每個 Agent 共享一層 GRU
        self.gru = nn.GRU(hidden_size * 2, hidden_size * 2, batch_first=True)

        # 每個 Agent 獨立選擇目標客戶
        self.agent_decision_heads = nn.ModuleList([
            nn.Linear(hidden_size * 2, output_size) for _ in range(num_agents)
        ])

    def forward(self, encoder_output, agent_inputs, current_num_agents):
        """
        encoder_output: (batch_size, hidden_size * 2) 來自 Encoder
        agent_inputs: (batch_size, num_agents, hidden_size * 2) 每個 Agent 的當前狀態
        """
        device = encoder_output.device
        self.to(device)
        batch_size = encoder_output.shape[0]
        agent_outputs = []

        # 為每個 Agent 獨立計算決策
        for i in range(current_num_agents):
            agent_input = agent_inputs[:, i, :].unsqueeze(1)
            _, hidden_state = self.gru(agent_input)
            selected_customer = self.agent_decision_heads[i](hidden_state.squeeze(0))
            agent_outputs.append(selected_customer)

        # 將所有 Agent 的決策輸出整合
        agent_outputs = torch.stack(agent_outputs, dim=1)  # (batch_size, num_agents, output_size)
        return agent_outputs
