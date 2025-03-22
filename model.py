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
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_heads = num_heads
        self.depot_embedding = nn.Parameter(torch.randn(1, 1, input_size))# (batch_size, 1, input_size)
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
        # 取得 x 的 device
        device = x.device  
        # print(f"[DEBUG] x.device: {device}")

        # 確保 self.depot_embedding 在相同 device
        self.to(device)  
        depot_embed = self.depot_embedding.to(device)
        
        # print(f"[DEBUG] depot_embed.device: {depot_embed.device}")
        # 讓 depot_embedding 的 batch_size 對齊
        batch_size = x.shape[0]
        depot_embed = depot_embed.expand(batch_size, -1, -1)

        # print(f"[DEBUG] Before concat, x.shape: {x.shape}, depot_embed.shape: {depot_embed.shape}")

        # 合併 depot_embedding 和 x
        x = torch.cat([depot_embed, x], dim=1) 

        # print(f"[DEBUG] After concat, x.shape: {x.shape}, x.device: {x.device}")

        # 確保 x 仍然在正確的 device
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
            nn.Linear(hidden_size * 2, output_size)
            for _ in range(num_agents)
        ])


    def forward(self, encoder_output, agent_inputs, current_num_agents, customer_positions=None):
        """
        encoder_output: (batch_size, hidden_size * 2)
        agent_inputs: (batch_size, num_agents, hidden_size * 2)
        customer_positions: (output_size, 2) 各個配送點 (X, Y)，需額外傳入
        """
        device = encoder_output.device
        self.to(device)
        batch_size = encoder_output.shape[0]
        agent_outputs = []

        for i in range(current_num_agents):
            print('current_num_agents', current_num_agents)

            agent_input = agent_inputs[:, i, :].unsqueeze(1)  # shape: (batch_size, 1, hidden_size*2)
            _, hidden_state = self.gru(agent_input)  # hidden_state: (1, batch, hidden_size*2)
            selected_customer = self.agent_decision_heads[i](hidden_state.squeeze(0))  # shape: (batch_size, output_size)

            # --------------------------
            # ✅ 加入距離懲罰 bias
            # --------------------------
            if customer_positions is not None:
                # 取出 agent 當前位置（假設為 agent_input 的前兩維為 (X, Y)）
                current_pos = agent_input[:, 0, :2]  # shape: (batch_size, 2)

                # 計算距離 (對所有 customers)
                distances = torch.cdist(current_pos, customer_positions.to(device))  # shape: (batch_size, output_size)
                distance_penalty = distances / 1000  # 調整懲罰強度
                selected_customer = selected_customer - distance_penalty  # 距離越遠越不想去

            # 防止過早回倉庫
            if selected_customer[:, 0].max() > 0.9:
                selected_customer[:, 0] *= 0.5

            agent_outputs.append(selected_customer)

        agent_outputs = torch.stack(agent_outputs, dim=1)  # shape: (batch_size, num_agents, output_size)
        return agent_outputs


