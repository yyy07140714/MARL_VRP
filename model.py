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
        self.depot_embedding = nn.Parameter(torch.randn(1, 1, input_size))  # (batch_size, 1, input_size)

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, 
                          dropout=dropout if num_layers > 1 else 0, 
                          batch_first=True, bidirectional=bidirectional)
        
        self.output_dim = hidden_size * (2 if bidirectional else 1)

        # ✅ 修正 num_heads 確保可以整除
        if self.output_dim % num_heads != 0:
            print(f"[WARN] 調整 MultiheadAttention 的 num_heads: {num_heads} → 1，因為 embed_dim={self.output_dim} 無法整除")
            num_heads = 1

        self.mha = nn.MultiheadAttention(self.output_dim, num_heads=num_heads, batch_first=True)
        
        self.ff = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        device = x.device
        self.to(device)
        depot_embed = self.depot_embedding.to(device)

        batch_size = x.shape[0]
        depot_embed = depot_embed.expand(batch_size, -1, -1)
        x = torch.cat([depot_embed, x], dim=1)

        x, _ = self.gru(x)
        attn_output, _ = self.mha(x, x, x)
        x = self.layer_norm(x + attn_output)
        ff_output = self.ff(x)
        x = self.layer_norm(x + ff_output)
        x = self.dropout(x)

        return x[:, -1, :]


class MultiAgentGRUDecoder(nn.Module):
    def __init__(self, hidden_size, num_agents):
        super(MultiAgentGRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_agents = num_agents

        self.gru = nn.GRU(hidden_size * 2, hidden_size * 2, batch_first=True)
        self.decision_head = nn.Linear(hidden_size * 2, 128)  # Q 向量
        self.customer_embed = nn.Linear(2, 128)               # K 向量（位置）

        # ✅ 每個 agent 的偏好向量
        self.agent_embed = nn.Embedding(num_agents, 128)      # 128 維偏好向量（與 Q 向量維度一致）

    def forward(self, encoder_output, agent_inputs, current_num_agents, customer_positions):
        """
        encoder_output: (batch_size, hidden_size * 2)
        agent_inputs: (batch_size, num_agents, hidden_size * 2)
        customer_positions: (output_size, 2)
        """
        device = encoder_output.device
        batch_size = encoder_output.shape[0]
        output_size = customer_positions.shape[0]

        agent_outputs = []
        customer_positions = customer_positions.to(device)
        customer_embeds = self.customer_embed(customer_positions)  # (output_size, 128)

        # 🔄 Global Route Recorder：融合所有車輛狀態
        global_input = []
        for i in range(current_num_agents):
            agent_feat = agent_inputs[:, i, :]  # (batch, hidden*2)
            agent_pos = agent_feat[:, :2]
            agent_load = agent_feat[:, 4:5]
            global_input.append(torch.cat([agent_pos, agent_load], dim=-1))

        global_input = torch.cat(global_input, dim=-1).unsqueeze(1)  # (batch, 1, num_agents*3)
        input_dim = global_input.shape[-1]
        global_gru = nn.GRU(input_dim, self.hidden_size * 2, batch_first=True).to(device)
        _, global_hidden = global_gru(global_input)
        global_hidden = global_hidden.squeeze(0)  # (batch, hidden*2)

        # 🧠 每台車進行決策
        for i in range(current_num_agents):
            agent_input = agent_inputs[:, i, :].unsqueeze(1)  # (batch, 1, hidden*2)
            _, hidden = self.gru(agent_input)
            agent_hidden = hidden.squeeze(0)  # (batch, hidden*2)

            # ✅ 融合 global state + agent 偏好
            fused = agent_hidden + global_hidden
            agent_pref = self.agent_embed(torch.tensor(i, device=device)).unsqueeze(0).expand(batch_size, -1)
            fused = fused + agent_pref  # (batch, hidden*2)

            q_vector = self.decision_head(fused)  # (batch, 128)

            # 📌 Attention Score
            dk = q_vector.size(-1)
            scores = torch.matmul(q_vector, customer_embeds.T) / dk**0.5  # (batch, output_size)

            # 📏 加上距離懲罰
            current_pos = agent_input[:, 0, :2]  # (batch, 2)
            distances = torch.cdist(current_pos, customer_positions)
            scores = scores - distances / 1000.0

            # ❗ 避免過早回 depot
            if scores[:, 0].max() > 0.9:
                scores[:, 0] *= 0.5

            agent_outputs.append(scores)

        agent_outputs = torch.stack(agent_outputs, dim=1)  # (batch, num_agents, output_size)
        return agent_outputs
