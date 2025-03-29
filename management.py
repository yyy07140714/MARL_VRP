# management.py
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

class ManagementModule:
    def __init__(self, encoder, decoder, optimizer, criterion, environment):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.criterion = criterion
        self.environment = environment

    def run_episode(self, initial_state, environment, training=True):
        environment.reset()
        state = initial_state
        total_reward = 0
        state_loss = 0
        routes = [[] for _ in range(environment.num_vehicles)]  # 記錄每輛車的路徑
        arrival_times = [[] for _ in range(environment.num_vehicles)]  # 記錄到達時間

        store_progress = tqdm(total=len(environment.df), desc="📍 店家分配進度", unit="店", leave=False)
        

        while not environment.is_done(state):
            # print(f"[DEBUG] is_done: {environment.is_done(state)}, 已訪問: {len(environment.visited_customers)}/{len(environment.df)-1}")
            encoded_state = self.encoder(state)
            current_num_agents = len(environment.vehicle_positions)
            decoder_input = encoded_state.unsqueeze(1).repeat(1, current_num_agents, 1)
            # 取所有配送點座標（包含 DC 的話就不用 iloc[1:]）
            customer_positions = torch.tensor(environment.df[1:][['X', 'Y']].values, dtype=torch.float32).to(encoded_state.device)
            
            agent_outputs = self.decoder(
                encoder_output=encoded_state,
                agent_inputs=decoder_input,
                current_num_agents=current_num_agents,
                customer_positions=customer_positions
            )

            # agent_outputs = self.decoder(encoded_state, decoder_input, current_num_agents)

            valid_outputs = agent_outputs.clone()

            # Mask 已訪問的點
            mask = torch.zeros_like(valid_outputs)
            for visited_idx in environment.visited_customers:
                valid_outputs[:, :, visited_idx] = 1

            valid_outputs = valid_outputs.masked_fill(mask.bool(), float('-inf'))

            # Fallback 修正：確保不是全為 -inf
            for b in range(valid_outputs.shape[0]):
                for a in range(valid_outputs.shape[1]):
                    if torch.all(valid_outputs[b, a] == float('-inf')):
                        valid_outputs[b, a] = torch.zeros_like(valid_outputs[b, a])  # 均勻初始化
                        print(f"[WARN] Agent {a} fallback to uniform output at batch {b}")

            # 最後再選 max
            if training:
                # 使用 softmax sampling
                probs = torch.softmax(valid_outputs, dim=-1)  # (1, num_agents, output_size)
                selected_indices = []

                for agent_probs in probs[0]:  # agent_probs: (output_size,)
                    sampled = torch.multinomial(agent_probs, num_samples=1)
                    selected_indices.append(sampled.item())
                selected_indices = [selected_indices]  # 為了維持 shape (1, num_agents)
            else:
                # 測試或推論：argmax
                selected_indices = torch.argmax(valid_outputs, dim=-1).cpu().numpy()

            new_positions = []

            for agent_id, selected_idx in enumerate(selected_indices[0]):
                selected_station = environment.df.iloc[selected_idx]
                station_x, station_y = selected_station["X"], selected_station["Y"]
                # print(f"[INFO] 車輛 {agent_id} 選擇站點: {selected_station['Name']} ({station_x}, {station_y})")
                new_positions.append([station_x, station_y])

                # 記錄行駛路徑與到達時間
                routes[agent_id].append((station_x, station_y))
                arrival_times[agent_id].append(environment.get_arrival_time([station_x, station_y], agent_id))

            state, _, visited_indices = environment.update_state(state, new_positions)
            
            # total_reward += reward

            store_progress.update(1)

        store_progress.close()

        # **計算最終總成本**
        total_cost = environment.calculate_total_cost(routes, arrival_times, visited_indices)
        final_reward = (- total_cost)  # reward 是負 cost（最小化成本）
        state_loss = torch.tensor(-final_reward, requires_grad=True).to(state.device)  # 當作 loss 使用

        # tqdm.write(f"[INFO] 總成本: {total_cost:.2f}, 最終獎勵: {final_reward:.2f}")

        return state_loss, final_reward

    def reset(self):
        self.environment.visited_customers = set()  
        self.environment.vehicle_positions = [[0, 0] for _ in range(self.environment.num_vehicles)]  

