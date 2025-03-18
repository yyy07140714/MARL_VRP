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
        
        start_index = 0
        environment.visited_customers.add(start_index)

        store_progress = tqdm(total=len(environment.df), desc="📍 店家分配進度", unit="店")
        print(f"[DEBUG] is_done: {environment.is_done(state)}, 已訪問: {len(environment.visited_customers)}/{len(environment.df)-1}")

        while not environment.is_done(state):
            encoded_state = self.encoder(state)
            current_num_agents = len(environment.vehicle_positions)
            decoder_input = encoded_state.unsqueeze(1).repeat(1, current_num_agents, 1)
            agent_outputs = self.decoder(encoded_state, decoder_input, current_num_agents)

            # 避免選擇已訪問的站點
            valid_outputs = agent_outputs.clone()
            for i, visited_idx in enumerate(environment.visited_customers):
                valid_outputs[:, :, visited_idx] = float('-inf')

            selected_indices = torch.argmax(valid_outputs, dim=-1).cpu().numpy()

            new_positions = []
            for agent_id, selected_idx in enumerate(selected_indices[0]):  # 取 batch 內的第一個
                selected_station = environment.df.iloc[selected_idx]
                station_x, station_y = selected_station["X"], selected_station["Y"]
                print(f"[INFO] 車輛 {agent_id} 選擇站點: {selected_station['Name']} ({station_x}, {station_y})")
                new_positions.append([station_x, station_y])

            # ✅ 更新狀態並計算獎勵
            state, reward = environment.update_state(state, new_positions)
            total_reward += reward  

            store_progress.update(1)

        store_progress.close()
        return total_reward

    def reset(self):
        self.environment.visited_customers = set()  # ✅ 清空已訪問店家
        self.environment.vehicle_positions = [[0, 0] for _ in range(self.environment.num_vehicles)]  # ✅ 重置車輛位置







