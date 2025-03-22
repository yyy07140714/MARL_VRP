# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import os
from model import GRUEncoder, MultiAgentGRUDecoder
from management import ManagementModule
from tqdm import tqdm
import numpy as np
import gc

def train_model(train_x, train_y, input_size, hidden_size, num_heads, epochs=10, lr=0.001, save_path="save_models/", environment=None):
    if environment is None:
        raise ValueError("Environment is required for training.")

    start_x, start_y = environment.start_x, environment.start_y
    num_stations = len(environment.df)
    print(f"[DEBUG] 總店數 (num_stations): {num_stations}")

    # **初始化模型**
    encoder = GRUEncoder(input_size, hidden_size, num_heads=num_heads)
    decoder = MultiAgentGRUDecoder(hidden_size, num_stations, num_agents=num_heads)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    management_module = ManagementModule(encoder, decoder, optimizer, criterion, environment)

    progress_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
    for epoch in progress_bar:
        batch_bar = tqdm(range(len(train_x)), desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
        
        for batch_idx in batch_bar:
            optimizer.zero_grad()  # **清空梯度**
            
            # **初始化狀態**
            initial_state = []
            for vehicle_pos in environment.vehicle_positions:
                initial_state.append([
                    vehicle_pos[0],  # X
                    vehicle_pos[1],  # Y
                    0, 86400,  # 預設時間窗
                    0,  # 預設需求
                    environment.current_time  # 當前時間
                ])

            initial_state = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0).to(train_x.device)
            # print(f"[DEBUG] initial_state shape: {initial_state.shape}")

            # **執行 episode 訓練**
            state_loss, total_reward = management_module.run_episode(initial_state, environment, training=True)
            print('state_loss',state_loss)
            # **梯度更新**
            loss = -state_loss
            loss.backward()
            optimizer.step()

            batch_bar.set_postfix({"Batch": batch_idx, "Batch Reward": f"{total_reward:.4f}"})

        print(f"Epoch [{epoch+1}/{epochs}], Total Reward: {total_reward:.4f}")
        progress_bar.set_postfix({"Epoch Reward": f"{total_reward:.4f}"})
        gc.collect()
        torch.cuda.empty_cache()
    print('訓練完成！')

    torch.save(encoder.state_dict(), os.path.join(save_path, "encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(save_path, "decoder.pth"))

    print(f"模型已儲存至 {save_path}")
    return encoder, management_module, decoder

