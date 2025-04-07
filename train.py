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
import random
from utils import plot_training_curves
import re

def train_model(train_x, train_y, input_size, hidden_size, num_heads, epochs=10, lr=0.001, save_path="save_models/", environment=None):
    if environment is None:
        raise ValueError("Environment is required for training.")

    start_x, start_y = environment.start_x, environment.start_y
    num_stations = len(environment.df)-1
    print(f"[DEBUG] 總店數 (num_stations): {num_stations}")

    # **初始化模型**
    encoder = GRUEncoder(input_size, hidden_size, num_heads=num_heads)
    decoder = MultiAgentGRUDecoder(hidden_size, num_stations, num_agents=num_heads)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    management_module = ManagementModule(encoder, decoder, optimizer, criterion, environment)

    epoch_losses = []
    epoch_rewards = []

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
            # print('state_loss',state_loss)
            # **梯度更新**
            state_loss.backward()
            loss = state_loss
            optimizer.step()

            batch_bar.set_postfix({"Batch": batch_idx, "Batch Reward": f"{total_reward:.4f}"})

        epoch_losses.append(loss.item())
        epoch_rewards.append(total_reward)
        # tqdm.write(f"Epoch [{epoch+1}/{epochs}], Total Reward: {total_reward:.4f}")
        progress_bar.set_description(f"Epoch {epoch+1}/{epochs} | Reward: {total_reward:.4f}")
        gc.collect()
        torch.cuda.empty_cache()
    print('訓練完成！')

    torch.save(encoder.state_dict(), os.path.join(save_path, "encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(save_path, "decoder.pth"))

    print(f"模型已儲存至 {save_path}")
    return encoder, management_module, decoder, epoch_losses, epoch_rewards


def get_latest_epoch(save_path):
    """從模型儲存目錄中找出最後一個 epoch 數"""
    pattern = re.compile(r'encoder_epoch(\d+)\.pth')
    epochs = []

    for fname in os.listdir(save_path):
        match = pattern.match(fname)
        if match:
            epochs.append(int(match.group(1)))

    return max(epochs) if epochs else 0


def train_model_multi_instance(instances, input_size, hidden_size, max_agents, epochs=10, lr=0.001, save_path="save_models/"):
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume_epoch = get_latest_epoch(save_path)
    if resume_epoch >= epochs:
        print(f"⚠️ 當前 resume_epoch={resume_epoch} 已達設定 epochs={epochs}，無需再訓練。")
        return None, None, None, [], []

    encoder = GRUEncoder(input_size, hidden_size, num_heads=max_agents).to(device)
    decoder = MultiAgentGRUDecoder(hidden_size, num_agents=max_agents).to(device)

    best_reward = float('-inf') 

    if resume_epoch > 0:
        encoder_path = os.path.join(save_path, f"encoder_epoch{resume_epoch}.pth")
        decoder_path = os.path.join(save_path, f"decoder_epoch{resume_epoch}.pth")
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))
        print(f"🔄 自動從 epoch {resume_epoch} 繼續訓練")

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    epoch_losses, epoch_rewards = [], []
    final_epoch = resume_epoch

    for epoch in range(resume_epoch, epochs):
        final_epoch = epoch
        total_loss, total_reward = 0, 0
        pbar = tqdm(instances, desc=f"Epoch {epoch+1}/{epochs}")

        for train_x, env in pbar:
            env.reset()
            management_module = ManagementModule(encoder, decoder, optimizer, criterion, env)

            initial_state = torch.tensor([
                [env.start_x, env.start_y, 0, 86400, 0, 0] for _ in range(env.num_vehicles)
            ], dtype=torch.float32).unsqueeze(0).to(train_x.device)

            optimizer.zero_grad()
            state_loss, reward = management_module.run_episode(initial_state, env, training=True)
            state_loss.backward()
            optimizer.step()

            total_loss += state_loss.item()
            total_reward += reward
            pbar.set_postfix({'Loss': f"{state_loss.item():.2f}", 'Reward': f"{reward:.2f}"})

        avg_loss = total_loss / len(instances)
        avg_reward = total_reward / len(instances)
        epoch_losses.append(avg_loss)
        epoch_rewards.append(avg_reward)

        print(f"📈 Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}")
        plot_training_curves(epoch_losses, epoch_rewards, save_path=f"Output/training_reward.png")

        # ✅ 刪除上一個 epoch 的模型檔案
        if epoch > 0:
            prev_e = epoch
            try:
                os.remove(os.path.join(save_path, f"encoder_epoch{prev_e}.pth"))
                os.remove(os.path.join(save_path, f"decoder_epoch{prev_e}.pth"))
            except FileNotFoundError:
                pass

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(encoder.state_dict(), os.path.join(save_path, f"best_encoder.pth"))
            torch.save(decoder.state_dict(), os.path.join(save_path, f"best_decoder.pth"))
            print(f"🌟 儲存最佳模型（Epoch {epoch+1}, Reward: {avg_reward:.2f}）")

        # ✅ 儲存目前 epoch 模型
        torch.save(encoder.state_dict(), os.path.join(save_path, f"encoder_epoch{epoch+1}.pth"))
        torch.save(decoder.state_dict(), os.path.join(save_path, f"decoder_epoch{epoch+1}.pth"))

    print(f"✅ 模型訓練完成並儲存最新 epoch={final_epoch+1} 至 {save_path}")
    return encoder, management_module, decoder, epoch_losses, epoch_rewards