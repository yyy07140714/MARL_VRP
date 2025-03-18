# main.py
import pandas as pd
import torch
import train
import predict
import os
from utils import visualize_routes
from management import ManagementModule
from environment import Environment
from model import GRUEncoder, MultiAgentGRUDecoder
import glob

# **讀取 CSV 數據**
csv_path = 'Data/daily_data/0301.csv'
df = pd.read_csv(csv_path, encoding='utf-8')
num_agents = df['N_V'].max()
print(f"今日車輛數量: {num_agents}")

num_stations = len(df)
input_size = 6  
hidden_size = num_agents * 16
output_size = num_stations
batch_size = 16
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# test data from csv
test_x = df[['X', 'Y', 'W_S', 'W_F', 'Demand', 'Service_time']].values.reshape(-1, num_stations, input_size)
test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
print('test_x', test_x)
# initialize environment and model
environment = Environment(df, num_agents)

print('Start training')
train.train_model(
    test_x, test_x, 
    input_size, hidden_size, num_heads=num_agents, 
    epochs=epochs, lr=0.01,
    save_path='save_models/',
    environment=environment
)

# load trained model
encoder, decoder = predict.load_models(input_size, hidden_size, output_size, num_agents)

# predict and generate routes csv
csv_paths, output_folder = predict.generate_routes(encoder, decoder, test_x, csv_path)

# csv_files = glob.glob('Output/0301/*.csv')
# print(f"🔍 找到 {len(csv_files)} 個 CSV 檔案: {csv_files}")
visualize_routes(csv_paths, output_folder, '0301')
print("路線已儲存")


# # **測試時不需要梯度計算**
# with torch.no_grad():
#     initial_state = torch.randn(batch_size, num_stations, input_size).to(device)
#     total_reward = management_module.run_episode(initial_state, environment, training=False)
#     print(f"Total Reward: {total_reward}")

# # **執行多次測試 episode**
# for episode in range(epochs):
#     with torch.no_grad():
#         initial_state = torch.randn(batch_size, num_stations, input_size).to(device)
#         total_reward = management_module.run_episode(initial_state, environment, training=False)
#         print(f"Episode {episode + 1}, Total Reward: {total_reward}")
