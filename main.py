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

csv_files = glob.glob('Data/district_data/*.csv')
print(f"找到 {len(csv_files)} 個 CSV 檔案：{csv_files}")

input_size = 6  
batch_size = 16
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# **依序處理每個 CSV**
for csv_path in csv_files:
    print(f"🚀 開始處理 {csv_path}...")

    # 讀取 CSV
    df = pd.read_csv(csv_path, encoding='utf-8')
    num_agents = df['N_V'].max()
    num_stations = len(df)
    hidden_size = num_agents * 16
    output_size = num_stations

    # **將數據轉換為 Tensor**
    test_x = df[['X', 'Y', 'W_S', 'W_F', 'Demand', 'Service_time']].values.reshape(-1, num_stations, input_size)
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)

    # 初始化環境
    environment = Environment(df, num_agents)

    print(f"🚀 開始訓練 {csv_path}...")
    train.train_model(
        test_x, test_x, 
        input_size, hidden_size, num_heads=num_agents, 
        epochs=epochs, lr=0.01,
        save_path='save_models/',
        environment=environment
    )

    # **載入訓練好的模型**
    encoder, decoder = predict.load_models(input_size, hidden_size, output_size, num_agents)

    # **執行預測，產生配送路線**
    csv_paths, output_folder = predict.generate_routes(encoder, decoder, test_x, csv_path)

    # **視覺化結果**
    visualize_routes(csv_paths, output_folder, os.path.basename(csv_path).split('.')[0])
    print(f"✅ {csv_path} 的路線已儲存\n")

print('Done')

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
