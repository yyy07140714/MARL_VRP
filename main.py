import pandas as pd
import torch
import train
import predict
import os
from utils import visualize_routes

# 讀取 CSV 數據
csv_path = 'Data/daily_data/0301.csv'  # 請替換為實際的 CSV 路徑
df = pd.read_csv(csv_path, encoding='utf-8')
file_name = os.path.basename(csv_path)
date_str = os.path.splitext(file_name)[0]
X_min, X_max = df["X"].min(), df["X"].max()
Y_min, Y_max = df["Y"].min(), df["Y"].max()
df["X"] = (df["X"] - X_min) / (X_max - X_min)
df["Y"] = (df["Y"] - Y_min) / (Y_max - Y_min)
# 取得當天的車輛數量
num_agents = df['N_V'].max()  # 取最大值作為車輛數量
print(f"今日車輛數量: {num_agents}")

# 設定 GRU 參數
input_size = 3  # (X, Y, Service_time)
hidden_size = 50
output_size = 2  # (預測下一個配送點的 X, Y)
seq_length = 20
batch_size = 16

# 生成模擬數據
train_x = torch.randn(batch_size, seq_length, input_size)  # 模擬歷史軌跡
train_y = torch.randn(batch_size, output_size)  # 預測下一個配送點

# 訓練模型並存檔
encoder, strategy_module = train.train_model(train_x, train_y, input_size, hidden_size, output_size, epochs=100, lr=0.001)

# 預測每日最佳路徑並存檔
predict.run_prediction(csv_path)

# 繪製路線圖
csv_files = [
    "Output/route_0301_1.csv",
    "Output/route_0301_2.csv",
    "Output/route_0301_3.csv"
]

visualize_routes(csv_files, save_dir="Output/", date_str=date_str)