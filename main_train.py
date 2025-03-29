# main_train.py
import pandas as pd
import torch
import train
import os
import glob
from environment import Environment
from predict import load_models, generate_routes


MAX_AGENTS = 15  # 所有資料中車輛數最大值
INPUT_SIZE = 6
BATCH_SIZE = 1  # 每個 instance 是一整個問題
EPOCHS = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_all_data(data_folder, input_size, device):
    instances = []
    csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
    print(f"📂 找到 {len(csv_files)} 個 CSV 檔案")
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, encoding='utf-8')
        num_agents = df['N_V'].max()
        num_stations = len(df)
        test_x = df[['X', 'Y', 'W_S', 'W_F', 'Demand', 'Service_time']].values.reshape(-1, num_stations, input_size)
        test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
        env = Environment(df, num_agents)
        instances.append((test_x, env))
        env.df_path = csv_path
    return instances

if __name__ == "__main__":
    all_instances = load_all_data("Data/district_data/", INPUT_SIZE, DEVICE)
    print(f"✅ 載入 {len(all_instances)} 筆訓練資料")

    encoder, management_module, decoder, losses, rewards = train.train_model_multi_instance(
        instances=all_instances,
        input_size=INPUT_SIZE,
        hidden_size=MAX_AGENTS * 16,
        max_agents=MAX_AGENTS,
        epochs=EPOCHS,
        lr=0.001,
        save_path='save_models/'
    )

    print("🚀 開始預測測試資料...")

    for test_x, env in all_instances:
        csv_path = env.df_path if hasattr(env, 'df_path') else '預設.csv'

        if test_x.dim() == 2:
            test_x = test_x.unsqueeze(0)

        # 重新載入模型（可跳過，但保證是存檔的版本）
        num_agents = env.num_vehicles
        output_size = len(env.df) - 1
        encoder, decoder = load_models(
            input_size=INPUT_SIZE,
            hidden_size=MAX_AGENTS * 16,
            output_size=output_size,
            num_agents=num_agents,
            model_path='save_models/'
        )

        # 執行路徑預測
        csv_paths, output_folder = generate_routes(encoder, decoder, test_x, csv_path)
        print(f"✅ 路徑輸出完成於：{output_folder}")
