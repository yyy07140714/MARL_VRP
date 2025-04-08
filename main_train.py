# main_train.py
import pandas as pd
import torch
import train
import predict
import os
import glob
from utils import visualize_routes, plot_training_curves
from management import ManagementModule
from environment import Environment
from model import GRUEncoder, MultiAgentGRUDecoder

INPUT_SIZE = 6
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def load_all_instances(folder_path):
    instances = []
    csv_paths = glob.glob(os.path.join(folder_path, "*.csv"))

    for csv_path in csv_paths:
        df_all = pd.read_csv(csv_path, encoding='utf-8')

        for date, df_day in df_all.groupby("Date"):
            # 先過濾掉包含 "ＤＣ" 的 row
            df_day = df_day[~df_day["Name"].str.contains("ＤＣ")].reset_index(drop=True)

            num_agents = df_day['N_V'].max()
            num_stations = len(df_day)
            test_x = df_day[['X', 'Y', 'W_S', 'W_F', 'Demand', 'Service_time']].values.reshape(-1, num_stations, INPUT_SIZE)
            test_x = torch.tensor(test_x, dtype=torch.float32).to(DEVICE)

            env = Environment(df_day, num_agents)
            os.makedirs("temp", exist_ok=True)
            env.df_path = f"temp/temp_{date}.csv"
            df_day.to_csv(env.df_path, index=False)

            instances.append((test_x, env))

    return instances


if __name__ == "__main__":
    all_instances = load_all_instances("Data/Daily Data/")
    print(f"✅ 載入 {len(all_instances)} 筆訓練資料")

    MAX_AGENTS = max(env.num_vehicles for _, env in all_instances)
    encoder, management_module, decoder, losses, rewards = train.train_model_multi_instance(
        instances=all_instances,
        input_size=INPUT_SIZE,
        hidden_size=MAX_AGENTS * 16,
        max_agents=MAX_AGENTS,
        epochs=EPOCHS,
        lr=0.001,
        save_path='save_models/'
    )

    # plot_training_curves(losses, rewards, save_path="Output/training_curve_month.png")

    print("🚀 開始預測每筆資料...")
    for test_x, env in all_instances:
        if test_x.dim() == 2:
            test_x = test_x.unsqueeze(0)

        num_agents = env.num_vehicles
        output_size = len(env.df) - 1
        encoder, decoder = predict.load_models(
            input_size=INPUT_SIZE,
            hidden_size=MAX_AGENTS * 16,
            output_size=output_size,
            num_agents=num_agents,
            best=True
        )

        csv_paths, output_folder = predict.generate_routes(encoder, decoder, test_x, env.df_path)
        visualize_routes(csv_paths, output_folder, f"route_{os.path.basename(env.df_path).split('.')[0]}")
        os.remove(env.df_path)

    print("🎉 全部資料處理完畢！")
