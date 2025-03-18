# predict.py
import torch
import numpy as np
import pandas as pd
import os
from model import GRUEncoder, MultiAgentGRUDecoder

def load_models(input_size, hidden_size, output_size, num_agents, model_path="save_models/"):
    encoder = GRUEncoder(input_size, hidden_size, num_heads=num_agents)
    decoder = MultiAgentGRUDecoder(hidden_size, output_size, num_agents)

    encoder.load_state_dict(torch.load(os.path.join(model_path, "encoder.pth")))
    decoder.load_state_dict(torch.load(os.path.join(model_path, "decoder.pth")))

    encoder.eval()
    decoder.eval()
    
    print("模型加載完成！")
    return encoder, decoder


def generate_routes(encoder, decoder, test_x, csv_path, output_dir="Output/"):
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(csv_path)  
    date_str = os.path.splitext(file_name)[0]
    date_folder = os.path.join(output_dir, date_str)
    os.makedirs(date_folder, exist_ok=True)  

    df_original = pd.read_csv(csv_path, encoding='utf-8')

    # 確保 test_x 存在
    if test_x is None or test_x.shape[0] == 0:
        print("⚠️ [錯誤] 無法產生預測結果, test_x 為空")
        return

    # 讀取車輛數量 (num_agents)
    num_agents = df_original["N_V"].max()
    print(f"今日車輛數量: {num_agents}")

    visited_stations = set()
    route_splits = [[df_original.iloc[0]] for _ in range(num_agents)]

    while len(visited_stations) < len(df_original):
        print(f"訪問站點數量: {len(visited_stations)}/{len(df_original)}")
        # 預測配送路徑
        encoded = encoder(test_x)
        agent_inputs = encoded.unsqueeze(1).expand(-1, num_agents, -1)  # (batch_size, num_agents, hidden_size*2)
        current_num_agents = agent_inputs.shape[1]
        predicted_routes = decoder(encoded, agent_inputs, current_num_agents).detach().cpu().numpy()

        mask = np.isin(np.arange(predicted_routes.shape[-1]), list(visited_stations))
        predicted_routes[:, :, mask] = -np.inf

        # 直接選擇機率最高的站點
        predicted_indices = np.argmax(predicted_routes, axis=-1).flatten()
        predicted_indices = np.clip(predicted_indices, 0, len(df_original) - 1)

        for agent_id, idx in enumerate(predicted_indices):
            if idx not in visited_stations:
                route_splits[agent_id].append(df_original.iloc[idx])
                visited_stations.add(idx)

    # 在每條路徑最後加入起點
    for route in route_splits:
        route.append(df_original.iloc[0])

    # 存儲不同車輛的 CSV
    csv_paths = []
    for agent_id, route_data in enumerate(route_splits):
        df_route = pd.DataFrame(route_data)
        output_path = os.path.join(output_dir, f"{date_str}/route_{date_str}_{agent_id+1}.csv")
        df_route.to_csv(output_path, index=False)
        csv_paths.append(output_path)
        print(f"✅ 車輛 {agent_id+1} 的配送路徑已儲存至 {output_path}")

    return csv_paths, date_folder


def run_prediction(csv_path, num_agents):
    input_size = 3
    hidden_size = num_agents * 16
    output_size = 2
    test_x = torch.randn(16, 20, input_size)  # 模擬輸入數據

    encoder, decoder = load_models(input_size, hidden_size, output_size, num_agents)

    encoded = encoder(test_x)
    agent_inputs = encoded.unsqueeze(1).expand(-1, num_agents, -1)  # (batch_size, num_agents, hidden_size*2)
    
    predicted_routes = decoder(encoded, agent_inputs).detach().numpy()
    
    generate_routes(encoder, decoder, test_x, csv_path)


def inverse_transform(predicted_x, predicted_y, X_min, X_max, Y_min, Y_max):
    original_x = predicted_x * (X_max - X_min) + X_min
    original_y = predicted_y * (Y_max - Y_min) + Y_min
    return original_x, original_y
