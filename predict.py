# predict.py
import torch
import numpy as np
import pandas as pd
import os
from model import GRUEncoder, MultiAgentGRUDecoder

def load_models(input_size, hidden_size, output_size, num_agents, model_path="save_models/", best=False):
    encoder = GRUEncoder(input_size, hidden_size, num_heads=num_agents)
    decoder = MultiAgentGRUDecoder(hidden_size, num_agents)

    encoder.load_state_dict(torch.load(os.path.join(model_path, "best_encoder.pth")))

    # 🔁 處理 decoder 預測時 agent 數不一致：只載入公共部分權重
    decoder_state_dict = torch.load(os.path.join(model_path, "best_decoder.pth"))
    model_state_dict = decoder.state_dict()
    
    # 過濾掉 agent_embed 不匹配的部分
    filtered_state_dict = {k: v for k, v in decoder_state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model_state_dict.update(filtered_state_dict)
    decoder.load_state_dict(model_state_dict)

    encoder.eval()
    decoder.eval()
    
    print("✅ 模型加載完成！（包含 decoder 動態嵌入）")
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

    while len(visited_stations) < len(df_original) -1 :
        # print(f"訪問站點數量: {len(visited_stations)}/{len(df_original)}")
        # 預測配送路徑
        device = next(encoder.parameters()).device
        customer_positions = torch.tensor(df_original.iloc[1:][['X', 'Y']].values, dtype=torch.float32).to(device)
        test_x = test_x.to(device)
        encoded = encoder(test_x)
        agent_inputs = encoded.unsqueeze(1).expand(-1, num_agents, -1)  # (batch_size, num_agents, hidden_size*2)
        current_num_agents = agent_inputs.shape[1]
        predicted_routes = decoder(encoded, agent_inputs, current_num_agents, customer_positions).detach().cpu().numpy()

        mask_idx = [i - 1 for i in visited_stations if i != 0]  # 因為 decoder 對的是 df[1:]
        mask = np.isin(np.arange(predicted_routes.shape[-1]), mask_idx)
        predicted_routes[:, :, mask] = -np.inf

        # 直接選擇機率最高的站點
        predicted_indices = np.argmax(predicted_routes, axis=-1).flatten()
        predicted_indices = np.clip(predicted_indices, 0, len(df_original) - 1)

        any_new = False
        for agent_id, idx in enumerate(predicted_indices):
            idx += 1
            if idx not in visited_stations:
                route_splits[agent_id].append(df_original.iloc[idx])
                visited_stations.add(idx)
                any_new = True

        if not any_new:
            print("⚠️ 無法找到更多可拜訪的站點，提前結束。")
            break

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
