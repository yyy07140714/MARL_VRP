import torch
import pandas as pd
import os
from model import GRUEncoder, StrategyModule

def load_models(input_size, hidden_size, output_size, model_path="save_models/"):
    encoder = GRUEncoder(input_size, hidden_size)
    strategy_module = StrategyModule(hidden_size, output_size)

    encoder.load_state_dict(torch.load(os.path.join(model_path, "encoder.pth")))
    strategy_module.load_state_dict(torch.load(os.path.join(model_path, "strategy_module.pth")))

    encoder.eval()
    strategy_module.eval()
    
    print("模型加載完成！")
    return encoder, strategy_module


def generate_routes(encoder, strategy_module, test_x, csv_path, output_dir="Output/"):
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

    # 預測配送路徑
    encoded = encoder(test_x)
    predicted_routes = strategy_module(encoded).detach().numpy()  # 預測 (X, Y)

    # 確保有足夠的資料對應
    num_predictions = predicted_routes.shape[0]
    df_original = df_original.iloc[:num_predictions]  # 確保資料數量對齊

    # 反標準化 (如果有標準化過)
    X_min, X_max = df_original["X"].min(), df_original["X"].max()
    Y_min, Y_max = df_original["Y"].min(), df_original["Y"].max()
    predicted_routes[:, 0] = predicted_routes[:, 0] * (X_max - X_min) + X_min
    predicted_routes[:, 1] = predicted_routes[:, 1] * (Y_max - Y_min) + Y_min

    # 分配配送點給不同車輛
    df_original["Predicted_X"] = predicted_routes[:, 0]
    df_original["Predicted_Y"] = predicted_routes[:, 1]
    
    # 根據 `num_agents` 切分數據
    route_splits = [[] for _ in range(num_agents)]
    for i, row in df_original.iterrows():
        route_splits[i % num_agents].append(row)
    
    csv_paths = []
    # 儲存不同車輛的路徑
    for agent_id, route_data in enumerate(route_splits):
        df_route = pd.DataFrame(route_data)  # 轉換為 DataFrame
        output_path = os.path.join(output_dir, f"route_{date_str}_{agent_id+1}.csv")
        df_route.to_csv(output_path, index=False)
        csv_paths.append(output_path)
        print(f"車輛 {agent_id+1} 的配送路徑已儲存至 {output_path}")

    return csv_paths, date_folder


def run_prediction(csv_path):
    input_size = 3
    hidden_size = 50
    output_size = 2
    test_x = torch.randn(16, 20, input_size)  # 模擬輸入數據

    encoder, strategy_module = load_models(input_size, hidden_size, output_size)
    generate_routes(encoder, strategy_module, test_x, csv_path)


def inverse_transform(predicted_x, predicted_y, X_min, X_max, Y_min, Y_max):
    original_x = predicted_x * (X_max - X_min) + X_min
    original_y = predicted_y * (Y_max - Y_min) + Y_min
    return original_x, original_y