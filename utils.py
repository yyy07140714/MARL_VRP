#utils.py
import os
import math
import pandas as pd
import matplotlib.pyplot as plt


# 視覺化站點及座標
def plot_vrp(df):
    plt.figure(figsize=(8, 6))

    plt.scatter(df["X"], df["Y"], c="blue", marker="o", label="Customers")

    # for i, row in df.iterrows():
    #     plt.text(row["X"], row["Y"], str(row["Number"]), fontsize=9, ha='right', va='bottom')

    plt.xlim(df["X"].min() - 500, df["X"].max() + 500)
    plt.ylim(df["Y"].min() - 500, df["Y"].max() + 500)

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Vehicle Routing Problem (VRP) - Customer Locations")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()

def visualize_routes(csv_paths, save_dir, date_str):
    plt.figure(figsize=(12, 8))
    total_distance = 0  # 初始化總距離

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    num_colors = len(colors)

    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)

        # 檢查是否包含 `X`, `Y` 欄位
        if 'X' not in df.columns or 'Y' not in df.columns:
            print(f"⚠️ [警告] CSV 檔案 {csv_path} 缺少 'X' 或 'Y' 欄位，跳過。")
            continue

        # 扣掉回原點的點（最後一筆）
        df_trimmed = df.iloc[:-1] if len(df) > 1 else df
        x = df_trimmed['X'].values
        y = df_trimmed['Y'].values

        # 計算單一路徑距離
        route_distance = 0.0
        for j in range(1, len(x)):
            dx = x[j] - x[j - 1]
            dy = y[j] - y[j - 1]
            dist = math.hypot(dx, dy)
            route_distance += dist

        total_distance += route_distance

        # 繪製路徑
        plt.plot(x, y, marker='o', linestyle='-',
                 color=colors[i % num_colors], label=f'Vehicle {i+1}')

        # 在中點附近標出該路線距離
        if len(x) >= 2:
            mid_idx = len(x) // 2
            # plt.text(x[mid_idx], y[mid_idx], f'{route_distance:.1f}m',
            #          fontsize=9, color=colors[i % num_colors])

    # 圖表設定與儲存
    plt.title(f"Vehicle Routes Visualization (Total Distance: {total_distance:.1f}m)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.show()

    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"routes_{date_str}.png")
    plt.savefig(image_path)
    plt.close()
    print(f"✅ 配送路徑圖已儲存至 {image_path}")


# def visualize_routes(csv_paths, save_dir, date_str):

#     plt.figure(figsize=(12, 8))
#     total_distance = 0  # 初始化總距離

#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 預設顏色列表
#     num_colors = len(colors)

#     for i, csv_path in enumerate(csv_paths):
#         df = pd.read_csv(csv_path)

#         # 檢查是否包含 `X`, `Y`, `Distance` 欄位
#         if 'X' not in df.columns or 'Y' not in df.columns:
#             print(f"⚠️ [警告] CSV 檔案 {csv_path} 缺少 'X' 或 'Y' 欄位，跳過。")
#             continue

#         # 計算總距離
#         if 'Distance' in df.columns:
#             total_distance += df['Distance'].sum()

#         # 繪製該車輛的路徑
#         plt.plot(df['X'], df['Y'], marker='o', linestyle='-', 
#                  color=colors[i % num_colors], label=f'Vehicle {i+1} ({csv_path})')

#     # 設定標題與標籤
#     plt.title(f"Vehicle Routes Visualization (Total Distance: {total_distance:.2f})")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.legend(loc='best', fontsize='small')
#     plt.grid(True)
#     plt.show()

#     # 儲存圖片
#     os.makedirs(save_dir, exist_ok=True)
#     image_path = os.path.join(save_dir, f"routes_{date_str}.png")
#     plt.savefig(image_path)
#     plt.close()
#     print(f"✅ 配送路徑圖已儲存至 {image_path}")


def print_model_summary(encoder, strategy_module, decoder, input_size):
    print("Encoder Summary:")
    summary(encoder, input_size=input_size)

    print("\nStrategy Module Summary:")
    summary(strategy_module, input_size=(encoder.output_dim,))

    print("\nDecoder Summary:")
    # Use decoder's hidden_size or output_size if available, otherwise use input_size
    decoder_input_size = getattr(decoder, 'hidden_size', 
                               getattr(decoder, 'output_size', 
                                     input_size[-1]))
    summary(decoder, input_size=(decoder_input_size * 2,))


def plot_training_curves(losses, rewards, save_path=None):
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss per Epoch', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)

    # Reward curve
    plt.subplot(1, 2, 2)
    plt.plot(rewards, label='Reward per Epoch', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Training Reward Curve')
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ 圖片已儲存至 {save_path}")
    else:
        plt.show()