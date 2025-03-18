#utils.py
import matplotlib.pyplot as plt
import pandas as pd
import os
from torchsummary import summary


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

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 預設顏色列表
    num_colors = len(colors)

    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)

        # 檢查是否包含 `X`, `Y`, `Distance` 欄位
        if 'X' not in df.columns or 'Y' not in df.columns:
            print(f"⚠️ [警告] CSV 檔案 {csv_path} 缺少 'X' 或 'Y' 欄位，跳過。")
            continue

        # 計算總距離
        if 'Distance' in df.columns:
            total_distance += df['Distance'].sum()

        # 繪製該車輛的路徑
        plt.plot(df['X'], df['Y'], marker='o', linestyle='-', 
                 color=colors[i % num_colors], label=f'Vehicle {i+1} ({csv_path})')

    # 設定標題與標籤
    plt.title(f"Vehicle Routes Visualization (Total Distance: {total_distance:.2f})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.show()

    # 儲存圖片
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"routes_{date_str}.png")
    plt.savefig(image_path)
    plt.close()
    print(f"✅ 配送路徑圖已儲存至 {image_path}")


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