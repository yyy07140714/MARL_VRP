import torch
import torch.optim as optim
import torch.nn as nn
import os
from model import GRUEncoder, StrategyModule

def train_model(train_x, train_y, input_size, hidden_size, output_size, epochs=10, lr=0.001, save_path="save_models/"):
    # 初始化模型
    encoder = GRUEncoder(input_size, hidden_size)
    strategy_module = StrategyModule(hidden_size, output_size)

    # 優化器 & 損失函數
    optimizer = optim.Adam(list(encoder.parameters()) + list(strategy_module.parameters()), lr=lr)
    criterion = nn.MSELoss()

    # 訓練過程
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        encoded = encoder(train_x)  # 經過 GRU 編碼
        predicted_y = strategy_module(encoded)  # MARL 決策下一步
        
        loss = criterion(predicted_y, train_y)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("訓練完成！")
    
    # 儲存模型
    torch.save(encoder.state_dict(), os.path.join(save_path, "encoder.pth"))
    torch.save(strategy_module.state_dict(), os.path.join(save_path, "strategy_module.pth"))

    print(f"模型已儲存至 {save_path}")
    return encoder, strategy_module