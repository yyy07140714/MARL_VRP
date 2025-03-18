# environment.py
import numpy as np
import torch

class Environment:
    def __init__(self, df, num_vehicles):
        self.df = df.copy()
        self.start_x = df.iloc[0]["X"]
        self.start_y = df.iloc[0]["Y"]
        self.current_time = 0
        self.vehicle_positions = [[self.start_x, self.start_y] for _ in range(num_vehicles)]
        self.visited_customers = set()
        self.vehicles_completed = set()
        self.num_vehicles = num_vehicles
        self.reset()

    def reset(self):
        self.visited_customers = set()  #清空已訪問店家
        self.vehicle_positions = [[self.start_x, self.start_y] for _ in range(self.num_vehicles)]
        print(f"[DEBUG] 重置車輛位置: {self.vehicle_positions}")
    

    def calculate_reward(self, previous_position, new_position):
        """
        計算獎勵，根據移動距離給予懲罰或獎勵
        """
        distance = np.sqrt((previous_position[0] - new_position[0])**2 +
                           (previous_position[1] - new_position[1])**2)
        print(f"[DEBUG] 移動距離: {distance}")
        return 10-distance/1000  # 移動距離越短，獎勵越高（懲罰長距離移動）
    

    def update_state(self, state, actions):
        """ 更新車輛狀態，並記錄訪問的客戶，計算獎勵 """
        total_reward = 0
        for i, action in enumerate(actions):
            action_x, action_y = action[0], action[1]
            prev_x, prev_y = self.vehicle_positions[i]

            # 計算移動距離獎勵
            reward = self.calculate_reward((prev_x, prev_y), (action_x, action_y))
            total_reward += reward

            # 更新車輛位置
            self.vehicle_positions[i] = [action_x, action_y]

            for j, row in self.df.iterrows():
                if np.isclose(row["X"], action_x) and np.isclose(row["Y"], action_y):
                    if j not in self.visited_customers:
                        self.visited_customers.add(j)  # 標記為已拜訪
                        state[0, i, 2] = row["W_S"]   # 更新 W_S
                        state[0, i, 3] = row["W_F"]   # 更新 W_F
                        state[0, i, 4] = row["Demand"]  # 更新 Demand
                        state[0, i, 5] = self.current_time  # 更新當前時間
                    break

        print(f"[DEBUG] 已訪問站點數: {len(self.visited_customers)}/{len(self.df)-1}, 總獎勵: {total_reward:.2f}")
        return state, total_reward



    def get_arrival_time(self, action, vehicle_id):
        """
        根據單輛車的動作計算到達時間，並檢查是否回到起點
        """
        x, y = action
        prev_x, prev_y = self.vehicle_positions[vehicle_id]

        distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        travel_time = distance / 10  # **假設速度為 10 單位/時間**
        self.current_time += travel_time

        # **更新車輛位置**
        self.vehicle_positions[vehicle_id] = (x, y)

        # **檢查是否回到起點**
        if (x, y) == (self.start_x, self.start_y):
            self.vehicles_completed.add(vehicle_id)  # 標記該車輛已完成配送
            del self.vehicle_positions[vehicle_id]  # 從運行中的車輛中移除

        return self.current_time


    def get_time_window(self, action):
        """
        取得單輛車的目標客戶時間窗 (e_j, l_j)
        action: (X, Y) 預測的座標
        """
        x, y = action
        customer_row = self.df[(self.df["X"] == x) & (self.df["Y"] == y)]
        
        if not customer_row.empty:
            e_j, l_j = customer_row.iloc[0]["W_S"], customer_row.iloc[0]["W_F"]
        else:
            e_j, l_j = 0, 86400  # 預設為一整天內可到達

        return [(e_j, l_j)]


    def is_done(self, state):
        """
        只要所有站點都訪問完就停止
        """
        total_stations = len(self.df) - 1  #減去起始站
        all_visited = len(self.visited_customers) >= total_stations  
        print(f"[DEBUG] 已拜訪客戶數: {len(self.visited_customers)}/{total_stations}")

        return all_visited
