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
        self.visited_customers = set()  # 清空已訪問
        self.vehicle_positions = [[self.start_x, self.start_y] for _ in range(self.num_vehicles)]
        self.current_time = 0
        # print(f"[DEBUG] 重置車輛位置: {self.vehicle_positions}")

    def calculate_route_length(self, route):
        """
        計算單輛車的總行駛距離
        """
        total_distance = 0
        for i in range(len(route) - 1):
            x1, y1 = route[i]
            x2, y2 = route[i + 1]
            total_distance += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return total_distance

    def calculate_time_window_penalty(self, arrival_times, customers, alpha=0.5, beta=2):
        """
        計算時間窗違反懲罰
        """
        total_penalty = 0
        for i, customer in enumerate(customers):
            # print(f"[DEBUG] customer: {i}")
            # print(f"[DEBUG] arrival_times: {arrival_times}")
            e_j, l_j = customer["W_S"], customer["W_F"]
            t_ij = arrival_times[i]

            early_penalty = max(0, e_j - t_ij) * alpha
            late_penalty = max(0, t_ij - l_j) * beta
            total_penalty += early_penalty + late_penalty
        return total_penalty

    def calculate_total_cost(self, routes, arrival_times, visited_indices):
        """
        計算總成本 (路徑長度 + 時間窗違反懲罰)
        """
        total_cost = 0
        for vehicle_id, route in enumerate(routes):
            if len(route) > 1:  # 確保有訪問客戶
                total_cost += self.calculate_route_length(route)

                # **只取該車輛實際訪問的站點**
                visited_customers = [self.df.iloc[j] for j in visited_indices[vehicle_id]]
                visited_customers_dict = [cust.to_dict() for cust in visited_customers]

                print(f"[DEBUG] 車輛 {vehicle_id} 訪問的站點數: {len(visited_customers_dict)}, arrival_times 長度: {len(arrival_times[vehicle_id])}")

                total_cost += self.calculate_time_window_penalty(arrival_times[vehicle_id], visited_customers_dict)

        return total_cost


    def calculate_reward(self, previous_position, new_position):
        """
        計算獎勵，根據移動距離給予懲罰或獎勵
        - 距離短：獎勵較高
        - 距離長：懲罰較高
        """
        distance = np.sqrt((previous_position[0] - new_position[0])**2 +
                        (previous_position[1] - new_position[1])**2)

        # 設定獎勵機制 (距離越短，獎勵越高)
        reward = 10 * np.exp(-distance / 10000)  # 控制幅度
        print(f"[DEBUG] 移動距離: {distance:.2f}, 獎勵: {reward:.2f}")
        
        return reward

    def update_state(self, state, actions):
        new_state = state.clone()
        total_reward = 0
        routes = [[] for _ in range(self.num_vehicles)]
        arrival_times = [[] for _ in range(self.num_vehicles)]
        visited_indices = [[] for _ in range(self.num_vehicles)]


        for i, action in enumerate(actions):
            action_x, action_y = action[0], action[1]

            matched_row = self.df[(np.isclose(self.df["X"], action_x)) & (np.isclose(self.df["Y"], action_y))]

            # ✅ 如果有匹配 row（確保非空）
            if not matched_row.empty:
                name = matched_row.iloc[0]["Name"]

                # ✅ 移動這段邏輯進來
                if name.endswith("ＤＣ"):
                    if len(self.visited_customers) < len(self.df) - 1:
                        print(f"[INFO] 🚛 車輛 {i} 想回倉庫，但仍有未配送的站點，繼續行駛")
                        continue
                    else:
                        print(f"[INFO] 🚛 車輛 {i} 完成配送，正式回倉庫")
                        self.vehicles_completed.add(i)
                        total_reward -= 50  # ❗ 太早回倉庫的懲罰

            # 接下來原本 iterrows 那段就保留處理 state 更新...



            for j, row in self.df.iterrows():
                if np.isclose(row["X"], action_x) and np.isclose(row["Y"], action_y):
                    if j not in self.visited_customers:
                        visited_indices[i].append(j)
                        self.visited_customers.add(j)
                        previous_position = state[0, i, :2].cpu().numpy()
                        new_state[0, i, 2] = row["W_S"]
                        new_state[0, i, 3] = row["W_F"]
                        new_state[0, i, 4] = row["Demand"]
                        new_state[0, i, 5] = self.current_time

                        new_position = np.array([action_x, action_y])
                        reward = self.calculate_reward(previous_position, new_position)
                        total_reward += reward

                        # 記錄路徑
                        routes[i].append((action_x, action_y))
                        arrival_times[i].append(self.get_arrival_time(action, i))
                    break

        total_cost = self.calculate_total_cost(routes, arrival_times, visited_indices)
        # print(f"[DEBUG] 已訪問站點數: {len(self.visited_customers)}/{len(self.df)-1}, 總獎勵: {total_reward:.2f}, 總成本: {total_cost:.2f}")
        return new_state, total_reward - total_cost, visited_indices


    def get_arrival_time(self, action, vehicle_id):
        """
        根據單輛車的動作計算到達時間，並檢查是否回到起點
        """
        x, y = action
        # print(f"[DEBUG] 車輛 {vehicle_id} 的動作: {action}")
        prev_x, prev_y = self.vehicle_positions[vehicle_id]

        distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        travel_time = distance / 10  # **假設速度為 10 單位/時間**
        self.current_time += travel_time

        # **更新車輛位置**
        self.vehicle_positions[vehicle_id] = (x, y)

        # **檢查是否回到起點**
        if np.isclose(x, self.start_x) and np.isclose(y, self.start_y):
            print(f"[INFO] 🚛 車輛 {vehicle_id} 回到倉庫")
            self.vehicles_completed.add(vehicle_id)

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
        # print(f"[DEBUG] 已拜訪客戶數: {len(self.visited_customers)}/{total_stations}")

        return all_visited
