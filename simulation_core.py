"""
simulation_core.py
-------------------
Logic đa tác tử dựa trên Mesa + Q-Learning.
Xe di chuyển LIÊN TỤC trên cạnh (edge) với tốc độ, không nhảy giữa các nút.

Trạng thái xe:
  - AT_NODE: Đang ở ngã tư → chọn GO / WAIT / REROUTE
  - ON_EDGE: Đang trên đường → chọn ACCEL / MAINTAIN / DECEL / STOP / U_TURN

MÔI TRƯỜNG quyết định: đèn xoay vòng, va chạm, rẽ phải được phép.
AGENT quyết định: tất cả hành động di chuyển (học qua Q-Table).
"""

import random
from collections import defaultdict
import mesa

from config import (
    REWARD_REACH_DEST,
    REWARD_RIGHT_ON_RED,
    PENALTY_RUN_RED,
    PENALTY_COLLISION,
    PENALTY_STEP,
    PENALTY_WAIT,
    PENALTY_UTURN,
    ALLOW_RIGHT_ON_RED,
    NUM_CARS,
    MAX_CARS_PER_NODE,
    MAX_STEPS_PER_EPOCH,
    SPEED_DEFAULT,
    SPEED_MIN,
    SPEED_MAX,
    SPEED_ACCEL,
    SPEED_DECEL,
    SAFE_DISTANCE,
    MAX_CARS_PER_EDGE_DIR,
    LEARNING_RATE,
    DISCOUNT_FACTOR,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY,
)
from network_manager import (
    create_grid_graph,
    update_traffic_lights,
    get_shortest_path,
    is_right_turn,
)


# ===================== ACTIONS =====================
# Khi ở ngã tư (AT_NODE)
A_GO = "go"             # Đi vào cạnh tiếp theo
A_WAIT = "wait"         # Chờ tại ngã tư
A_REROUTE = "reroute"   # Tìm đường khác
NODE_ACTIONS = [A_GO, A_WAIT, A_REROUTE]

# Khi đang trên đường (ON_EDGE)
A_ACCEL = "accel"       # Tăng tốc
A_MAINTAIN = "maintain"  # Giữ tốc độ
A_DECEL = "decel"       # Giảm tốc
A_STOP = "stop"         # Dừng hẳn trên đường
A_UTURN = "uturn"       # Quay đầu
EDGE_ACTIONS = [A_ACCEL, A_MAINTAIN, A_DECEL, A_STOP, A_UTURN]


# ===================== CAR AGENT =====================
class CarAgent(mesa.Agent):
    """
    Tác tử xe – di chuyển liên tục trên cạnh.
    Q-Learning quyết định mọi hành động.
    """

    def __init__(self, model, origin, destination):
        super().__init__(model)
        self.origin = origin
        self.destination = destination

        # --- Vị trí ---
        self.current_node = origin     # Nút hiện tại (nếu AT_NODE)
        self.previous_node = None
        self.edge_from = None          # Nút đầu cạnh đang đi
        self.edge_to = None            # Nút cuối cạnh đang đi
        self.edge_progress = 0.0       # 0.0 (đầu cạnh) → 1.0 (cuối cạnh)
        self.on_edge = False           # True = đang trên đường

        # --- Tốc độ ---
        self.speed = SPEED_DEFAULT

        # --- Trạng thái ---
        self.path = []
        self.accumulated_reward = 0.0
        self.reached = False
        self.wait_count = 0

        # --- Q-Learning ---
        self.q_table = defaultdict(float)
        self.epsilon = EPSILON_START
        self.last_state = None
        self.last_action = None

        self._recalc_path()

    # ==================== VỊ TRÍ PIXEL ====================
    def get_pixel_pos(self):
        """Trả về tọa độ pixel hiện tại (để Pygame vẽ)."""
        G = self.model.G
        if self.on_edge and self.edge_from is not None:
            p1 = G.nodes[self.edge_from]["pos"]
            p2 = G.nodes[self.edge_to]["pos"]
            t = self.edge_progress
            return (
                p1[0] + (p2[0] - p1[0]) * t,
                p1[1] + (p2[1] - p1[1]) * t,
            )
        else:
            return G.nodes[self.current_node]["pos"]

    # ==================== QUAN SÁT ====================
    def _get_state(self):
        """Quan sát môi trường → tuple trạng thái cho Q-Table."""
        G = self.model.G

        if self.on_edge:
            # Đang trên đường
            progress_bucket = int(self.edge_progress * 4)  # 0,1,2,3
            speed_bucket = (0 if self.speed <= 0
                            else 1 if self.speed < SPEED_MAX * 0.6
                            else 2)
            car_ahead = self._car_ahead_distance()
            dist_bucket = (0 if car_ahead < SAFE_DISTANCE
                           else 1 if car_ahead < SAFE_DISTANCE * 3
                           else 2)
            # Đèn cuối đường
            light = G.nodes[self.edge_to]["traffic_light"]
            return ("edge", light, progress_bucket, speed_bucket, dist_bucket)
        else:
            # Đang ở ngã tư
            if not self.path:
                return ("node", "none", 0, 0, False)
            next_node = self.path[0]
            light = G.nodes[next_node]["traffic_light"]
            cars = min(self._count_cars_at(next_node), 2)
            wait_b = 0 if self.wait_count == 0 else (1 if self.wait_count <= 2 else 2)
            can_right = (
                ALLOW_RIGHT_ON_RED
                and self.previous_node is not None
                and is_right_turn(G, self.previous_node,
                                  self.current_node, next_node)
            )
            return ("node", light, cars, wait_b, can_right)

    # ==================== Q-LEARNING ====================
    def _choose_action(self, state):
        """Epsilon-greedy."""
        actions = EDGE_ACTIONS if self.on_edge else NODE_ACTIONS
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_vals = [self.q_table[(state, a)] for a in actions]
        max_q = max(q_vals)
        best = [a for a, q in zip(actions, q_vals) if q == max_q]
        return random.choice(best)

    def _update_q(self, state, action, reward, next_state):
        """Bellman update."""
        old_q = self.q_table[(state, action)]
        next_actions = EDGE_ACTIONS if next_state[0] == "edge" else NODE_ACTIONS
        future_q = max(self.q_table[(next_state, a)] for a in next_actions)
        new_q = old_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * future_q - old_q
        )
        self.q_table[(state, action)] = new_q

    def _decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    # ==================== HELPERS ====================
    def _recalc_path(self, avoid_nodes=None):
        G = self.model.G
        src = self.current_node if not self.on_edge else self.edge_to
        if avoid_nodes:
            H = G.copy()
            for n in avoid_nodes:
                if n != src and n != self.destination:
                    H.remove_node(n)
            full_path = get_shortest_path(H, src, self.destination)
        else:
            full_path = get_shortest_path(G, src, self.destination)
        # Nếu on_edge, path bắt đầu từ edge_to
        if self.on_edge and full_path and full_path[0] == self.edge_to:
            self.path = full_path[1:]
        else:
            self.path = full_path[1:] if len(full_path) > 1 else []

    def _count_cars_at(self, node):
        return sum(1 for a in self.model.agents
                   if a is not self and not a.on_edge and not a.reached and a.current_node == node)

    def _is_node_full(self, node):
        return self._count_cars_at(node) >= MAX_CARS_PER_NODE

    def _car_ahead_distance(self):
        """Khoảng cách đến xe gần nhất phía trước trên cùng cạnh CÙNG CHIỀU."""
        min_dist = 999.0
        for a in self.model.agents:
            if (a is not self and a.on_edge
                    and a.edge_from == self.edge_from
                    and a.edge_to == self.edge_to  # cùng chiều
                    and a.edge_progress > self.edge_progress):
                dist = a.edge_progress - self.edge_progress
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _count_same_dir_on_edge(self):
        """Số xe khác đang đi cùng chiều trên cùng cạnh."""
        return sum(1 for a in self.model.agents
                   if a is not self and a.on_edge and not a.reached
                   and a.edge_from == self.edge_from
                   and a.edge_to == self.edge_to)

    def _is_edge_full(self, from_node, to_node):
        """Kiểm tra cạnh có đầy xe cùng chiều chưa."""
        count = sum(1 for a in self.model.agents
                    if a is not self and a.on_edge and not a.reached
                    and a.edge_from == from_node
                    and a.edge_to == to_node)
        return count >= MAX_CARS_PER_EDGE_DIR

    def _do_reroute(self):
        crowded = set()
        for a in self.model.agents:
            if a is not self and not a.on_edge and not a.reached:
                crowded.add(a.current_node)
        if self.path:
            crowded.add(self.path[0])
        self._recalc_path(avoid_nodes=crowded)

    # ==================== BƯỚC MÔ PHỎNG ====================
    def step(self):
        if self.reached:
            return

        # Đảm bảo có đường
        if not self.on_edge and not self.path:
            self._recalc_path()
            if not self.path:
                return

        # 1) Quan sát
        state = self._get_state()
        # 2) Chọn hành động
        action = self._choose_action(state)
        # 3) Thực thi → nhận reward
        reward = self._execute(action)
        # 4) Quan sát mới
        new_state = self._get_state()
        # 5) Cập nhật Q-Table
        if self.last_state is not None:
            self._update_q(self.last_state, self.last_action, reward, state)
        self.last_state = state
        self.last_action = action
        # 6) Epsilon decay
        self._decay_epsilon()
        # Tích lũy
        self.accumulated_reward += reward

    def _execute(self, action):
        """Thực thi hành động. MÔI TRƯỜNG trả reward."""
        if self.on_edge:
            return self._execute_on_edge(action)
        else:
            return self._execute_at_node(action)

    # ---------- Tại ngã tư ----------
    def _execute_at_node(self, action):
        G = self.model.G

        if action == A_WAIT:
            self.wait_count += 1
            return PENALTY_WAIT

        if action == A_REROUTE:
            self.wait_count = 0
            self._do_reroute()
            return PENALTY_WAIT

        # action == A_GO
        if not self.path:
            self.wait_count += 1
            return PENALTY_WAIT

        next_node = self.path[0]
        light = G.nodes[next_node]["traffic_light"]
        can_right = (
            ALLOW_RIGHT_ON_RED
            and self.previous_node is not None
            and is_right_turn(G, self.previous_node, self.current_node, next_node)
        )

        # Đèn đỏ + không rẽ phải → vi phạm nếu cố đi
        if light == "red" and not can_right:
            self.wait_count += 1
            return PENALTY_RUN_RED

        # Cạnh cùng chiều đã đầy → không vào được
        if self._is_edge_full(self.current_node, next_node):
            self.wait_count += 1
            return PENALTY_COLLISION

        # Đi vào cạnh
        self.on_edge = True
        self.edge_from = self.current_node
        self.edge_to = next_node
        self.edge_progress = 0.0
        self.speed = SPEED_DEFAULT
        self.path.pop(0)
        self.wait_count = 0

        reward = PENALTY_STEP
        if can_right and light == "red":
            reward += REWARD_RIGHT_ON_RED
        return reward

    # ---------- Trên đường ----------
    def _execute_on_edge(self, action):
        G = self.model.G

        # Áp dụng hành động lên tốc độ
        if action == A_ACCEL:
            self.speed = min(SPEED_MAX, self.speed + SPEED_ACCEL)
        elif action == A_DECEL:
            self.speed = max(SPEED_MIN, self.speed - SPEED_DECEL)
        elif action == A_STOP:
            self.speed = 0.0
        elif action == A_UTURN:
            # Quay đầu: đổi chiều trên cạnh
            self.edge_from, self.edge_to = self.edge_to, self.edge_from
            self.edge_progress = 1.0 - self.edge_progress
            self.speed = SPEED_DEFAULT * 0.5  # Quay đầu chậm
            # Tính lại path từ nút mới
            self._recalc_path()
            return PENALTY_UTURN
        # A_MAINTAIN: không thay đổi tốc độ

        # Dừng lại (speed = 0)
        if self.speed <= 0:
            return PENALTY_WAIT

        # Kiểm tra khoảng cách an toàn
        ahead_dist = self._car_ahead_distance()
        if ahead_dist < SAFE_DISTANCE:
            # Gần va chạm → phạt
            self.speed = 0.0
            return PENALTY_COLLISION

        # Di chuyển
        self.edge_progress += self.speed

        # Đã đến cuối cạnh?
        if self.edge_progress >= 1.0:
            return self._arrive_at_node()

        return PENALTY_STEP

    def _arrive_at_node(self):
        """Xe đến cuối cạnh → vào ngã tư."""
        G = self.model.G
        target = self.edge_to

        # Kiểm tra đèn tại ngã tư đích
        light = G.nodes[target]["traffic_light"]
        can_right = False
        if self.edge_from is not None and self.path:
            next_next = self.path[0] if self.path else None
            if next_next is not None:
                can_right = (
                    ALLOW_RIGHT_ON_RED
                    and is_right_turn(G, self.edge_from, target, next_next)
                )

        # Đèn đỏ + không rẽ phải → dừng lại ở cuối cạnh, chờ
        if light == "red" and not can_right:
            self.edge_progress = 0.98  # Dừng sát ngã tư
            self.speed = 0.0
            return PENALTY_STEP  # Không phạt, chỉ chờ

        # Ngã tư đầy → chờ
        if self._is_node_full(target):
            self.edge_progress = 0.98
            self.speed = 0.0
            return PENALTY_COLLISION

        # Vào ngã tư thành công
        self.previous_node = self.edge_from
        self.current_node = target
        self.on_edge = False
        self.edge_from = None
        self.edge_to = None
        self.edge_progress = 0.0

        # Đến đích?
        if self.current_node == self.destination:
            self.reached = True
            return REWARD_REACH_DEST

        # Tính lại đường nếu cần
        if not self.path:
            self._recalc_path()

        return PENALTY_STEP


# ===================== TRAFFIC MODEL =====================
class TrafficModel(mesa.Model):
    """
    Mô hình MÔI TRƯỜNG – chỉ quản lý đèn và đồ thị.
    Không quyết định thay agent.
    """

    def __init__(self, num_cars=NUM_CARS):
        super().__init__()
        self.G = create_grid_graph()
        self.num_cars = num_cars
        self.step_count = 0
        self.epoch = 1
        self.epoch_rewards = []  # Lưu avg reward mỗi epoch

        all_nodes = list(self.G.nodes)
        for _ in range(num_cars):
            origin = random.choice(all_nodes)
            destination = random.choice(all_nodes)
            while destination == origin:
                destination = random.choice(all_nodes)
            CarAgent(self, origin, destination)

    def step(self):
        update_traffic_lights(self.G)
        self.step_count += 1
        self.agents.shuffle_do("step")
        # Tự chuyển epoch nếu tất cả đến đích hoặc quá nhiều bước
        if self.all_reached() or self.step_count >= MAX_STEPS_PER_EPOCH:
            self.reset_epoch()

    def reset_epoch(self):
        """Reset vị trí xe, GIỮ NGUYÊN Q-Table (bộ não đã học)."""
        self.epoch_rewards.append(self.get_avg_reward())
        self.epoch += 1
        self.step_count = 0

        all_nodes = list(self.G.nodes)
        for agent in self.agents:
            # Vị trí mới ngẫu nhiên
            origin = random.choice(all_nodes)
            destination = random.choice(all_nodes)
            while destination == origin:
                destination = random.choice(all_nodes)
            # Reset trạng thái, GIỮ q_table + epsilon
            agent.origin = origin
            agent.destination = destination
            agent.current_node = origin
            agent.previous_node = None
            agent.edge_from = None
            agent.edge_to = None
            agent.edge_progress = 0.0
            agent.on_edge = False
            agent.speed = SPEED_DEFAULT
            agent.path = []
            agent.accumulated_reward = 0.0
            agent.reached = False
            agent.wait_count = 0
            agent.last_state = None
            agent.last_action = None
            agent._recalc_path()

    def skip_to_epoch(self, target_epoch):
        """Chạy nhanh (không render) đến epoch mục tiêu."""
        while self.epoch < target_epoch:
            for _ in range(MAX_STEPS_PER_EPOCH):
                update_traffic_lights(self.G)
                self.step_count += 1
                self.agents.shuffle_do("step")
                if self.all_reached() or self.step_count >= MAX_STEPS_PER_EPOCH:
                    self.reset_epoch()
                    break

    def get_avg_reward(self):
        rewards = [a.accumulated_reward for a in self.agents]
        return sum(rewards) / len(rewards) if rewards else 0

    def get_reached_count(self):
        return sum(1 for a in self.agents if a.reached)

    def all_reached(self):
        return all(a.reached for a in self.agents)
