"""
network_manager.py
-------------------
Quản lý mạng lưới giao thông bằng NetworkX.
Đèn giao thông là thuộc tính MÔI TRƯỜNG, không phải Agent.
"""

import math
import networkx as nx
import random

from config import (
    GRID_ROWS, GRID_COLS, CELL_SIZE, MARGIN,
    RING_NODES, RING_RADIUS,
    LIGHT_CYCLE_MIN, LIGHT_CYCLE_MAX,
    WINDOW_WIDTH, WINDOW_HEIGHT,
)


def create_grid_graph(rows=GRID_ROWS, cols=GRID_COLS):
    """
    Tạo đồ thị dạng lưới (grid) đại diện cho bản đồ thành phố.

    Returns:
        G (nx.DiGraph): Đồ thị có hướng.
            - Mỗi node có: pos (x, y), traffic_light ("green"/"red"), light_timer, col
            - Mỗi edge có: weight, direction ("right"/"left"/"down"/"up")
    """
    G = nx.DiGraph()

    # ---------- Tạo nút ----------
    for r in range(rows):
        for c in range(cols):
            node_id = r * cols + c
            x = MARGIN + c * CELL_SIZE
            y = MARGIN + r * CELL_SIZE

            initial_light = random.choice(["green", "red"])
            light_cycle = random.randint(LIGHT_CYCLE_MIN, LIGHT_CYCLE_MAX)
            G.add_node(
                node_id,
                pos=(x, y),
                row=r,
                col=c,
                traffic_light=initial_light,
                light_timer=random.randint(0, light_cycle - 1),
                light_cycle=light_cycle,
            )

    # ---------- Tạo cạnh (đường 2 chiều) với HƯỚNG ----------
    for r in range(rows):
        for c in range(cols):
            node_id = r * cols + c
            # Nối sang phải
            if c + 1 < cols:
                neighbor = r * cols + (c + 1)
                weight = random.randint(1, 5)
                G.add_edge(node_id, neighbor, weight=weight, direction="right")
                G.add_edge(neighbor, node_id, weight=weight, direction="left")
            # Nối xuống dưới
            if r + 1 < rows:
                neighbor = (r + 1) * cols + c
                weight = random.randint(1, 5)
                G.add_edge(node_id, neighbor, weight=weight, direction="down")
                G.add_edge(neighbor, node_id, weight=weight, direction="up")

    return G


def create_ring_graph(num_nodes=RING_NODES, radius=RING_RADIUS):
    """
    Tạo đồ thị dạng vòng tròn khép kín (ring).

    Mỗi nút nối 2 chiều với 2 nút kế bên.
      - Chiều xuôi (clockwise):        direction = "cw"
      - Chiều ngược (counter-clockwise): direction = "ccw"

    Returns:
        G (nx.DiGraph): Đồ thị có hướng dạng vòng.
    """
    G = nx.DiGraph()

    cx = WINDOW_WIDTH // 2      # Tâm vòng tròn
    cy = WINDOW_HEIGHT // 2

    # ---------- Tạo nút ----------
    for i in range(num_nodes):
        angle = 2 * math.pi * i / num_nodes - math.pi / 2   # Bắt đầu từ trên cùng
        x = int(cx + radius * math.cos(angle))
        y = int(cy + radius * math.sin(angle))

        initial_light = random.choice(["green", "red"])
        light_cycle = random.randint(LIGHT_CYCLE_MIN, LIGHT_CYCLE_MAX)
        G.add_node(
            i,
            pos=(x, y),
            traffic_light=initial_light,
            light_timer=random.randint(0, light_cycle - 1),
            light_cycle=light_cycle,
        )

    # ---------- Tạo cạnh (vòng khép kín, 2 chiều) ----------
    for i in range(num_nodes):
        next_node = (i + 1) % num_nodes
        weight = random.randint(1, 5)
        G.add_edge(i, next_node, weight=weight, direction="cw")
        G.add_edge(next_node, i, weight=weight, direction="ccw")

    return G


def update_traffic_lights(G):
    """
    Cập nhật trạng thái đèn giao thông cho TẤT CẢ các nút.
    Hàm này được gọi bởi TrafficModel mỗi bước simulation.
    """
    for node_id in G.nodes:
        data = G.nodes[node_id]
        data["light_timer"] += 1
        if data["light_timer"] >= data["light_cycle"]:
            data["light_timer"] = 0
            if data["traffic_light"] == "green":
                data["traffic_light"] = "red"
            else:
                data["traffic_light"] = "green"


def get_shortest_path(G, source, target):
    """Trả về danh sách nút trên đường ngắn nhất (Dijkstra)."""
    try:
        return nx.shortest_path(G, source, target, weight="weight")
    except nx.NetworkXNoPath:
        return []


def get_node_pos(G, node_id):
    """Trả về tọa độ (x, y) của một nút."""
    return G.nodes[node_id]["pos"]


def is_right_turn(G, from_node, via_node, to_node):
    """
    Kiểm tra xem hướng di chuyển from→via→to có phải là RẼ PHẢI không.

    Quy tắc (nhìn từ trên xuống, trục y hướng xuống):
      Đang đi phải  → rẽ phải = xuống
      Đang đi xuống → rẽ phải = trái
      Đang đi trái  → rẽ phải = lên
      Đang đi lên   → rẽ phải = phải
    """
    if not G.has_edge(from_node, via_node) or not G.has_edge(via_node, to_node):
        return False

    incoming_dir = G.edges[from_node, via_node].get("direction", "")
    outgoing_dir = G.edges[via_node, to_node].get("direction", "")

    right_turn_map = {
        "right": "down",
        "down": "left",
        "left": "up",
        "up": "right",
    }

    return right_turn_map.get(incoming_dir) == outgoing_dir
