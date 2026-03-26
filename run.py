"""
run.py
-------
Điểm khởi đầu – Khởi tạo model Mesa rồi chạy visualizer Pygame.
"""

from config import NUM_CARS
from simulation_core import TrafficModel
from visualizer import run_visualization


def main():
    # Số xe lấy từ config.py (bạn có thể chỉnh sửa ở đó)

    print("=" * 50)
    print("  MAS Traffic Simulation")
    print("  Mesa  |  NetworkX  |  Pygame")
    print("=" * 50)
    print(f"  Số xe  : {NUM_CARS}")
    print("  Nhấn ESC hoặc đóng cửa sổ để thoát.")
    print("=" * 50)

    model = TrafficModel(num_cars=NUM_CARS)
    run_visualization(model)


if __name__ == "__main__":
    main()
