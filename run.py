"""
run.py
-------
Điểm khởi đầu – Khởi tạo model Mesa rồi chạy visualizer Pygame.
"""

from config import NUM_CARS
from simulation_core import TrafficModel
from visualizer import run_visualization


def main():
    # Số xe lấy từ config.py

    print("=" * 50)
    print("  MAS Traffic Simulation")
    print("  Mesa  |  NetworkX  |  Pygame")
    print("=" * 50)
    print(f"  Num Cars: {NUM_CARS}")
    print("  Press ESC or close window to exit.")
    print("=" * 50)

    model = TrafficModel(num_cars=NUM_CARS)
    run_visualization(model)


if __name__ == "__main__":
    main()
