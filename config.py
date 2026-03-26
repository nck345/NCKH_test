"""
config.py
----------
File cấu hình tập trung cho toàn bộ mô phỏng.
"""

# =============================================================
#                    BẢNG ĐIỂM THƯỞNG / PHẠT
# =============================================================

REWARD_REACH_DEST = 20       # Thưởng khi đến đích

PENALTY_RUN_RED = -50        # Phạt khi vượt đèn đỏ
PENALTY_COLLISION = -30      # Phạt khi va chạm / gần va chạm
PENALTY_STEP = -0.1          # Phạt nhẹ mỗi bước (khuyến khích nhanh)
PENALTY_WAIT = -1.0          # Phạt khi chờ vô ích (đèn xanh mà vẫn đứng)
PENALTY_UTURN = -2.0         # Phạt khi quay đầu (tốn thời gian)

# Đèn đỏ rẽ phải
ALLOW_RIGHT_ON_RED = True
REWARD_RIGHT_ON_RED = 0


# =============================================================
#                    CẤU HÌNH TỐC ĐỘ XE
# =============================================================

SPEED_DEFAULT = 0.15         # Tốc độ ban đầu (% cạnh / bước)
SPEED_MIN = 0.0              # Tốc độ tối thiểu (dừng hẳn)
SPEED_MAX = 0.35             # Tốc độ tối đa
SPEED_ACCEL = 0.05           # Mức tăng khi tăng tốc
SPEED_DECEL = 0.05           # Mức giảm khi giảm tốc

SAFE_DISTANCE = 0.2          # Khoảng cách an toàn giữa 2 xe trên cùng cạnh


# =============================================================
#                    CẤU HÌNH MẠNG LƯỚI (NETWORK)
# =============================================================

GRID_ROWS = 4
GRID_COLS = 5
CELL_SIZE = 140
MARGIN = 80

LIGHT_CYCLE_MIN = 5         # Chu kỳ đèn ngắn nhất (s)
LIGHT_CYCLE_MAX = 12        # Chu kỳ đèn dài nhất (s)


# =============================================================
#                    CẤU HÌNH MÔ PHỎNG (SIMULATION)
# =============================================================

NUM_CARS = 5
MAX_CARS_PER_NODE = 3        # Số xe tối đa tại 1 ngã tư
MAX_STEPS_PER_EPOCH = 500    # Tự chuyển epoch khi quá nhiều bước


# =============================================================
#                    Q-LEARNING (Agent tự học)
# =============================================================

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995


# =============================================================
#                    CẤU HÌNH HIỂN THỊ (PYGAME)
# =============================================================

WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 650
FPS = 10                     # Tăng FPS vì xe di chuyển liên tục
