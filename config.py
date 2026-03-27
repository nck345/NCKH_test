"""
config.py
----------
File cấu hình tập trung cho toàn bộ mô phỏng.
"""

# =============================================================
#                    BẢNG ĐIỂM THƯỞNG / PHẠT
# =============================================================

REWARD_REACH_DEST = 100      # Thưởng khi đến đích (Tăng từ 20)

PENALTY_RUN_RED = -10        # Phạt khi vượt đèn đỏ (Giảm từ -50)
PENALTY_COLLISION = -10      # Phạt khi va chạm / gần va chạm (Giảm từ -30)
PENALTY_TIME = -0.5          # Chi phí thời gian mỗi bước (Gộp từ STEP và WAIT)
PENALTY_UTURN = -1.0         # Phạt khi quay đầu (tốn thời gian)

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

SAFE_DISTANCE = 0.2          # Khoảng cách an toàn giữa 2 xe cùng chiều trên cạnh



# =============================================================
#                    CẤU HÌNH MẠNG LƯỚI (NETWORK)
# =============================================================

# Loại môi trường: "grid" hoặc "ring"
ENV_TYPE = "ring"

# --- Grid ---
GRID_ROWS = 4
GRID_COLS = 5
CELL_SIZE = 140
MARGIN = 80

# --- Ring (vòng tròn khép kín) ---
RING_NODES = 10              # Số nút trên vòng tròn
RING_RADIUS = 250            # Bán kính vòng tròn (pixel)

LIGHT_CYCLE_MIN = 10         # Chu kỳ đèn ngắn nhất (s)
LIGHT_CYCLE_MAX = 20        # Chu kỳ đèn dài nhất (s)


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
