# Mô phỏng Giao thông Multi-Agent System (MAS)

Dự án mô phỏng giao thông đa tác tử (multi-agent): các xe tự học cách di chuyển tối ưu thông qua cơ chế **Thưởng/Phạt**, trong một thành phố có **đèn giao thông** (môi trường).

## Công nghệ

| Thư viện | Vai trò |
| :--- | :--- |
| **Mesa** | Framework đa tác tử – `CarAgent(mesa.Agent)` + `TrafficModel(mesa.Model)` |
| **NetworkX** | Quản lý đồ thị giao thông (nút = ngã tư, cạnh = đường) |
| **Pygame** | Hiển thị trực quan (vẽ bản đồ, xe, đèn, bảng điểm) |

## Cấu trúc thư mục

```
NCKH_test/
├── config.py             # ⚙️ File cấu hình (điểm thưởng/phạt, số xe, kích thước...)
├── run.py                # Điểm khởi đầu – chạy file này để bắt đầu
├── network_manager.py    # Tạo bản đồ (NetworkX) + quản lý đèn giao thông (môi trường)
├── simulation_core.py    # CarAgent (mesa.Agent) + TrafficModel (mesa.Model)
├── visualizer.py         # Hiển thị Pygame (đường, đèn, xe, bảng điểm)
└── README.md             # File này
```

## Cơ chế Thưởng/Phạt

| Hành động | Điểm | Ghi chú |
| :--- | :---: | :--- |
| Đến đích | **+20** | |
| Dừng đúng đèn đỏ | **+2** | |
| Rẽ phải khi đèn đỏ | **0** | Được phép, không phạt |
| Vượt đèn đỏ (thẳng/trái) | **-15** | |
| Va chạm | **-10** | |
| Mỗi bước di chuyển | **-0.1** | |

> Tất cả giá trị trên có thể chỉnh sửa trong `config.py`.

## Cài đặt & Chạy

```bash
pip install mesa networkx pygame-ce
python run.py
```

> **Lưu ý:** Sử dụng `pygame-ce` (community edition) thay vì `pygame` vì Python 3.14 chưa tương thích với `pygame` gốc. API giống hệt nhau.

Nhấn **ESC** hoặc đóng cửa sổ để thoát.
