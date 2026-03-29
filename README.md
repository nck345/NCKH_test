# Traffic Simulation (Mesa + tkinter, turn-based)

Du an nay mo phong giao thong da tac tu theo kieu **turn-based** tren luoi 7x7.

## Kien truc hien tai

- Engine: `Mesa` (`CarAgent`, `TrafficModel`)
- GUI: `tkinter` (ve den giao thong + xe theo style grid)
- Cau hinh runtime: `config.json` (duoc validate boi `config_loader.py`)
- Co che epoch:
  - ket thuc epoch khi dat dieu kien (`all reached` hoac `max_turns_per_epoch`)
  - chuyen epoch thu cong
  - co the skip epoch bang phim tat

## Dieu khien

- `Space`: Pause/Resume
- `N`: Bo qua epoch hien tai va sang epoch tiep theo ngay
- `S`: Nhap epoch dich de skip nhanh toi epoch do
- `Esc`: Thoat

## Cau truc thu muc

```text
NCKH_test/
├── config.json         # Cau hinh runtime chinh
├── config_loader.py    # Parse + validate config.json
├── run.py              # Entry point
├── simulation_core.py  # Model/Agent turn-based + Q-learning + epoch
├── visualizer.py       # GUI tkinter (grid, den, xe, overlay epoch)
├── .gitignore
└── README.md
```

## Cai dat

```bash
pip install mesa
```

`tkinter` la thu vien built-in cua Python tren phan lon ban phan phoi desktop.

## Chay chuong trinh

```bash
python run.py
```

## Ghi chu cau hinh

File `config.json` gom cac nhom chinh:

- `simulation`: `max_turns_per_epoch`, `ms_per_turn`, `manual_epoch_advance`
- `grid`: `size`, `no_cars_per_cell`, `max_cars_per_cell`
- `cars`: `total_cars`, `spawn_rates`, `multiple_cars_per_turn`
- `traffic_lights`: `switch_interval`, `initial_state`, `intelligence`
- `policy`: `priority`
- `q_learning`: cac he so hoc
- `rewards`: diem thuong/phat
- `ui`: kich thuoc cua so + tieu de
