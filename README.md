# Traffic Simulation (Mesa + tkinter, turn-based)

Du an nay mo phong giao thong da tac tu theo kieu **turn-based** tren luoi 7x7.

## Kien truc hien tai

- Engine: `Mesa` (`CarAgent`, `TrafficModel`)
- GUI: `tkinter` (ve den giao thong + xe theo style grid)
- Cau hinh runtime: `config.json` (duoc validate boi `config_loader.py`)
- Duong 2 chieu dang mang nhieu nut giao: moi truc duong co 2 dai o (moi dai la 1 chieu)
- Moi giao diem ngang-doc tao thanh mot nut den giao thong 2x2
- Den giao thong duoc hien thi bang 4 duong vien quanh nut (2 ngang, 2 doc) va doi mau theo chu ky
- Moi o chi cho phep toi da 1 xe
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
- `grid`: `size`, `no_cars_per_cell`, `max_cars_per_cell`, `intersection_spacing_cells`
- `cars`: `total_cars`, `spawn_rates`, `multiple_cars_per_turn`
- `traffic_lights`: `switch_interval`, `initial_state`, `intelligence`
- `policy`: `priority`
- `q_learning`: cac he so hoc
- `rewards`: diem thuong/phat (`reach_destination`, `time_penalty`, `run_red_penalty`, `wrong_way_penalty`, `collision_penalty`)
- `ui`: kich thuoc cua so + tieu de

`spawn_rates` duoc tu dong can chinh theo `grid.size`, sau do duoc map vao nhieu diem vao tren bien ban do:
- neu thieu phan tu: tu dong bo sung
- neu du phan tu: tu dong cat bot
- neu bo trong/khong khai bao: mac dinh la `1` cho moi diem spawn

Tuong tu, mot so tham so phu thuoc cung duoc tu can chinh:
- `grid.max_cars_per_cell` duoc co dinh ve `1` de dam bao moi o chi co 1 xe
- `grid.intersection_spacing_cells`: khoang cach giua 2 nut giao lien ke (don vi: o), toi thieu `1`
- cac gia tri `q_learning` (`learning_rate`, `discount_factor`, `epsilon_*`) se tu clamp vao mien hop le
