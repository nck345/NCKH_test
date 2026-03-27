"""
visualizer.py
--------------
Hiển thị mô phỏng bằng Pygame.
Xe di chuyển liên tục trên cạnh.

Phím tắt:
  N     = Chuyển sang epoch mới ngay lập tức
  S     = Nhập số epoch muốn skip tới (trong console)
  SPACE = Tạm dừng / tiếp tục
  ESC   = Thoát
"""

import pygame
import sys

from config import WINDOW_WIDTH, WINDOW_HEIGHT, FPS

# --------------- Hằng số hiển thị ---------------

COLOR_BG = (25, 25, 35)
COLOR_ROAD = (70, 75, 90)
COLOR_GREEN = (0, 200, 80)
COLOR_RED = (220, 50, 50)
COLOR_NODE_BORDER = (180, 180, 200)
COLOR_TEXT = (230, 230, 230)
COLOR_TEXT_DIM = (150, 150, 170)
COLOR_EPOCH = (100, 200, 255)
COLOR_HOTKEY = (255, 220, 100)

CAR_COLORS = [
    (255, 210, 60), (60, 180, 255), (255, 100, 150),
    (100, 255, 150), (255, 160, 60), (180, 120, 255),
    (255, 255, 130), (100, 220, 220), (255, 130, 130),
    (130, 255, 200), (200, 200, 100), (220, 160, 220),
]
COLOR_CAR_DONE = (200, 200, 200)

NODE_RADIUS = 14
CAR_RADIUS = 9


def run_visualization(model):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("MAS Traffic Simulation – Q-Learning")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("consolas", 20, bold=True)
    font_large = pygame.font.SysFont("consolas", 16, bold=True)
    font_medium = pygame.font.SysFont("consolas", 13)
    font_small = pygame.font.SysFont("consolas", 11)

    G = model.G
    paused = False
    skip_msg = ""
    skip_msg_timer = 0

    # Input trong Pygame (không dùng terminal)
    input_mode = False     # Đang nhập số epoch?
    input_text = ""        # Chuỗi đang gõ

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                # === Chế độ nhập số epoch ===
                if input_mode:
                    if event.key == pygame.K_RETURN:
                        # Xác nhận
                        try:
                            target = int(input_text)
                            if target > model.epoch:
                                # Vẽ thông báo đang skip
                                screen.fill(COLOR_BG)
                                msg = font_title.render(
                                    f"Skipping to epoch {target}...",
                                    True, COLOR_EPOCH,
                                )
                                r = msg.get_rect(center=(
                                    WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
                                ))
                                screen.blit(msg, r)
                                pygame.display.flip()
                                # Skip
                                model.skip_to_epoch(target)
                                skip_msg = f">> Da skip toi epoch {model.epoch}"
                                skip_msg_timer = 120
                            else:
                                skip_msg = "Epoch khong hop le!"
                                skip_msg_timer = 90
                        except ValueError:
                            skip_msg = "Nhap sai!"
                            skip_msg_timer = 60
                        input_mode = False
                        input_text = ""
                        paused = False

                    elif event.key == pygame.K_ESCAPE:
                        # Hủy nhập
                        input_mode = False
                        input_text = ""
                        paused = False

                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]

                    elif event.unicode.isdigit():
                        input_text += event.unicode

                    continue  # Không xử lý phím tắt khác khi đang nhập

                # === Phím tắt bình thường ===
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_n:
                    # Epoch đã xong → bắt đầu epoch mới
                    if model.epoch_done:
                        model.reset_epoch()
                        skip_msg = f">> Epoch {model.epoch}"
                        skip_msg_timer = 60

                elif event.key == pygame.K_s:
                    input_mode = True
                    input_text = ""
                    paused = True

        # Bước simulation
        if not paused:
            model.step()

        screen.fill(COLOR_BG)

        # === Tiêu đề + Epoch ===
        title = font_title.render(
            f"Multi-Agent Traffic Simulation", True, COLOR_TEXT
        )
        screen.blit(title, (15, 8))

        epoch_text = font_large.render(
            f"Epoch: {model.epoch}   Step: {model.step_count}"
            f"   Reached: {model.get_reached_count()}/{model.num_cars}",
            True, COLOR_EPOCH,
        )
        screen.blit(epoch_text, (15, 32))

        # === Phím tắt ===
        keys_text = font_small.render(
            "[N] Epoch moi   [S] Skip epoch   [SPACE] Tam dung   [ESC] Thoat",
            True, COLOR_HOTKEY,
        )
        screen.blit(keys_text, (15, 54))

        # === Thông báo tạm ===
        if skip_msg_timer > 0:
            skip_msg_timer -= 1
            ms = font_large.render(skip_msg, True, COLOR_EPOCH)
            screen.blit(ms, (WINDOW_WIDTH // 2 - ms.get_width() // 2, 75))

        # === 1) Vẽ đường ===
        for u, v in G.edges():
            p1 = G.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]
            pygame.draw.line(screen, COLOR_ROAD, p1, p2, 2)

        # === 2) Vẽ nút + đèn ===
        for node_id in G.nodes:
            data = G.nodes[node_id]
            pos = data["pos"]
            pygame.draw.circle(screen, COLOR_NODE_BORDER, pos, NODE_RADIUS + 2)
            color = COLOR_GREEN if data["traffic_light"] == "green" else COLOR_RED
            pygame.draw.circle(screen, color, pos, NODE_RADIUS)
            label = font_small.render(str(node_id), True, (0, 0, 0))
            rect = label.get_rect(center=pos)
            screen.blit(label, rect)

        # === 3) Vẽ xe ===
        agents_list = list(model.agents)
        for idx, agent in enumerate(agents_list):
            car_color = (CAR_COLORS[idx % len(CAR_COLORS)]
                         if not agent.reached else COLOR_CAR_DONE)

            px, py = agent.get_pixel_pos()
            draw_pos = (int(px), int(py))

            pygame.draw.circle(screen, (255, 255, 255), draw_pos, CAR_RADIUS + 2)
            pygame.draw.circle(screen, car_color, draw_pos, CAR_RADIUS)

            # ID
            id_label = font_small.render(str(idx), True, (0, 0, 0))
            id_rect = id_label.get_rect(center=draw_pos)
            screen.blit(id_label, id_rect)

            # Trạng thái nhỏ
            if agent.reached:
                st = "DONE"
            elif agent.on_edge:
                st = f"v={agent.speed:.2f}"
            else:
                st = f"@{agent.current_node}"
            st_surf = font_small.render(st, True, car_color)
            screen.blit(st_surf, (draw_pos[0] + CAR_RADIUS + 3, draw_pos[1] - 6))

        # === 4) Bảng thống kê (dưới trái) ===
        panel_y = WINDOW_HEIGHT - 115
        panel_surf = pygame.Surface((400, 105), pygame.SRCALPHA)
        panel_surf.fill((35, 35, 50, 220))
        screen.blit(panel_surf, (10, panel_y))
        pygame.draw.rect(screen, COLOR_TEXT_DIM,
                         (10, panel_y, 400, 105), 1, border_radius=6)

        ep_info = [
            f"Epoch: {model.epoch}    Step: {model.step_count}",
            f"Xe den dich: {model.get_reached_count()} / {model.num_cars}",
            f"Diem TB epoch nay: {model.get_avg_reward():+.2f}",
        ]
        # Hiện avg reward epoch trước
        if model.epoch_rewards:
            ep_info.append(
                f"Diem TB epoch truoc: {model.epoch_rewards[-1]:+.2f}"
            )

        for i, line in enumerate(ep_info):
            surf = font_medium.render(line, True, COLOR_TEXT)
            screen.blit(surf, (22, panel_y + 8 + i * 20))

        # Tạm dừng / Input overlay / Epoch kết thúc
        if input_mode:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))

            box_w, box_h = 400, 120
            box_x = (WINDOW_WIDTH - box_w) // 2
            box_y = (WINDOW_HEIGHT - box_h) // 2
            pygame.draw.rect(screen, (40, 40, 60),
                             (box_x, box_y, box_w, box_h), border_radius=10)
            pygame.draw.rect(screen, COLOR_EPOCH,
                             (box_x, box_y, box_w, box_h), 2, border_radius=10)

            prompt = font_large.render(
                f"Skip toi epoch (hien tai: {model.epoch})", True, COLOR_TEXT
            )
            screen.blit(prompt, (box_x + 20, box_y + 15))

            input_box = pygame.Rect(box_x + 80, box_y + 50, 240, 35)
            pygame.draw.rect(screen, (60, 60, 80), input_box, border_radius=5)
            pygame.draw.rect(screen, COLOR_EPOCH, input_box, 2, border_radius=5)
            txt = font_title.render(input_text + "|", True, COLOR_EPOCH)
            screen.blit(txt, (input_box.x + 10, input_box.y + 5))

            hint = font_small.render(
                "[Enter] Xac nhan    [Esc] Huy", True, COLOR_TEXT_DIM
            )
            screen.blit(hint, (box_x + 80, box_y + 95))

        elif model.epoch_done:
            # Epoch kết thúc → hiện overlay chờ
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            screen.blit(overlay, (0, 0))

            done_text = font_title.render(
                f"EPOCH {model.epoch} KET THUC!", True, COLOR_EPOCH
            )
            r = done_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 15))
            screen.blit(done_text, r)

            hint_text = font_large.render(
                "[N] Epoch moi    [S] Skip toi epoch", True, COLOR_HOTKEY
            )
            r2 = hint_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
            screen.blit(hint_text, r2)

        elif paused:
            ps = font_title.render("|| TAM DUNG ||", True, COLOR_HOTKEY)
            r = ps.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            screen.blit(ps, r)

        # === 5) Bảng xe (phải) ===
        list_x = WINDOW_WIDTH - 520
        list_y = 70
        list_w = 510
        list_h = 28 + len(agents_list) * 20
        ls = pygame.Surface((list_w, list_h), pygame.SRCALPHA)
        ls.fill((35, 35, 50, 200))
        screen.blit(ls, (list_x, list_y))
        pygame.draw.rect(screen, COLOR_TEXT_DIM,
                         (list_x, list_y, list_w, list_h), 1, border_radius=4)

        header = font_medium.render(
            " ID  Tu->Den  Speed  Reward  DenDo VaCham  Status", True, COLOR_TEXT
        )
        screen.blit(header, (list_x + 5, list_y + 4))

        for idx, agent in enumerate(agents_list):
            cc = (CAR_COLORS[idx % len(CAR_COLORS)]
                  if not agent.reached else COLOR_CAR_DONE)
            y = list_y + 24 + idx * 20
            pygame.draw.rect(screen, cc, (list_x + 6, y + 2, 10, 10))

            st = "DONE" if agent.reached else ("EDGE" if agent.on_edge else "NODE")
            line = (f" {idx:>2}  {agent.origin:>2}->{agent.destination:<2}"
                    f"  {agent.speed:>5.2f}"
                    f"  {agent.accumulated_reward:>+7.1f}"
                    f"  {agent.red_light_count:>5}"
                    f"  {agent.collision_count:>6}"
                    f"  {st:>10}")
            surf = font_small.render(line, True, COLOR_TEXT)
            screen.blit(surf, (list_x + 20, y))

        # === 6) Chú thích ===
        leg_x = WINDOW_WIDTH - 280
        leg_y = WINDOW_HEIGHT - 100
        items = [
            (COLOR_GREEN, "Nga tu (Xanh)"),
            (COLOR_RED, "Nga tu (Do)"),
            (CAR_COLORS[0], "Xe dang di"),
            (COLOR_CAR_DONE, "Xe da den dich"),
        ]
        lgs = pygame.Surface((270, 90), pygame.SRCALPHA)
        lgs.fill((35, 35, 50, 200))
        screen.blit(lgs, (leg_x, leg_y))
        pygame.draw.rect(screen, COLOR_TEXT_DIM,
                         (leg_x, leg_y, 270, 90), 1, border_radius=4)
        for i, (c, t) in enumerate(items):
            y = leg_y + 8 + i * 20
            pygame.draw.circle(screen, c, (leg_x + 14, y + 6), 5)
            screen.blit(font_small.render(t, True, COLOR_TEXT), (leg_x + 28, y))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()
