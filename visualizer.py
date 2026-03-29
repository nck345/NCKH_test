"""
Tkinter visualizer for the turn-based traffic simulator.
"""

from __future__ import annotations

import colorsys
import tkinter as tk
from tkinter import messagebox, simpledialog

from simulation_core import TrafficModel


class TrafficVisualizer:
    def __init__(self, model: TrafficModel):
        self.model = model
        self.config = model.config
        self._closed = False
        self._running = True

        self.root = tk.Tk()
        self.root.title(self.config.ui.title)
        self.root.geometry(
            f"{self.config.ui.window_width}x{self.config.ui.window_height}"
        )
        self.root.configure(bg="#1a1a2e")
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.epoch_var = tk.StringVar()
        self.turn_var = tk.StringVar()
        self.reached_var = tk.StringVar()
        self.reward_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.notice_var = tk.StringVar()
        self._notice_ticks = 0

        self._build_widgets()
        self._car_colors = self._build_car_palette(self.config.cars.total_cars)

        self.root.bind("<space>", lambda _: self.toggle_running())
        self.root.bind("<n>", lambda _: self.start_next_epoch())
        self.root.bind("<N>", lambda _: self.start_next_epoch())
        self.root.bind("<s>", lambda _: self.prompt_skip_epoch())
        self.root.bind("<S>", lambda _: self.prompt_skip_epoch())
        self.root.bind("<Escape>", lambda _: self._close())

        self._refresh_header()
        self._draw_scene()
        self.root.after(self.config.simulation.ms_per_turn, self._tick)

    def _build_widgets(self) -> None:
        header = tk.Frame(self.root, bg="#1a1a2e")
        header.pack(fill=tk.X, padx=10, pady=(10, 6))

        labels = [
            self.epoch_var,
            self.turn_var,
            self.reached_var,
            self.reward_var,
        ]
        for var in labels:
            tk.Label(
                header,
                textvariable=var,
                fg="#dfe6ff",
                bg="#1a1a2e",
                font=("Arial", 10, "bold"),
            ).pack(side=tk.LEFT, padx=(0, 18))

        tk.Label(
            header,
            textvariable=self.status_var,
            fg="#f7c948",
            bg="#1a1a2e",
            font=("Arial", 10, "bold"),
        ).pack(side=tk.RIGHT)

        tk.Label(
            self.root,
            textvariable=self.notice_var,
            fg="#8be9fd",
            bg="#1a1a2e",
            font=("Arial", 10, "bold"),
        ).pack(fill=tk.X, padx=10, pady=(0, 4))

        self.canvas = tk.Canvas(self.root, bg="#16213e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.canvas.bind("<Configure>", lambda _: self._draw_scene())

        controls = tk.Frame(self.root, bg="#1a1a2e")
        controls.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.pause_button = tk.Button(
            controls,
            text="Pause [Space]",
            command=self.toggle_running,
            width=14,
        )
        self.pause_button.pack(side=tk.LEFT, padx=(0, 8))

        self.next_epoch_button = tk.Button(
            controls,
            text="Next/Skip [N]",
            command=self.start_next_epoch,
            width=14,
        )
        self.next_epoch_button.pack(side=tk.LEFT, padx=(0, 8))

        self.skip_button = tk.Button(
            controls,
            text="Skip Epoch [S]",
            command=self.prompt_skip_epoch,
            width=14,
        )
        self.skip_button.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            controls,
            text="Quit [Esc]",
            command=self._close,
            width=12,
        ).pack(side=tk.LEFT)

    def _build_car_palette(self, count: int) -> list[str]:
        palette: list[str] = []
        for idx in range(max(count, 1)):
            hue = (idx * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            palette.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
        return palette

    def _tick(self) -> None:
        if self._closed:
            return

        if self._running and not self.model.epoch_done:
            self.model.step()

        if self.model.epoch_done:
            if self.config.simulation.manual_epoch_advance:
                self._running = False
            else:
                self.model.reset_epoch()
                self._running = True

        if self._notice_ticks > 0:
            self._notice_ticks -= 1
            if self._notice_ticks == 0:
                self.notice_var.set("")

        self._sync_controls()
        self._refresh_header()
        self._draw_scene()
        self.root.after(self.config.simulation.ms_per_turn, self._tick)

    def _sync_controls(self) -> None:
        self.pause_button.configure(
            text="Resume [Space]" if not self._running else "Pause [Space]"
        )
        self.next_epoch_button.configure(state=tk.NORMAL)

    def toggle_running(self) -> None:
        if self.model.epoch_done and self.config.simulation.manual_epoch_advance:
            return
        self._running = not self._running
        self._sync_controls()
        self._refresh_header()

    def start_next_epoch(self) -> None:
        self.model.skip_current_epoch()
        self._running = True
        self._set_notice(f"Started epoch {self.model.epoch}")
        self._sync_controls()
        self._refresh_header()
        self._draw_scene()

    def prompt_skip_epoch(self) -> None:
        was_running = self._running
        self._running = False
        self._sync_controls()

        target = simpledialog.askinteger(
            "Skip Epoch",
            f"Skip to epoch (current: {self.model.epoch}):",
            parent=self.root,
            minvalue=1,
        )
        if target is None:
            if was_running and not self.model.epoch_done:
                self._running = True
            self._sync_controls()
            self._refresh_header()
            return

        if target <= self.model.epoch:
            messagebox.showinfo(
                "Skip Epoch",
                f"Target epoch must be greater than current epoch ({self.model.epoch}).",
                parent=self.root,
            )
            if was_running and not self.model.epoch_done:
                self._running = True
            self._sync_controls()
            self._refresh_header()
            return

        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            self.model.skip_to_epoch(target)
            self._set_notice(f"Skipped to epoch {self.model.epoch}")
        finally:
            self.root.config(cursor="")

        if was_running and not self.model.epoch_done:
            self._running = True

        self._sync_controls()
        self._refresh_header()
        self._draw_scene()

    def _refresh_header(self) -> None:
        self.epoch_var.set(f"Epoch: {self.model.epoch}")
        self.turn_var.set(f"Turn: {self.model.turn_count}")
        self.reached_var.set(
            f"Reached: {self.model.get_reached_count()}/{self.config.cars.total_cars}"
        )
        self.reward_var.set(f"Avg Reward: {self.model.get_avg_reward():+.2f}")

        if self.model.epoch_done:
            self.status_var.set("Status: Epoch complete")
        elif self._running:
            self.status_var.set("Status: Running")
        else:
            self.status_var.set("Status: Paused")

    def _set_notice(self, message: str, ticks: int = 45) -> None:
        self.notice_var.set(message)
        self._notice_ticks = ticks

    def _draw_scene(self) -> None:
        canvas = self.canvas
        canvas.delete("all")

        w = max(canvas.winfo_width(), 100)
        h = max(canvas.winfo_height(), 100)
        size = self.model.size

        cell_size = min((w - 60) // size, (h - 120) // (size + 2))
        if cell_size < 8:
            return

        grid_w = size * cell_size
        grid_h = size * cell_size
        origin_x = (w - grid_w) // 2
        origin_y = (h - (grid_h + (2 * cell_size))) // 2 + cell_size

        self._draw_grid(origin_x, origin_y, cell_size)
        self._draw_unavailable_cells(origin_x, origin_y, cell_size)
        self._draw_traffic_lights(origin_x, origin_y, cell_size)
        self._draw_labels(origin_x, origin_y, cell_size)
        self._draw_cars(origin_x, origin_y, cell_size)
        self._draw_legend(origin_x, origin_y, cell_size)

        if self.model.epoch_done:
            self._draw_epoch_overlay(w, h)

    def _draw_grid(self, ox: int, oy: int, cell: int) -> None:
        size = self.model.size
        x2 = ox + size * cell
        y2 = oy + size * cell
        self.canvas.create_rectangle(ox, oy, x2, y2, fill="#111936", outline="#3b4a77", width=2)

        for i in range(size + 1):
            self.canvas.create_line(ox, oy + i * cell, x2, oy + i * cell, fill="#2a2a4a")
            self.canvas.create_line(ox + i * cell, oy, ox + i * cell, y2, fill="#2a2a4a")

    def _draw_unavailable_cells(self, ox: int, oy: int, cell: int) -> None:
        for y in range(1, self.model.size, 2):
            for x in range(1, self.model.size, 2):
                x1 = ox + x * cell
                y1 = oy + y * cell
                self.canvas.create_rectangle(
                    x1, y1, x1 + cell, y1 + cell, fill="#0f0f23", outline="#0f0f23"
                )

    def _draw_traffic_lights(self, ox: int, oy: int, cell: int) -> None:
        bar = max(cell // 10, 2)
        for (x, y), state in self.model.traffic_lights.items():
            x1 = ox + x * cell
            y1 = oy + y * cell

            top_bottom = "#00e676" if state == "Green" else "#ff1744"
            left_right = "#ff1744" if state == "Green" else "#00e676"

            self.canvas.create_rectangle(
                x1, y1, x1 + bar, y1 + cell, fill=left_right, outline=left_right
            )
            self.canvas.create_rectangle(
                x1 + cell - bar, y1, x1 + cell, y1 + cell, fill=left_right, outline=left_right
            )
            self.canvas.create_rectangle(
                x1, y1, x1 + cell, y1 + bar, fill=top_bottom, outline=top_bottom
            )
            self.canvas.create_rectangle(
                x1, y1 + cell - bar, x1 + cell, y1 + cell, fill=top_bottom, outline=top_bottom
            )

    def _draw_labels(self, ox: int, oy: int, cell: int) -> None:
        start_labels = ["A", "B", "C", "D"]
        dest_labels = ["M", "N", "O", "P"]

        for idx, x in enumerate(self.model.config.starting_point_x):
            sx1 = ox + x * cell
            sy1 = oy + self.model.size * cell
            self.canvas.create_rectangle(
                sx1, sy1, sx1 + cell, sy1 + cell, fill="#4a6fa5", outline="#5a7fb5"
            )
            self.canvas.create_text(
                sx1 + cell // 2,
                sy1 + cell // 2,
                text=start_labels[idx] if idx < len(start_labels) else str(idx),
                fill="white",
                font=("Arial", max(cell // 4, 8), "bold"),
            )

            dx1 = sx1
            dy1 = oy - cell
            self.canvas.create_rectangle(
                dx1, dy1, dx1 + cell, dy1 + cell, fill="#3a5f45", outline="#4a7055"
            )
            self.canvas.create_text(
                dx1 + cell // 2,
                dy1 + cell // 2,
                text=dest_labels[idx] if idx < len(dest_labels) else str(idx),
                fill="white",
                font=("Arial", max(cell // 4, 8), "bold"),
            )

    def _draw_cars(self, ox: int, oy: int, cell: int) -> None:
        slots = self.config.grid.no_cars_per_cell
        max_slots = slots * slots
        sub_size = max(cell // slots, 6)
        cars_per_cell: dict[tuple[int, int], int] = {}

        for agent in self.model.agents:
            if not agent.started or agent.reached:
                continue

            pos = agent.cell
            count = cars_per_cell.get(pos, 0)
            slot_index = min(count, max_slots - 1)
            cars_per_cell[pos] = count + 1

            oxi = (slot_index // slots) * sub_size
            oyi = (slot_index % slots) * sub_size

            x, y = pos
            base_x = ox + x * cell
            base_y = oy + y * cell

            rx1 = base_x + oxi
            ry1 = base_y + oyi
            rx2 = rx1 + sub_size
            ry2 = ry1 + sub_size

            color = self._car_colors[agent.car_id % len(self._car_colors)]
            self.canvas.create_rectangle(
                rx1, ry1, rx2, ry2, fill=color, outline="#ffffff", width=1
            )
            self.canvas.create_text(
                (rx1 + rx2) // 2,
                (ry1 + ry2) // 2,
                text=str(agent.car_id),
                fill="black",
                font=("Arial", max(sub_size // 3, 6), "bold"),
            )

    def _draw_legend(self, ox: int, oy: int, cell: int) -> None:
        x1 = ox
        y1 = oy + self.model.size * cell + cell + 8
        x2 = ox + self.model.size * cell
        y2 = y1 + 24
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#111936", outline="#2a2a4a")
        text = "Traffic light: Green=vertical pass, Red=horizontal pass | Car: square + ID"
        self.canvas.create_text(
            (x1 + x2) // 2,
            (y1 + y2) // 2,
            text=text,
            fill="#cdd6ff",
            font=("Arial", 9),
        )

    def _draw_epoch_overlay(self, w: int, h: int) -> None:
        summary = self.model.last_epoch_summary
        self.canvas.create_rectangle(
            0, 0, w, h, fill="#000000", stipple="gray50", outline=""
        )

        lines = [f"EPOCH {self.model.epoch} COMPLETE"]
        if summary is not None:
            lines.extend(
                [
                    f"Turns: {summary.turns}",
                    f"Reached: {summary.reached}/{summary.total}",
                    f"Average reward: {summary.avg_reward:+.2f}",
                ]
            )
        lines.append("Press [N] or click Next Epoch")

        for idx, line in enumerate(lines):
            self.canvas.create_text(
                w // 2,
                h // 2 - 45 + idx * 22,
                text=line,
                fill="#f7f7ff" if idx == 0 else "#d0d9ff",
                font=("Arial", 14 if idx == 0 else 12, "bold" if idx == 0 else "normal"),
            )

    def _close(self) -> None:
        self._closed = True
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_visualization(model: TrafficModel) -> None:
    TrafficVisualizer(model).run()
