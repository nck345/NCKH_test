"""
Turn-based traffic simulation core powered by Mesa.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Iterable

import mesa

from config_loader import AppConfig, CarPriority


A_LEFT = "left"
A_UP = "up"
A_RIGHT = "right"
A_DOWN = "down"
A_WAIT = "wait"

MOVE_ACTIONS = (A_LEFT, A_UP, A_RIGHT, A_DOWN)
ALL_ACTIONS = (A_LEFT, A_UP, A_RIGHT, A_DOWN, A_WAIT)
ACTION_ORDER = (A_UP, A_RIGHT, A_DOWN, A_LEFT)


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    turns: int
    reached: int
    total: int
    avg_reward: float


class CarAgent(mesa.Agent):
    """Discrete-turn car agent with Q-learning + priority fallback."""

    def __init__(
        self,
        model: "TrafficModel",
        car_id: int,
        spawn_turn: int,
        start_x: int,
        start_y: int,
        destination_x: int,
        destination_y: int,
    ) -> None:
        super().__init__(model)
        self.car_id = car_id

        self.spawn_turn = spawn_turn
        self.start_x = start_x
        self.start_y = start_y
        self.destination_x = destination_x
        self.destination_y = destination_y

        self.x = start_x
        self.y = start_y
        self.started = False
        self.reached = False

        self.accumulated_reward = 0.0
        self.blocked_count = 0
        self.wait_count = 0
        self.red_light_count = 0
        self.wrong_way_count = 0
        self.collision_count = 0
        self.turns_alive = 0

        self.last_action = A_WAIT
        self.last_wrong_way = False

        self.q_table: defaultdict[tuple[tuple, str], float] = defaultdict(float)
        self.epsilon = self.model.config.q_learning.epsilon_start

    @property
    def is_active(self) -> bool:
        return self.started and not self.reached

    @property
    def cell(self) -> tuple[int, int]:
        return self.x, self.y

    def reset_for_epoch(
        self,
        spawn_turn: int,
        start_x: int,
        start_y: int,
        destination_x: int,
        destination_y: int,
    ) -> None:
        self.spawn_turn = spawn_turn
        self.start_x = start_x
        self.start_y = start_y
        self.destination_x = destination_x
        self.destination_y = destination_y
        self.x = start_x
        self.y = start_y
        self.started = False
        self.reached = False
        self.accumulated_reward = 0.0
        self.blocked_count = 0
        self.wait_count = 0
        self.red_light_count = 0
        self.wrong_way_count = 0
        self.collision_count = 0
        self.turns_alive = 0
        self.last_action = A_WAIT
        self.last_wrong_way = False

    def step(self) -> None:
        if self.reached:
            return

        if not self.started:
            if self.model.turn_count < self.spawn_turn:
                return
            if self.model.get_occupancy(self.cell) >= self.model.cell_capacity:
                self.blocked_count += 1
                return
            self.started = True
            self.model.add_to_occupancy(self.cell)

        self.turns_alive += 1
        state = self._get_state()
        valid_actions = self.model.get_valid_actions(self)
        action = self._choose_action(state, valid_actions)
        self.last_action = action

        next_pos, moved, _, ran_red, wrong_way = self.model.try_move(self, action)
        reward = self.model.config.rewards.time_penalty
        self.last_wrong_way = False

        if not moved:
            if action == A_WAIT:
                self.wait_count += 1
            else:
                self.blocked_count += 1
        else:
            if ran_red:
                self.red_light_count += 1
                reward += self.model.config.rewards.run_red_penalty
            if wrong_way:
                self.wrong_way_count += 1
                self.last_wrong_way = True
                reward += self.model.config.rewards.wrong_way_penalty
            self.model.move_agent(self, next_pos)
            if self.y == self.destination_y and self.x == self.destination_x:
                self.reached = True
                self.model.remove_from_occupancy(next_pos)
                reward += self.model.config.rewards.reach_destination

        self.accumulated_reward += reward
        next_state = self._get_state()
        next_valid_actions = self.model.get_valid_actions(self)
        self._update_q(state, action, reward, next_state, next_valid_actions)
        self._decay_epsilon()

    def _get_state(self) -> tuple:
        left_info = self.model.peek_cell(self, A_LEFT)
        up_info = self.model.peek_cell(self, A_UP)
        right_info = self.model.peek_cell(self, A_RIGHT)
        down_info = self.model.peek_cell(self, A_DOWN)
        flow = self.model.get_cell_flow(self.cell)
        return (
            self.x,
            self.y,
            self.destination_x,
            self.destination_y,
            flow,
            left_info,
            up_info,
            right_info,
            down_info,
        )

    def _choose_action(self, state: tuple, valid_actions: list[str]) -> str:
        if not valid_actions:
            return A_WAIT

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_values = {action: self.q_table[(state, action)] for action in valid_actions}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        if len(best_actions) == 1:
            return best_actions[0]

        fallback = self.model.priority_fallback(self, valid_actions)
        if fallback is not None:
            return fallback
        return random.choice(best_actions)

    def _update_q(
        self,
        state: tuple,
        action: str,
        reward: float,
        next_state: tuple,
        next_valid_actions: Iterable[str],
    ) -> None:
        alpha = self.model.config.q_learning.learning_rate
        gamma = self.model.config.q_learning.discount_factor

        old_q = self.q_table[(state, action)]
        if self.reached:
            next_max = 0.0
        else:
            candidates = [self.q_table[(next_state, a)] for a in next_valid_actions]
            next_max = max(candidates) if candidates else 0.0

        self.q_table[(state, action)] = old_q + alpha * (reward + gamma * next_max - old_q)

    def _decay_epsilon(self) -> None:
        cfg = self.model.config.q_learning
        self.epsilon = max(cfg.epsilon_min, self.epsilon * cfg.epsilon_decay)


class TrafficModel(mesa.Model):
    """Environment-managed traffic simulation with epoch support."""

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.size = config.grid.size
        if self.size < 5:
            raise ValueError("grid.size must be >= 5 for a multi-intersection layout.")

        self.cell_capacity = config.grid.max_cars_per_cell
        self.turn_count = 0
        self.epoch = 1
        self.epoch_done = False

        self.last_epoch_summary: EpochSummary | None = None
        self.epoch_summaries: list[EpochSummary] = []

        self.traffic_lights: dict[tuple[int, int, int, int], str] = {}
        self._light_timer = 0
        self._occupancy: dict[tuple[int, int], int] = {}
        self.intersection_spacing_cells = self.config.grid.intersection_spacing_cells

        self.horizontal_pair_starts = self._build_lane_pair_starts(self.size)
        self.vertical_pair_starts = self._build_lane_pair_starts(self.size)

        self.horizontal_pairs = tuple((start, start + 1) for start in self.horizontal_pair_starts)
        self.vertical_pairs = tuple((start, start + 1) for start in self.vertical_pair_starts)

        self.row_flow: dict[int, str] = {}
        for right_row, left_row in self.horizontal_pairs:
            self.row_flow[right_row] = A_RIGHT
            self.row_flow[left_row] = A_LEFT

        self.col_flow: dict[int, str] = {}
        for up_col, down_col in self.vertical_pairs:
            self.col_flow[up_col] = A_UP
            self.col_flow[down_col] = A_DOWN

        self.road_rows = tuple(sorted(self.row_flow))
        self.road_cols = tuple(sorted(self.col_flow))

        self.intersection_blocks = self._build_intersection_blocks()
        self.intersection_cells = frozenset(self._build_intersection_cells())
        self.road_cells = frozenset(self._build_road_cells())
        self._cell_to_block: dict[tuple[int, int], tuple[int, int, int, int]] = (
            self._build_cell_to_block_map()
        )

        self.endpoint_labels = self._build_endpoint_labels()
        self.entry_points, self.exit_points = self._build_boundary_points()
        if not self.entry_points or not self.exit_points:
            raise ValueError("Unable to create entry/exit points for this grid size.")

        self._spawn_schedule: list[tuple[int, int, int, int, int]] = self._build_spawn_schedule()

        self._build_lights()
        self._create_cars()
        self._rebuild_occupancy()

    def _build_lane_pair_starts(self, length: int) -> tuple[int, ...]:
        starts: list[int] = []
        idx = 1
        stride = 2 + self.intersection_spacing_cells
        while idx + 1 < length - 1:
            starts.append(idx)
            idx += stride  # lane width 2 + configurable gap

        if not starts:
            fallback = max(1, min(length - 3, (length // 2) - 1))
            starts.append(fallback)
        return tuple(starts)

    def _build_intersection_blocks(self) -> tuple[tuple[int, int, int, int], ...]:
        blocks: list[tuple[int, int, int, int]] = []
        for x0, x1 in self.vertical_pairs:
            for y0, y1 in self.horizontal_pairs:
                blocks.append((x0, x1, y0, y1))
        blocks.sort(key=lambda item: (item[2], item[0]))
        return tuple(blocks)

    def _build_intersection_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for x0, x1, y0, y1 in self.intersection_blocks:
            cells.add((x0, y0))
            cells.add((x0, y1))
            cells.add((x1, y0))
            cells.add((x1, y1))
        return cells

    def _build_cell_to_block_map(self) -> dict[tuple[int, int], tuple[int, int, int, int]]:
        mapping: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        for block in self.intersection_blocks:
            x0, x1, y0, y1 = block
            mapping[(x0, y0)] = block
            mapping[(x0, y1)] = block
            mapping[(x1, y0)] = block
            mapping[(x1, y1)] = block
        return mapping

    def _build_road_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for y in self.road_rows:
            for x in range(self.size):
                cells.add((x, y))
        for x in self.road_cols:
            for y in range(self.size):
                cells.add((x, y))
        return cells

    def _build_endpoint_labels(self) -> dict[tuple[int, int], str]:
        labels: dict[tuple[int, int], str] = {}
        west_points = sorted([(0, y) for y in self.road_rows], key=lambda p: p[1])
        east_points = sorted([(self.size - 1, y) for y in self.road_rows], key=lambda p: p[1])
        north_points = sorted([(x, 0) for x in self.road_cols], key=lambda p: p[0])
        south_points = sorted([(x, self.size - 1) for x in self.road_cols], key=lambda p: p[0])

        for idx, pos in enumerate(west_points, start=1):
            labels[pos] = f"W{idx}"
        for idx, pos in enumerate(east_points, start=1):
            labels[pos] = f"E{idx}"
        for idx, pos in enumerate(north_points, start=1):
            labels[pos] = f"N{idx}"
        for idx, pos in enumerate(south_points, start=1):
            labels[pos] = f"S{idx}"
        return labels

    def _build_boundary_points(self) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        entry: list[tuple[int, int]] = []
        exit_: list[tuple[int, int]] = []

        west_points = sorted([(0, y) for y in self.road_rows], key=lambda p: p[1])
        east_points = sorted([(self.size - 1, y) for y in self.road_rows], key=lambda p: p[1])
        north_points = sorted([(x, 0) for x in self.road_cols], key=lambda p: p[0])
        south_points = sorted([(x, self.size - 1) for x in self.road_cols], key=lambda p: p[0])

        for point in west_points:
            (entry if self.get_cell_flow(point) == A_RIGHT else exit_).append(point)
        for point in east_points:
            (entry if self.get_cell_flow(point) == A_LEFT else exit_).append(point)
        for point in north_points:
            (entry if self.get_cell_flow(point) == A_DOWN else exit_).append(point)
        for point in south_points:
            (entry if self.get_cell_flow(point) == A_UP else exit_).append(point)

        return tuple(entry), tuple(exit_)

    def point_label(self, pos: tuple[int, int]) -> str:
        return self.endpoint_labels.get(pos, f"{pos[0]},{pos[1]}")

    def get_cell_flow(self, pos: tuple[int, int]) -> str | None:
        x, y = pos
        if pos in self.intersection_cells:
            return None
        if y in self.row_flow:
            return self.row_flow[y]
        if x in self.col_flow:
            return self.col_flow[x]
        return None

    def _create_cars(self) -> None:
        for car_id, schedule in enumerate(self._spawn_schedule):
            spawn_turn, start_x, start_y, dest_x, dest_y = schedule
            CarAgent(
                self,
                car_id,
                spawn_turn,
                start_x,
                start_y,
                dest_x,
                dest_y,
            )

    def _build_spawn_schedule(self) -> list[tuple[int, int, int, int, int]]:
        lane_count = len(self.entry_points)
        rates = list(self.config.cars.spawn_rates)
        if not rates:
            rates = [1]
        lane_rates = [rates[idx % len(rates)] for idx in range(lane_count)]
        if sum(lane_rates) <= 0:
            lane_rates[0] = 1

        schedule: list[tuple[int, int, int, int, int]] = []
        no_cars_left = self.config.cars.total_cars
        turn = 1

        while no_cars_left > 0:
            for lane_idx, start_pos in enumerate(self.entry_points):
                if no_cars_left <= 0:
                    break
                rate = lane_rates[lane_idx]
                for _ in range(rate):
                    if no_cars_left <= 0:
                        break
                    dest_pos = random.choice(self.exit_points)
                    if len(self.exit_points) > 1:
                        while dest_pos == start_pos:
                            dest_pos = random.choice(self.exit_points)

                    start_x, start_y = start_pos
                    dest_x, dest_y = dest_pos
                    schedule.append((turn, start_x, start_y, dest_x, dest_y))
                    no_cars_left -= 1
                    if not self.config.cars.multiple_cars_per_turn:
                        turn += 1
            if self.config.cars.multiple_cars_per_turn:
                turn += 1

        return schedule

    def _alternate_light(self, state: str) -> str:
        return "Red" if state == "Green" else "Green"

    def _build_lights(self) -> None:
        self.traffic_lights.clear()
        base_state = self.config.traffic_lights.initial_state
        for block_idx, block in enumerate(self.intersection_blocks):
            block_state = base_state if block_idx % 2 == 0 else self._alternate_light(base_state)
            self.traffic_lights[block] = block_state
        self._light_timer = 0

    def step(self) -> None:
        if self.epoch_done:
            return

        self.turn_count += 1
        self._update_traffic_lights()
        self.agents.shuffle_do("step")

        if self.all_reached() or self.turn_count >= self.config.simulation.max_turns_per_epoch:
            self._finish_epoch()

    def _update_traffic_lights(self) -> None:
        self._light_timer += 1
        if self._light_timer < self.config.traffic_lights.switch_interval:
            return
        self._light_timer = 0
        for pos, state in list(self.traffic_lights.items()):
            self.traffic_lights[pos] = self._alternate_light(state)

    def _finish_epoch(self) -> None:
        self.epoch_done = True
        self.last_epoch_summary = EpochSummary(
            epoch=self.epoch,
            turns=self.turn_count,
            reached=self.get_reached_count(),
            total=self.config.cars.total_cars,
            avg_reward=self.get_avg_reward(),
        )
        self.epoch_summaries.append(self.last_epoch_summary)

    def reset_epoch(self) -> None:
        self.epoch += 1
        self.turn_count = 0
        self.epoch_done = False

        self._spawn_schedule = self._build_spawn_schedule()
        self._build_lights()
        self._occupancy.clear()

        for agent, schedule in zip(self.agents, self._spawn_schedule):
            spawn_turn, start_x, start_y, dest_x, dest_y = schedule
            agent.reset_for_epoch(
                spawn_turn,
                start_x,
                start_y,
                dest_x,
                dest_y,
            )

    def skip_to_epoch(self, target_epoch: int) -> None:
        """
        Fast-forward simulation to the beginning of target epoch.
        """
        if target_epoch <= self.epoch:
            return

        while self.epoch < target_epoch:
            if self.epoch_done:
                self.reset_epoch()
                continue
            self.step()

    def skip_current_epoch(self) -> None:
        """
        Skip the current epoch immediately and start the next epoch.
        """
        if not self.epoch_done:
            self._finish_epoch()
        self.reset_epoch()

    def all_reached(self) -> bool:
        return all(agent.reached for agent in self.agents)

    def get_reached_count(self) -> int:
        return sum(1 for agent in self.agents if agent.reached)

    def get_avg_reward(self) -> float:
        rewards = [agent.accumulated_reward for agent in self.agents]
        return sum(rewards) / len(rewards) if rewards else 0.0

    def is_road_cell(self, x: int, y: int) -> bool:
        return (x, y) in self.road_cells

    def is_unavailable(self, x: int, y: int) -> bool:
        return not self.is_road_cell(x, y)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def get_light(self, x: int, y: int) -> str | None:
        block = self._cell_to_block.get((x, y))
        if block is None:
            return None
        return self.traffic_lights.get(block)

    def get_intersection_light(self, block: tuple[int, int, int, int]) -> str:
        return self.traffic_lights.get(block, self.config.traffic_lights.initial_state)

    def _get_block_reason(
        self,
        agent: CarAgent,
        x: int,
        y: int,
        action: str,
    ) -> str | None:
        if not self.in_bounds(x, y):
            return "out_of_bounds"
        if self.is_unavailable(x, y):
            return "not_road"
        if self.get_occupancy((x, y)) >= self.cell_capacity:
            return "occupied"
        return None

    def can_enter_cell(self, agent: CarAgent, x: int, y: int, action: str) -> bool:
        return self._get_block_reason(agent, x, y, action) is None

    def _light_allows_move(self, action: str, x: int, y: int) -> bool:
        light_state = self.get_light(x, y)
        if light_state is None:
            return True
        if action in (A_UP, A_DOWN):
            return light_state == "Green"
        if action in (A_LEFT, A_RIGHT):
            return light_state == "Red"
        return True

    def get_valid_actions(self, agent: CarAgent) -> list[str]:
        if not agent.started or agent.reached:
            return [A_WAIT]

        actions: list[str] = [A_WAIT]
        for action in MOVE_ACTIONS:
            nx, ny = self._next_pos(agent.cell, action)
            if self.can_enter_cell(agent, nx, ny, action):
                actions.append(action)
        return actions

    def _next_pos(self, pos: tuple[int, int], action: str) -> tuple[int, int]:
        x, y = pos
        if action == A_LEFT:
            return x - 1, y
        if action == A_RIGHT:
            return x + 1, y
        if action == A_UP:
            return x, y - 1
        if action == A_DOWN:
            return x, y + 1
        return x, y

    def _is_wrong_way_move(
        self,
        agent: CarAgent,
        action: str,
        next_pos: tuple[int, int],
    ) -> bool:
        if action not in MOVE_ACTIONS:
            return False

        current_flow = self.get_cell_flow(agent.cell)
        next_flow = self.get_cell_flow(next_pos)

        if current_flow is not None and action != current_flow:
            return True
        if next_flow is not None and action != next_flow:
            return True
        return False

    def try_move(
        self,
        agent: CarAgent,
        action: str,
    ) -> tuple[tuple[int, int], bool, str | None, bool, bool]:
        if action == A_WAIT:
            return agent.cell, False, None, False, False

        nx, ny = self._next_pos(agent.cell, action)
        reason = self._get_block_reason(agent, nx, ny, action)
        if reason is not None:
            return agent.cell, False, reason, False, False

        ran_red = not self._light_allows_move(action, nx, ny)
        wrong_way = self._is_wrong_way_move(agent, action, (nx, ny))
        return (nx, ny), True, None, ran_red, wrong_way

    def move_agent(self, agent: CarAgent, new_pos: tuple[int, int]) -> None:
        old_pos = agent.cell
        self.remove_from_occupancy(old_pos)
        agent.x, agent.y = new_pos
        self.add_to_occupancy(new_pos)

    def _rebuild_occupancy(self) -> None:
        self._occupancy.clear()
        for agent in self.agents:
            if agent.started and not agent.reached:
                self.add_to_occupancy(agent.cell)

    def add_to_occupancy(self, pos: tuple[int, int]) -> None:
        self._occupancy[pos] = self._occupancy.get(pos, 0) + 1

    def remove_from_occupancy(self, pos: tuple[int, int]) -> None:
        if pos not in self._occupancy:
            return
        self._occupancy[pos] -= 1
        if self._occupancy[pos] <= 0:
            del self._occupancy[pos]

    def get_occupancy(self, pos: tuple[int, int]) -> int:
        return self._occupancy.get(pos, 0)

    def peek_cell(self, agent: CarAgent, action: str) -> tuple[int, int]:
        nx, ny = self._next_pos(agent.cell, action)
        if not self.in_bounds(nx, ny) or self.is_unavailable(nx, ny):
            return -1, 0
        light = self.get_light(nx, ny)
        light_code = 0 if light is None else (1 if light == "Green" else 2)
        return min(self.get_occupancy((nx, ny)), self.cell_capacity), light_code

    def priority_fallback(self, agent: CarAgent, valid_actions: list[str]) -> str | None:
        moves = [a for a in valid_actions if a in MOVE_ACTIONS]
        if not moves:
            return A_WAIT if A_WAIT in valid_actions else None

        priority = self.config.policy.priority
        order = self._directional_preference(agent)

        if priority in (CarPriority.NO_PRIORITY, CarPriority.GREEN_LIGHT):
            for action in order:
                if action in moves:
                    return action
            return moves[0]

        if priority == CarPriority.LOWER_TRAFFIC:
            scored: list[tuple[int, int, int, str]] = []
            for action in moves:
                nx, ny = self._next_pos(agent.cell, action)
                occupancy = self.get_occupancy((nx, ny))
                wrong_way_bias = 1 if self._is_wrong_way_move(agent, action, (nx, ny)) else 0
                direction_bias = order.index(action) if action in order else len(order)
                scored.append((occupancy, wrong_way_bias, direction_bias, action))
            scored.sort()
            return scored[0][3]

        return moves[0]

    def _directional_preference(self, agent: CarAgent) -> list[str]:
        scored: list[tuple[int, int, int, str]] = []
        current_flow = self.get_cell_flow(agent.cell)
        for action in ACTION_ORDER:
            nx, ny = self._next_pos(agent.cell, action)
            if not self.in_bounds(nx, ny):
                continue
            wrong_way = 1 if self._is_wrong_way_move(agent, action, (nx, ny)) else 0
            flow_bias = 0 if current_flow is None or action == current_flow else 1
            distance = abs(nx - agent.destination_x) + abs(ny - agent.destination_y)
            scored.append((wrong_way, flow_bias, distance, action))
        scored.sort(key=lambda item: (item[0], item[1], item[2], ACTION_ORDER.index(item[3])))
        return [action for _, _, _, action in scored]
