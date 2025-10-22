"""
A* Path Planning Algorithm
Finds the shortest path from start to goal while avoiding obstacles
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass(order=True)
class Node:
    """Node for A* algorithm"""
    f: float  # Total cost (g + h)
    g: float = field(compare=False)  # Cost from start
    h: float = field(compare=False)  # Heuristic to goal
    pos: Tuple[int, int] = field(compare=False)  # (row, col)
    parent: Optional['Node'] = field(compare=False, default=None)


class AStarPlanner:
    """
    A* path planner for grid-based navigation
    """

    def __init__(self, inflation_radius: float = 0.3):
        """
        Initialize A* planner

        Args:
            inflation_radius: Safety distance from obstacles (meters)
        """
        self.inflation_radius = inflation_radius

    def plan(self,
             occupancy_grid,
             start_pos: Tuple[float, float],
             goal_pos: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal

        Args:
            occupancy_grid: OccupancyGrid object with map data
            start_pos: Start position (x, y) in world coordinates
            goal_pos: Goal position (x, y) in world coordinates

        Returns:
            List of waypoints [(x, y), ...] in world coordinates, or None if no path found
        """
        # Convert world coordinates to grid coordinates
        start_row, start_col = occupancy_grid.world_to_map(start_pos[0], start_pos[1])
        goal_row, goal_col = occupancy_grid.world_to_map(goal_pos[0], goal_pos[1])

        # Check if start and goal are valid
        if not occupancy_grid.is_valid(start_row, start_col):
            print(f"Start position ({start_pos}) is outside map bounds")
            return None

        if not occupancy_grid.is_valid(goal_row, goal_col):
            print(f"Goal position ({goal_pos}) is outside map bounds")
            return None

        # Inflate obstacles for safety
        inflated_map = self._inflate_obstacles(occupancy_grid)

        # Adjust start or goal if they fall inside inflated obstacles
        if inflated_map[start_row, start_col] > 50:
            new_start = self._find_nearest_free_cell(inflated_map, (start_row, start_col))
            if new_start is None:
                print(f"Start position is in obstacle and no nearby free cell found")
                return None
            print(f"Start position is in obstacle; shifting to nearest free cell at {new_start}")
            start_row, start_col = new_start

        if inflated_map[goal_row, goal_col] > 50:
            new_goal = self._find_nearest_free_cell(inflated_map, (goal_row, goal_col))
            if new_goal is None:
                print(f"Goal position is in obstacle and no nearby free cell found")
                return None
            print(f"Goal position is in obstacle; shifting to nearest free cell at {new_goal}")
            goal_row, goal_col = new_goal

        # Run A* algorithm
        path_indices = self._astar(inflated_map, (start_row, start_col), (goal_row, goal_col))

        if path_indices is None:
            print("No path found")
            return None

        # Convert grid coordinates back to world coordinates
        path_world = []
        for row, col in path_indices:
            x, y = occupancy_grid.map_to_world(row, col)
            path_world.append((x, y))

        return path_world

    def _inflate_obstacles(self, occupancy_grid) -> np.ndarray:
        """
        Inflate obstacles by inflation_radius for safety margin

        Args:
            occupancy_grid: OccupancyGrid object

        Returns:
            Inflated map data
        """
        inflated_map = occupancy_grid.data.copy()
        inflation_cells = int(np.ceil(self.inflation_radius / occupancy_grid.resolution))

        # Find all occupied cells
        occupied = np.where(occupancy_grid.data > 50)

        # Inflate around each occupied cell
        for row, col in zip(occupied[0], occupied[1]):
            for dr in range(-inflation_cells, inflation_cells + 1):
                for dc in range(-inflation_cells, inflation_cells + 1):
                    # Check if within circular radius
                    if dr*dr + dc*dc <= inflation_cells*inflation_cells:
                        r, c = row + dr, col + dc
                        if occupancy_grid.is_valid(r, c):
                            inflated_map[r, c] = max(inflated_map[r, c], 60)

        return inflated_map

    def _find_nearest_free_cell(self,
                                grid_map: np.ndarray,
                                start: Tuple[int, int],
                                max_search_cells: int = 50) -> Optional[Tuple[int, int]]:
        """Find the nearest free cell to the start location."""
        from collections import deque

        height, width = grid_map.shape
        visited = set()
        queue = deque()

        queue.append((start[0], start[1], 0))
        visited.add((start[0], start[1]))

        while queue:
            row, col, dist = queue.popleft()

            if grid_map[row, col] <= 50:
                return (row, col)

            if dist >= max_search_cells:
                continue

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc, dist + 1))

        return None

    def _astar(self,
               grid_map: np.ndarray,
               start: Tuple[int, int],
               goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* algorithm implementation

        Args:
            grid_map: 2D grid map (inflated)
            start: Start position (row, col)
            goal: Goal position (row, col)

        Returns:
            List of grid positions [(row, col), ...] or None
        """
        height, width = grid_map.shape

        # Open set (priority queue)
        open_set = []
        start_node = Node(
            f=self._heuristic(start, goal),
            g=0,
            h=self._heuristic(start, goal),
            pos=start,
            parent=None
        )
        heapq.heappush(open_set, start_node)

        # Closed set
        closed_set = set()

        # Best g-score for each position
        g_score = {start: 0}

        # 8-connected neighbors
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        while open_set:
            current = heapq.heappop(open_set)

            # Goal reached
            if current.pos == goal:
                return self._reconstruct_path(current)

            # Skip if already visited
            if current.pos in closed_set:
                continue

            closed_set.add(current.pos)

            # Explore neighbors
            for dr, dc in neighbors:
                neighbor_pos = (current.pos[0] + dr, current.pos[1] + dc)
                row, col = neighbor_pos

                # Check bounds
                if not (0 <= row < height and 0 <= col < width):
                    continue

                # Check if obstacle
                if grid_map[row, col] > 50:
                    continue

                # Check if already visited
                if neighbor_pos in closed_set:
                    continue

                # Calculate movement cost (diagonal = sqrt(2), straight = 1)
                move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                tentative_g = current.g + move_cost

                # Check if this is a better path
                if neighbor_pos not in g_score or tentative_g < g_score[neighbor_pos]:
                    g_score[neighbor_pos] = tentative_g
                    h = self._heuristic(neighbor_pos, goal)
                    neighbor_node = Node(
                        f=tentative_g + h,
                        g=tentative_g,
                        h=h,
                        pos=neighbor_pos,
                        parent=current
                    )
                    heapq.heappush(open_set, neighbor_node)

        return None  # No path found

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Euclidean distance heuristic

        Args:
            pos1: Position 1 (row, col)
            pos2: Position 2 (row, col)

        Returns:
            Euclidean distance
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """
        Reconstruct path from goal node to start

        Args:
            node: Goal node

        Returns:
            List of positions from start to goal
        """
        path = []
        current = node
        while current is not None:
            path.append(current.pos)
            current = current.parent

        return path[::-1]  # Reverse to get start -> goal
