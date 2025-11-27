"""
Custom Cat Implementations
These are additional test cats for testing.
"""

from cat_env import Cat


class KANPatrolCat(Cat):
    """Patrols in a rectangular circuit around the grid."""

    def _get_sprite_path(self) -> str:
        return "images/trainer-dp.png"

    def move(self) -> None:
        # Define patrol waypoints (clockwise around perimeter)
        if not hasattr(self, 'patrol_index'):
            self.patrol_index = 0
            self.waypoints = [
                (0, 0), (0, 3), (0, 7),
                (3, 7), (7, 7), (7, 3),
                (7, 0), (3, 0)
            ]

        # Get target waypoint
        target = self.waypoints[self.patrol_index]

        # Move toward target using Manhattan distance priority
        if self.pos[0] != target[0]:
            if self.pos[0] < target[0]:
                self.pos[0] += 1
            else:
                self.pos[0] -= 1
        elif self.pos[1] != target[1]:
            if self.pos[1] < target[1]:
                self.pos[1] += 1
            else:
                self.pos[1] -= 1
        else:
            # Reached waypoint, move to next
            self.patrol_index = (self.patrol_index + 1) % len(self.waypoints)


class KANDiagonalCat(Cat):
    """Escapes to the corner diagonal from the player's quadrant."""

    def _get_sprite_path(self) -> str:
        return "images/trainer-dp.png"

    def move(self) -> None:
        # Determine which quadrant player is in
        player_in_top = self.player_pos[0] < self.grid_size // 2
        player_in_left = self.player_pos[1] < self.grid_size // 2

        # Target opposite corner
        if player_in_top and player_in_left:
            target = (self.grid_size - 1, self.grid_size - 1)
        elif player_in_top and not player_in_left:
            target = (self.grid_size - 1, 0)
        elif not player_in_top and player_in_left:
            target = (0, self.grid_size - 1)
        else:
            target = (0, 0)

        # Move toward target corner
        dr = 0 if self.pos[0] == target[0] else (1 if self.pos[0] < target[0] else -1)
        dc = 0 if self.pos[1] == target[1] else (1 if self.pos[1] < target[1] else -1)

        self.pos[0] = min(max(0, self.pos[0] + dr), self.grid_size - 1)
        self.pos[1] = min(max(0, self.pos[1] + dc), self.grid_size - 1)


# Export custom cat types
cat_types = {
    "patrol": KANPatrolCat,
    "diagonal": KANDiagonalCat
}
