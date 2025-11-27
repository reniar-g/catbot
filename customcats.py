"""
Custom Cat Implementations
These are additional test cats for testing.
"""

import random
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

class RAIAdaptiveCat(Cat):
    """Tracks player movement patterns and switches between evasion, mirroring, and counter strategies."""

    def _get_sprite_path(self) -> str:
        return "images/trainer-dp.png"

    def move(self) -> None:
        # Initialize tracking
        if not hasattr(self, 'action_history'):
            self.action_history = []
            self.last_distances = []
        
        # Track player actions
        if self.last_player_action is not None:
            self.action_history.append(self.last_player_action)
            if len(self.action_history) > 10:
                self.action_history.pop(0)
        
        # Track distances
        current_distance = abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1])
        self.last_distances.append(current_distance)
        if len(self.last_distances) > 5:
            self.last_distances.pop(0)
        
        # Calculate aggression score (% of moves coming closer)
        aggression = 0.5
        if len(self.last_distances) >= 2:
            aggression = sum(1 for i in range(1, len(self.last_distances)) 
                           if self.last_distances[i] < self.last_distances[i-1]) / (len(self.last_distances) - 1)
        
        possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # High aggression: evade, mirror, or counter
        if aggression > 0.6:
            roll = random.random()
            if roll < 0.4:
                # Evade to max distance
                best_move = max(possible_moves, key=lambda m: 
                    abs(min(max(0, self.pos[0] + m[0]), self.grid_size - 1) - self.player_pos[0]) + 
                    abs(min(max(0, self.pos[1] + m[1]), self.grid_size - 1) - self.player_pos[1]))
            elif roll < 0.7 and len(self.action_history) >= 3:
                # Mirror their 3rd-to-last move
                best_move = [(-1, 0), (1, 0), (0, -1), (0, 1)][self.action_history[-3]]
            else:
                # Counter with opposite move
                if self.last_player_action is not None:
                    best_move = [(-1, 0), (1, 0), (0, -1), (0, 1)][[1, 0, 3, 2][self.last_player_action]]
                else:
                    best_move = random.choice(possible_moves)
        
        # Low aggression: approach or random
        elif aggression < 0.3:
            if random.random() < 0.6:
                # Move closer
                best_move = min(possible_moves, key=lambda m: 
                    abs(min(max(0, self.pos[0] + m[0]), self.grid_size - 1) - self.player_pos[0]) + 
                    abs(min(max(0, self.pos[1] + m[1]), self.grid_size - 1) - self.player_pos[1]))
            else:
                best_move = random.choice(possible_moves)
        
        # Neutral: unpredictable mix
        else:
            roll = random.random()
            if roll < 0.3:
                best_move = random.choice(possible_moves)
            elif roll < 0.6:
                # Evade
                best_move = max(possible_moves, key=lambda m: 
                    abs(min(max(0, self.pos[0] + m[0]), self.grid_size - 1) - self.player_pos[0]) + 
                    abs(min(max(0, self.pos[1] + m[1]), self.grid_size - 1) - self.player_pos[1]))
            else:
                # Follow most common player direction
                if len(self.action_history) >= 5:
                    most_common = max(set(self.action_history[-5:]), key=self.action_history[-5:].count)
                    best_move = [(-1, 0), (1, 0), (0, -1), (0, 1)][most_common]
                else:
                    best_move = random.choice(possible_moves)
        
        # Apply move
        self.pos[0] = min(max(0, self.pos[0] + best_move[0]), self.grid_size - 1)
        self.pos[1] = min(max(0, self.pos[1] + best_move[1]), self.grid_size - 1)


class RAIKnightCat(Cat):
    """Distance maximizer with predictable phase cycling and occasional diagonal moves."""

    def _get_sprite_path(self) -> str:
        return "images/trainer-dp.png"

    def move(self) -> None:
        # Initialize state
        if not hasattr(self, 'move_counter'):
            self.move_counter = 0
        
        self.move_counter += 1
        if self.move_counter > 8:
            self.move_counter = 0
        
        phase = 0 if self.move_counter <= 4 else 1
        
        # Phase 0: Diagonal moves + standard moves
        if phase == 0:
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        # Phase 1: Standard cardinal moves only
        else:
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Always maximize distance
        best_move = max(moves, key=lambda m: 
            abs(self.pos[0] + m[0] - self.player_pos[0]) + abs(self.pos[1] + m[1] - self.player_pos[1]))
        
        self.pos[0] = min(max(0, self.pos[0] + best_move[0]), self.grid_size - 1)
        self.pos[1] = min(max(0, self.pos[1] + best_move[1]), self.grid_size - 1)


class RAIChaosCat(Cat):
    """Chaotic movement alternating between calculated evasion and pure random behavior."""

    def _get_sprite_path(self) -> str:
        return "images/trainer-dp.png"

    def move(self) -> None:
        # Initialize state
        if not hasattr(self, 'move_counter'):
            self.move_counter = 0
        
        self.move_counter += 1
        if self.move_counter > 8:
            self.move_counter = 0
        
        # Alternate between calculated and chaotic modes
        mode = 0 if self.move_counter <= 4 else 1
        
        possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        if mode == 0:
            # Calculated evasion: maximize distance
            best_move = max(possible_moves, key=lambda m: 
                abs(self.pos[0] + m[0] - self.player_pos[0]) + abs(self.pos[1] + m[1] - self.player_pos[1]))
        else:
            # Pure chaos: random 70% of the time, calculated 30%
            if random.random() < 0.7:
                best_move = random.choice(possible_moves)
            else:
                best_move = max(possible_moves, key=lambda m: 
                    abs(self.pos[0] + m[0] - self.player_pos[0]) + abs(self.pos[1] + m[1] - self.player_pos[1]))
        
        # Apply move with boundary checking
        new_r = min(max(0, self.pos[0] + best_move[0]), self.grid_size - 1)
        new_c = min(max(0, self.pos[1] + best_move[1]), self.grid_size - 1)
        self.pos[0], self.pos[1] = new_r, new_c


class RAIHybridCat(Cat):
    """Hybrid cat combining all 5 default cat behaviors in adaptive sequence."""

    def _get_sprite_path(self) -> str:
        return "images/trainer-dp.png"

    def move(self) -> None:
        # Initialize state
        if not hasattr(self, 'behavior_counter'):
            self.behavior_counter = 0
        
        self.behavior_counter += 1
        if self.behavior_counter > 40:
            self.behavior_counter = 1
        
        # Cycle through 5 default cat behaviors (8 moves each)
        behavior = (self.behavior_counter - 1) // 8
        current_distance = abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1])
        
        # 0: Batmeow - stationary
        if behavior == 0:
            return
        
        # 1: Mittens - random movement
        elif behavior == 1:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(dirs)
            d = dirs[0]
            new_r = min(max(0, self.pos[0] + d[0]), self.grid_size - 1)
            new_c = min(max(0, self.pos[1] + d[1]), self.grid_size - 1)
            self.pos[0] = new_r
            self.pos[1] = new_c
        
        # 2: Paotsin - adaptive evasion based on player proximity
        elif behavior == 2:
            if current_distance > 4:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                d = dirs[0]
                new_r = min(max(0, self.pos[0] + d[0]), self.grid_size - 1)
                new_c = min(max(0, self.pos[1] + d[1]), self.grid_size - 1)
                self.pos[0] = new_r
                self.pos[1] = new_c
                return
            
            possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(possible_moves)
            best_move = None
            best_distance = None
            
            for dr, dc in possible_moves:
                new_r = min(max(0, self.pos[0] + dr), self.grid_size - 1)
                new_c = min(max(0, self.pos[1] + dc), self.grid_size - 1)
                
                distance = abs(new_r - self.player_pos[0]) + abs(new_c - self.player_pos[1])
                
                if best_distance is None:
                    best_move = (dr, dc)
                    best_distance = distance
                elif self.player_moved_closer():
                    if distance > best_distance:
                        best_move = (dr, dc)
                        best_distance = distance
                else:
                    if distance < best_distance:
                        best_move = (dr, dc)
                        best_distance = distance
            
            if self.player_moved_closer():
                self.pos[0] = min(max(0, self.pos[0] + best_move[0]), self.grid_size - 1)
                self.pos[1] = min(max(0, self.pos[1] + best_move[1]), self.grid_size - 1)
            elif random.random() < 0.65:
                self.pos[0] = min(max(0, self.pos[0] + best_move[0]), self.grid_size - 1)
                self.pos[1] = min(max(0, self.pos[1] + best_move[1]), self.grid_size - 1)
            else:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                d = dirs[0]
                new_r = min(max(0, self.pos[0] + d[0]), self.grid_size - 1)
                new_c = min(max(0, self.pos[1] + d[1]), self.grid_size - 1)
                self.pos[0] = new_r
                self.pos[1] = new_c
        
        # 3: Peekaboo - teleportation when adjacent
        elif behavior == 3:
            is_adjacent = (
                abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1]) == 1
            )
            
            if not is_adjacent:
                return
            
            if (self.pos[0] == 0 and self.player_pos[0] == 0 and self.player_pos[1] == self.pos[1] - 1) or (self.pos[1] == 0 and self.player_pos[1] == 0 and self.player_pos[0] == self.pos[0] - 1):
                return
            
            edge_positions = []
            for i in range(self.grid_size):
                edge_positions.extend([
                    (0, i), 
                    (self.grid_size-1, i), 
                    (i, 0),          
                    (i, self.grid_size-1)   
                ])
            
            edge_positions = list(set(edge_positions))
            
            safe_positions = []
            for pos in edge_positions:
                if abs(pos[0] - self.player_pos[0]) + abs(pos[1] - self.player_pos[1]) > 1:
                    safe_positions.append(pos)
            
            if safe_positions:
                new_pos = random.choice(safe_positions)
                self.pos[0] = new_pos[0]
                self.pos[1] = new_pos[1]
        
        # 4: Squiddyboi - multi-space jumps
        else:
            is_adjacent = (
                abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1]) == 1
            )
            if not is_adjacent:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                
                for d in dirs:
                    new_r = min(max(0, self.pos[0] + d[0]), self.grid_size - 1)
                    new_c = min(max(0, self.pos[1] + d[1]), self.grid_size - 1)
                    
                    would_be_adjacent = (
                        abs(new_r - self.player_pos[0]) + abs(new_c - self.player_pos[1]) == 1
                    )
                    
                    if not would_be_adjacent:
                        self.pos[0] = new_r
                        self.pos[1] = new_c
                        return
                return
            
            dr = self.player_pos[0] - self.pos[0]
            dc = self.player_pos[1] - self.pos[1]
            if dr != 0:
                dr = dr // abs(dr)
            if dc != 0:
                dc = dc // abs(dc)
            
            target_r = self.player_pos[0] + 2 * dr
            target_c = self.player_pos[1] + 2 * dc
            
            if (0 <= target_r < self.grid_size and 0 <= target_c < self.grid_size):
                self.pos[0] = target_r
                self.pos[1] = target_c
                return
            
            target_r = self.player_pos[0] + dr
            target_c = self.player_pos[1] + dc
            
            if (0 <= target_r < self.grid_size and 0 <= target_c < self.grid_size):
                self.pos[0] = target_r
                self.pos[1] = target_c
                return
            
            new_r = min(max(0, self.pos[0] - dr), self.grid_size - 1)
            new_c = min(max(0, self.pos[1] - dc), self.grid_size - 1)
            self.pos[0] = new_r
            self.pos[1] = new_c


# Export custom cat types
cat_types = {
    "patrol": KANPatrolCat,
    "diagonal": KANDiagonalCat,
    "adaptive": RAIAdaptiveCat,
    "knight": RAIKnightCat,
    "chaos": RAIChaosCat,
    "hybrid": RAIHybridCat
}
