import random
from ..bot_control import Move
from enum import Enum
import numpy as np

DEBUG = False

MOVE_TO_VECTOR = {
    Move.UP: np.array([0, 1],  dtype=np.int16),
    Move.RIGHT: np.array([1, 0],  dtype=np.int16),
    Move.LEFT: np.array([-1, 0], dtype=np.int16),
    Move.DOWN: np.array([0, -1], dtype=np.int16),

    # Don't care about the stay move here
    # Move.STAY: np.array([0, 0],  dtype=np.int16)
}

class State(Enum):
    SEARCHING = 1
    TRAVELLING = 2
    CREATING = 3    
    ADMIRING = 4

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

class LearoundoDaVinci:
    MIN_RADIUS = 1
    MAX_RADIUS = 4
    SEARCHES_PER_ROUND = 50

    def __init__(self):
        self.state = State.SEARCHING        
        
        self.target = None
        self.circle = None
        self.point_list = None

        self.radius_to_search = 2

        self.point_lists = { }
        self.circle_masks = { }
        for radius in range(self.MIN_RADIUS, self.MAX_RADIUS + 1):
            self.point_lists[radius] = self.generate_point_list_cache(radius)
            self.circle_masks[radius] = self.generate_circle_mask(radius)

    def get_name(self):
        # Thanks ChatGPT for the name suggestion
        return "LeaRoundo Da Vinci"

    def get_contributor(self):
        return "Hein"

    def position_hash(self, position):
        return f"{position[0]}.{position[1]}"

    def is_in_circle(self, position, center, radius):
        offset = [position[0] - center[0], position[1] - center[1]]
        return sum(pow(element, 2) for element in offset) <= radius**2

    def generate_point_list_cache(self, radius):
        """
        Used by circle spot finding algoritm. Returns a regular list
        of all points in a circle. All relative to zero
        """
        cache = { }
        for x in range(-radius - 1, radius + 1):
            for y in range(-radius - 1, radius + 1):
                position = [x, y]
                if self.is_in_circle(position, [0, 0], radius + 0.5):
                    cache[self.position_hash(position)] = np.array(position)
        return cache

    def generate_circle_mask(self, radius):
        points = self.point_lists[radius]
        size = 1 + radius * 2
        mask = np.zeros((size, size), dtype=np.bool8)
        center = (size - 1) // 2
        center = np.array([center, center], dtype=np.int16)
        for x in range(size):
            for y in range(size):
                position = np.subtract(np.array([x, y]), center)
                if self.position_hash(position) in points:
                    mask[y, x] = True
        return mask

    def find_space_for_circle(self, position, grid, radius):
        """
        Try to find a space in the grid where we can draw
        a circle of radius, ideally close to position
        """
        offset_x = random.randint(0, self.size - 1)
        offset_y = random.randint(0, self.size - 1)

        # Mask all the zero tiles so we don't overwrite them
        masked_array = np.ma.masked_where(grid == 0, grid)
        # Now do the modulus so we can determine which tiles we can overwrite
        # And remove the mask again afterwards
        normalized_grid = ((self.id - masked_array) % 3).data
        # Now get a grid with `True` where the tile is zero, or we can overwrite it
        open_grid = np.logical_or(grid == 0, normalized_grid == 2)

        searches = 0
        for i in range(0, self.size):
            for j in range(0, self.size):
                x = (i + offset_x) % self.size
                y = (j + offset_y) % self.size

                left = x - radius
                right = x + radius
                top = y + radius
                bottom = y - radius
                
                # Will this circle fit in the canvas?
                if left <= 0 or right >= self.size or \
                    bottom <= 0 or top >= self.size:
                    continue

                # Can we draw an entire circle here?
                # Do this by masking the section in the grid
                # we are looking at, and see if all those
                # tiles are overwritable
                masked_open_grid = np.ma.masked_where(
                    self.circle_masks[radius] == False, 
                    open_grid[bottom:(top+1), left:(right+1)])
                if np.all(masked_open_grid):
                    # Found a spot!
                    return np.array([x, y])
                
                # Make sure we don't search for too long
                searches += 1
                if searches > self.SEARCHES_PER_ROUND:
                    return None

    def move_to_target(self, careful=False):
        if not careful:
            if self.target[0] > self.position[0]:
                return Move.RIGHT
            elif self.target[0] < self.position[0]:
                return Move.LEFT
            elif self.target[1] > self.position[1]:
                return Move.UP
            else:
                return Move.DOWN
        else:
            # We must remain within the current circle, meaning our
            # next step must be inside the circle point list
            assert(self.circle is not None)
            normalized_position = np.subtract(self.position, self.circle.center)
            point_list = self.point_lists[self.circle.radius]
            if self.target[0] > self.position[0] and \
                    self.position_hash([
                        normalized_position[0] + 1,
                        normalized_position[1]
                    ]) in point_list:
                return Move.RIGHT
            elif self.target[0] < self.position[0] and \
                    self.position_hash([
                        normalized_position[0] - 1,
                        normalized_position[1]
                    ]) in point_list:
                return Move.LEFT
            elif self.target[1] > self.position[1] and \
                    self.position_hash([
                        normalized_position[0], 
                        normalized_position[1] + 1
                    ]) in point_list:
                return Move.UP
            else:
                return Move.DOWN
            

    def find_next_fill_move(self):
        """
        We are busy filling the circle. Find a next move which will
        continue filling
        """
        assert(self.point_list is not None)
        assert(len(self.point_list) != 0)
        
        # We're not.Then try to find a random tile next to us that we can 
        # step onto that's still in the circle. We do it using the normalized
        # position so that it's easy to compare to our point list
        normalized_position = np.subtract(self.position, self.circle.center)
        for move in random.sample(list(MOVE_TO_VECTOR.keys()), len(MOVE_TO_VECTOR)):
            new_position = np.add(normalized_position, MOVE_TO_VECTOR[move])
            # Do we still need to step on this tile?
            if self.position_hash(new_position) in self.point_list:
                return move

        # We still haven't found a move. If we haven't found a valid spot to move to 
        # in the next move, then we will find a random spot on the circle and move there
        if self.target is not None and np.array_equal(self.position, self.target):
            # We've reached the target. We should find a new one. This is a weird
            # case though. TODO Should we reach here? When should we?
            self.target = None            
        if self.target is None:
            # Don't have a random target yet. Get one
            self.target = np.add(self.circle.center, random.choice(list(self.point_list.values())))        

        # If we reach here then a target is set. So we need to 
        # find the next correct move to reach this target. It's+

        # always possible to move within a circle to any point
        # within a circle without stepping outside. We just need
        # to be careful.
        # TODO Be careful
        return self.move_to_target(careful = True)

    def set_state(self, state):
        if DEBUG:
            name = next(name for name, value in vars(State).items() if value == state)
            print(f"State: {name}")
        self.state = state

    def determine_next_move(self, grid, enemies, game_info):
        if game_info.current_round == 1:
            self.set_state(State.SEARCHING)
            self.size = game_info.grid_size

        move = None

        if self.state == State.SEARCHING:
            # Search for a random circle size
            radius_to_search = random.choice(list(range(self.MIN_RADIUS, self.MAX_RADIUS + 1)))
            self.target = self.find_space_for_circle(self.position, grid, radius_to_search)
            if self.target is not None:
                self.circle = Circle(self.target, radius_to_search)
                self.point_list = self.point_lists[self.circle.radius].copy()
                self.set_state(State.TRAVELLING)
        
        if self.state == State.TRAVELLING:
            assert(self.target is not None)
            if np.array_equal(self.position, self.target):
                self.set_state(State.CREATING)
            else:
                if self.target[0] > self.position[0]:
                    move = Move.RIGHT
                elif self.target[0] < self.position[0]:
                    move = Move.LEFT
                elif self.target[1] > self.position[1]:
                    move = Move.UP
                else:
                    move = Move.DOWN

        if self.state == State.CREATING:
            if len(self.point_list) == 0:
                # Done creating
                self.target = self.circle.center
                self.set_state(State.ADMIRING)
            else:
                move = self.find_next_fill_move()

        if self.state == State.ADMIRING:
            if np.array_equal(self.position, self.target):
                self.circle = None
                self.set_state(State.SEARCHING)
            else:
                move = self.move_to_target()
            
        if move is not None and self.circle is not None:
            new_position = np.subtract(np.add(self.position, MOVE_TO_VECTOR[move]), self.circle.center)
            self.point_list.pop(self.position_hash(new_position), None)
        else:
            move = Move.STAY

        return move