import math
import random
import colorama as colorama

"""
    Preparations class - class which contains function to build all proces around main algorithm
"""


class Preparations(object):
    """ Check every other cell around main cell """

    @classmethod
    def set_every_element_around_direction(cls, coords, grid_size):
        y, x = coords
        width, height = grid_size
        dirs = []

        if y > 0:
            dirs.append((-1, 0))
        if y < height - 1:
            dirs.append((1, 0))
        if x > 0:
            dirs.append((0, -1))
        if x < width - 1:
            dirs.append((0, 1))

        return dirs

    """Find coefficients of grid"""

    @classmethod
    def initialize_grid_with_coefficients(cls, grid_size, cell_values):
        grid_of_factor = []
        for _ in range(grid_size[1]):
            row = []
            for _ in range(grid_size[0]):
                row.append(set(cell_values))
            grid_of_factor.append(row)
        return grid_of_factor

    @classmethod
    def check_which_cell_have_to_be_near(cls, view_grid):
        height = len(view_grid)
        width = len(view_grid[0])
        list_of_this_elements = set()
        how_many_of_proper_elements_is_in_grid = {}

        for y, row in enumerate(view_grid):
            for x, cell in enumerate(row):
                if cell not in how_many_of_proper_elements_is_in_grid:
                    how_many_of_proper_elements_is_in_grid[cell] = 0
                how_many_of_proper_elements_is_in_grid[cell] += 1

                for next_cell_coordinate in Preparations.set_every_element_around_direction((y, x), (width, height)):
                    next_cell = view_grid[y + next_cell_coordinate[0]][x + next_cell_coordinate[1]]
                    list_of_this_elements.add((cell, next_cell, next_cell_coordinate))

        return list_of_this_elements, how_many_of_proper_elements_is_in_grid

    """
        Output - output generated grid 
    """

    @classmethod
    def output(cls, generated_grid, colors_to_show):
        for row in generated_grid:
            output_array = []
            for val in row:
                color = colors_to_show[val]
                output_array.append(color + val + colorama.Style.RESET_ALL)

            print("".join(output_array))


"""
    WaveFunctionCollapse class - class containing wave function collapse algorithm
"""


class WaveFunctionCollapse(object):

    def __init__(self, grid_setup_for_working_on, weights_of_characters):
        self.grid_setup_for_working_on = grid_setup_for_working_on
        self.weights_of_characters = weights_of_characters

    """
        set_coefficient - set coefficient to all cell 
    """

    @classmethod
    def set_coefficient(cls, input_grid_size, weights_of_characters):
        coefficient_of_main_grid = Preparations.initialize_grid_with_coefficients(input_grid_size,
                                                                                  list(weights_of_characters.keys()))
        return WaveFunctionCollapse(coefficient_of_main_grid, weights_of_characters)

    """
        get - get specific cell
    """

    def get(self, coordinate):
        y, x = coordinate
        return self.grid_setup_for_working_on[y][x]

    """
        get_collapsed - get next iteration
    """

    def get_collapsed(self, co_ords):
        opts = self.get(co_ords)
        assert (len(opts) == 1)
        return next(iter(opts))

    """
        get_all_collapsed - get all row of iterations
    """

    def get_all_collapsed(self):
        height = len(self.grid_setup_for_working_on)
        width = len(self.grid_setup_for_working_on[0])

        collapsed = []
        for y in range(height):
            row: list[str] = []
            for x in range(width):
                row.append(self.get_collapsed((y, x)))
            collapsed.append(row)

        return collapsed

    """
        shannon_entropy - shannon entropy algorithm of specific coordinate 
    """

    def shannon_entropy(self, co_ords):
        y, x = co_ords

        sum_of_weights = 0
        sum_of_weight_log_weights = 0
        for opt in self.grid_setup_for_working_on[y][x]:
            weight = self.weights_of_characters[opt]
            sum_of_weights += weight
            sum_of_weight_log_weights += weight * math.log(weight)

        return math.log(sum_of_weights) - (sum_of_weight_log_weights / sum_of_weights)

    """
        is_fully_collapsed - is generated row ending or not
    """

    def is_fully_collapsed(self):
        for row in self.grid_setup_for_working_on:
            for sq in row:
                if len(sq) > 1:
                    return False
        return True

    """
        collapse - chose tile for getting coordinates
    """

    def collapse(self, co_ords):
        y, x = co_ords
        opts = self.grid_setup_for_working_on[y][x]
        filtered_tiles_with_weights = [(tile, weight) for tile, weight in self.weights_of_characters.items() if
                                       tile in opts]

        total_weights = sum([weight for _, weight in filtered_tiles_with_weights])
        rnd = random.random() * total_weights

        chosen = filtered_tiles_with_weights[0][0]
        for tile, weight in filtered_tiles_with_weights:
            rnd -= weight
            if rnd < 0:
                chosen = tile
                break

        self.grid_setup_for_working_on[y][x] = {chosen}

    """
        constrain - remove tile which do not have to be in specific cell
    """

    def constrain(self, co_ords, forbidden_tile):
        y, x = co_ords
        self.grid_setup_for_working_on[y][x].remove(forbidden_tile)


"""
    ModelLearn Class - class specify for initialize WaveFunctionCollapse and to build model which is use for 
    generating new grid based on forwarding grid.
"""


class ModelLearn(object):

    def __init__(self, grid_size, weights, labels):
        self.grid_size = grid_size
        self.labels = labels
        self.wavefunction = WaveFunctionCollapse.set_coefficient(grid_size, weights)

    """
        run - run model
    """

    def run(self):
        while not self.wavefunction.is_fully_collapsed():
            self.iterate()
        return self.wavefunction.get_all_collapsed()

    """
        iterate - step by step walk through every cell of grid
    """

    def iterate(self):
        co_ords = self.find_minimal_entropy_coordinates()
        self.wavefunction.collapse(co_ords)
        self.propagate(co_ords)

    """
        check - function to test if every element of the list of 'labels' contain elements we want
    """

    def check(self, tile1, tile2, direction):
        return (tile1, tile2, direction) in self.labels

    """
        propagate - check a nearby elements if entropy of those elements was changed
    """
    def propagate(self, co_ords):
        stack = [co_ords]

        while len(stack) > 0:
            cur_co_ords = stack.pop()
            cur_possible_tiles = self.wavefunction.get(cur_co_ords)
            for d in Preparations.set_every_element_around_direction(cur_co_ords, self.grid_size):
                other_co_ords = (cur_co_ords[0] + d[0], cur_co_ords[1] + d[1])

                for other_tile in set(self.wavefunction.get(other_co_ords)):
                    other_tile_is_possible = any([
                        self.check(cur_tile, other_tile, d) for cur_tile in cur_possible_tiles
                    ])
                    if not other_tile_is_possible:
                        self.wavefunction.constrain(other_co_ords, other_tile)
                        stack.append(other_co_ords)

    """
        find_minimal_entropy_coordinates - evaluate entropy of set grid
    """

    def find_minimal_entropy_coordinates(self):
        min_entropy = None
        min_co_ords = (0, 0)

        width, height = self.grid_size
        for y in range(height):
            for x in range(width):
                if len(self.wavefunction.get((y, x))) == 1:
                    continue

                entropy = self.wavefunction.shannon_entropy((y, x))
                noise = entropy - (random.random() / 1000)
                if min_entropy is None or noise < min_entropy:
                    min_entropy = noise
                    min_co_ords = (y, x)
        return min_co_ords


if __name__ == "__main__":
    grid = [
        ["Z", "Z", "Z", "Z", "Z"],
        ["Z", "Z", "P", "P", "P"],
        ["P", "Z", "P", "W", "W"],
        ["P", "P", "P", "W", "W"],
        ["W", "W", "W", "W", "W"],
    ]

    colors = {
        'Z': colorama.Fore.GREEN,
        'W': colorama.Fore.BLUE,
        'P': colorama.Fore.YELLOW,
    }

    labeled_coordinates_with_cell_determination = Preparations.check_which_cell_have_to_be_near(grid)
    model = ModelLearn((10, 10), labeled_coordinates_with_cell_determination[1],
                       labeled_coordinates_with_cell_determination[0])

    Preparations.output(model.run(), colors)
