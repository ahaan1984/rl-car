from dataclasses import dataclass


@dataclass
class Constants:
    WIDTH = 800
    HEIGHT = 600
    TITLE = 'Procedural Race Track'
    STARTING_GRID_TILE = 'static/grid_tile.png'
    START_TILE_HEIGHT = 10
    START_TILE_WIDTH = 10
    KERB_TILE = 'static/kerb_tile.png'
    KERB_TILE_HEIGHT = 7
    KERB_TILE_WIDTH = 12
    WHITE = [255, 255, 255]
    BLACK = [0, 0, 0]
    RED = [255, 0, 0]
    BLUE = [0, 0, 255]
    GRASS_GREEN = [58, 156, 53]
    GREY = [186, 182, 168]
    KERB_PLACEMENT_X_CORRECTION = 5
    KERB_PLACEMENT_Y_CORRECTION = 4
    KERB_POINT_ANGLE_OFFSET = 5
    STEP_TO_NEXT_KERB_POINT = 4
    CHECKPOINT_POINT_ANGLE_OFFSET = 3
    CHECKPOINT_MARGIN = 5
    TRACK_POINT_ANGLE_OFFSET = 3

    MIN_POINTS = 20
    MAX_POINTS = 30

    SPLINE_POINTS = 1000

    MARGIN = 50
    MIN_DISTANCE = 20
    MAX_DISPLACEMENT = 80
    DIFFICULTY = 0.1
    DISTANCE_BETWEEN_POINTS = 20
    MAX_ANGLE = 90

    MIN_KERB_ANGLE = 20
    MAX_KERB_ANGLE = 90

    TRACK_WIDTH = 40

    FULL_CORNER_NUM_POINTS = 19

    N_CHECKPOINTS = 10
    COOL_TRACK_SEEDS = [
        911,
        639620465,
        666574559,
        689001243,
        608068482,
        1546,
        8,
        83,
        945,
        633,
        10,
        23,
        17,
        123,
        1217,
        12,
        5644,
        5562,
        2317,
        1964,
        95894,
        95521
    ]