import pygame
import sys
import math
import random as rn
import numpy as np
from scipy.spatial import ConvexHull
from scipy import interpolate
import argparse

from utils import Constants

class TrackGeneration(Constants):
    """
    This class handles all logical/algorithmic aspects of track generation,
    including random point generation, shaping the track, smoothing, etc.
    """

    def __init__(self):
        self.min_points = Constants.MIN_POINTS
        self.max_points = Constants.MAX_POINTS
        self.margin = Constants.MARGIN
        self.min_distance = Constants.MIN_DISTANCE
        self.difficulty = Constants.DIFFICULTY
        self.max_displacement = Constants.MAX_DISPLACEMENT
        self.distance_between_points = Constants.DISTANCE_BETWEEN_POINTS
        self.max_angle = Constants.MAX_ANGLE
        self.min_kerb_angle = Constants.MIN_KERB_ANGLE
        self.max_kerb_angle = Constants.MAX_KERB_ANGLE

    def random_points(self):
        point_count = rn.randrange(self.min_points, self.max_points + 1, 1)
        points = []
        for i in range(point_count):
            x = rn.randrange(self.margin, Constants.WIDTH - self.margin + 1, 1)
            y = rn.randrange(self.margin, Constants.HEIGHT - self.margin + 1, 1)

            distances = list(filter(lambda d: d < self.min_distance, 
                [math.sqrt((p[0] - x) ** 2 + (p[1] - y) ** 2) for p in points]))

            if not distances:
                points.append((x, y))
        return np.array(points)

    def get_track_points(self, hull, points):
        """
        Get the original points from the random set that will be used
        as the track's starting shape
        """
        return np.array([points[hull.vertices[i]] for i in range(len(hull.vertices))])

    def shape_track(self, track_points):
        """
        Shapes the track by adding midpoints (with displacement),
        fixes angles, and pushes points apart.
        """
        # Double the amount of points with displacement
        track_set = [[0, 0] for _ in range(len(track_points) * 2)]
        for i in range(len(track_points)):
            displacement = math.pow(rn.random(), self.difficulty) * self.max_displacement
            disp = [displacement * x for x in self._make_rand_vector(2)]
            track_set[i * 2] = list(track_points[i])
            track_set[i * 2 + 1][0] = int(
                (track_points[i][0] + track_points[(i + 1) % len(track_points)][0]) / 2 + disp[0]
            )
            track_set[i * 2 + 1][1] = int(
                (track_points[i][1] + track_points[(i + 1) % len(track_points)][1]) / 2 + disp[1]
            )

        # refine shape
        for _ in range(3):
            track_set = self.fix_angles(track_set, self.max_angle)
            track_set = self.push_points_apart(track_set, self.distance_between_points)

        # ensure points lie within margins
        final_set = []
        for point in track_set:
            if point[0] < self.margin:
                point[0] = self.margin
            elif point[0] > (Constants.WIDTH - self.margin):
                point[0] = Constants.WIDTH - self.margin
            if point[1] < self.margin:
                point[1] = self.margin
            elif point[1] > (Constants.HEIGHT - self.margin):
                point[1] = Constants.HEIGHT - self.margin
            final_set.append(point)
        return final_set

    def get_corners_with_kerb(self, points):
        """
        Returns corner keypoints that require kerbs
        """
        require_kerb = []
        for i in range(len(points)):
            prev_point = i - 1 if i > 0 else len(points) - 1
            next_point = (i + 1) % len(points)
            px = points[prev_point][0] - points[i][0]
            py = points[prev_point][1] - points[i][1]
            pl = math.sqrt(px * px + py * py)
            px /= pl
            py /= pl
            nx = points[next_point][0] - points[i][0]
            ny = points[next_point][1] - points[i][1]
            nl = math.sqrt(nx * nx + ny * ny)
            nx /= nl
            ny /= nl
            a = math.atan(px * ny - py * nx)
            if (self.min_kerb_angle <= abs(math.degrees(a)) <= self.max_kerb_angle):
                require_kerb.append(points[i])
        return require_kerb

    def smooth_track(self, track_points):
        """
        Smooth the track points using periodic splines
        """
        x = np.array([p[0] for p in track_points])
        y = np.array([p[1] for p in track_points])
        # append starting point to make it periodic
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

        tck, _ = interpolate.splprep([x, y], s=0, per=True)
        xi, yi = interpolate.splev(np.linspace(0, 1, Constants.SPLINE_POINTS), tck)
        return [(int(xi[i]), int(yi[i])) for i in range(len(xi))]

    def get_full_corners(self, track_points, corners):
        """
        For each corner keypoint, get a range of points from the smoothed track
        """
        corners_in_track = self.get_corners_from_kp(track_points, corners)
        f_corners = []
        offset = Constants.FULL_CORNER_NUM_POINTS
        for corner in corners_in_track:
            i = track_points.index(corner)
            # repeat track to easily fetch surrounding points
            tmp_track_points = track_points + track_points + track_points
            f_corner = tmp_track_points[i + len(track_points) - 1 - offset:
                                        i + len(track_points) - 1 + offset]
            f_corners.append(f_corner)
        return f_corners

    def get_corners_from_kp(self, complete_track, corner_kps):
        return [self.find_closest_point(complete_track, corner) for corner in corner_kps]

    def find_closest_point(self, points, keypoint):
        min_dist = None
        closest_point = None
        for p in points:
            dist = math.hypot(p[0] - keypoint[0], p[1] - keypoint[1])
            if min_dist is None or dist < min_dist:
                min_dist = dist
                closest_point = p
        return closest_point

    def get_checkpoints(self, track_points, n_checkpoints=Constants.N_CHECKPOINTS):
        """
        Returns equally spaced checkpoints from the track
        """
        checkpoint_step = len(track_points) // n_checkpoints
        checkpoints = []
        for i in range(n_checkpoints):
            index = i * checkpoint_step
            checkpoints.append(track_points[index])
        return checkpoints

    # ------------------------
    # Internal helper methods
    # ------------------------
    def _make_rand_vector(self, dims):
        vec = [rn.gauss(0, 1) for _ in range(dims)]
        mag = sum(x ** 2 for x in vec) ** 0.5
        return [x / mag for x in vec]

    def push_points_apart(self, points, distance):
        distance_sq = distance * distance
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p_dist = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                if p_dist < distance:
                    dx = points[j][0] - points[i][0]
                    dy = points[j][1] - points[i][1]
                    dl = math.sqrt(dx * dx + dy * dy)
                    dx /= dl
                    dy /= dl
                    diff = distance - dl
                    dx *= diff
                    dy *= diff
                    points[j][0] = int(points[j][0] + dx)
                    points[j][1] = int(points[j][1] + dy)
                    points[i][0] = int(points[i][0] - dx)
                    points[i][1] = int(points[i][1] - dy)
        return points

    def fix_angles(self, points, max_angle):
        for i in range(len(points)):
            prev_point = i - 1 if i > 0 else len(points) - 1
            next_point = (i + 1) % len(points)
            px = points[i][0] - points[prev_point][0]
            py = points[i][1] - points[prev_point][1]
            pl = math.sqrt(px * px + py * py)
            px /= pl
            py /= pl
            nx = -(points[i][0] - points[next_point][0])
            ny = -(points[i][1] - points[next_point][1])
            nl = math.sqrt(nx * nx + ny * ny)
            nx /= nl
            ny /= nl
            a = math.atan2(px * ny - py * nx, px * nx + py * ny)
            if abs(math.degrees(a)) <= max_angle:
                continue
            diff = math.radians(max_angle * math.copysign(1, a)) - a
            c = math.cos(diff)
            s = math.sin(diff)
            new_x = (nx * c - ny * s) * nl
            new_y = (nx * s + ny * c) * nl
            points[next_point][0] = int(points[i][0] + new_x)
            points[next_point][1] = int(points[i][1] + new_y)
        return points


class TrackRenderer:
    """
    Handles all drawing functionality (points, lines, track, kerbs, etc.)
    """

    def draw_points(self, surface, color, points):
        for p in points:
            self.draw_single_point(surface, color, p)

    def draw_convex_hull(self, hull, surface, points, color):
        for i in range(len(hull.vertices) - 1):
            self.draw_single_line(surface, color, points[hull.vertices[i]], points[hull.vertices[i + 1]])
            if i == len(hull.vertices) - 2:
                self.draw_single_line(surface, color, points[hull.vertices[0]], points[hull.vertices[-1]])

    def draw_lines_from_points(self, surface, color, points):
        for i in range(len(points) - 1):
            self.draw_single_line(surface, color, points[i], points[i + 1])
            if i == len(points) - 2:
                self.draw_single_line(surface, color, points[0], points[-1])

    def draw_single_point(self, surface, color, pos, radius=2):
        pygame.draw.circle(surface, color, pos, radius)

    def draw_single_line(self, surface, color, init, end):
        pygame.draw.line(surface, color, init, end)

    def draw_track(self, surface, color, points, corners):
        """
        Draws the actual track (road, kerbs, starting grid).
        """
        radius = Constants.TRACK_WIDTH // 2
        self._draw_corner_kerbs(surface, corners, radius)

        chunk_dimensions = (radius * 2, radius * 2)
        for point in points:
            blit_pos = (point[0] - radius, point[1] - radius)
            track_chunk = pygame.Surface(chunk_dimensions, pygame.SRCALPHA)
            pygame.draw.circle(track_chunk, color, (radius, radius), radius)
            surface.blit(track_chunk, blit_pos)

        starting_grid = self._draw_starting_grid(radius * 2)
        # rotate and place starting grid
        offset = Constants.TRACK_POINT_ANGLE_OFFSET
        vec_p = [points[offset][1] - points[0][1], -(points[offset][0] - points[0][0])]
        n_vec_p = [vec_p[0] / math.hypot(vec_p[0], vec_p[1]), 
                   vec_p[1] / math.hypot(vec_p[0], vec_p[1])]
        angle = math.degrees(math.atan2(n_vec_p[1], n_vec_p[0]))
        rot_grid = pygame.transform.rotate(starting_grid, -angle)
        start_pos = (points[0][0] - math.copysign(1, n_vec_p[0]) * n_vec_p[0] * radius,
                     points[0][1] - math.copysign(1, n_vec_p[1]) * n_vec_p[1] * radius)
        surface.blit(rot_grid, start_pos)

    def draw_checkpoint(self, track_surface, points, checkpoint, debug=False):
        margin = Constants.CHECKPOINT_MARGIN
        radius = Constants.TRACK_WIDTH // 2 + margin
        offset = Constants.CHECKPOINT_POINT_ANGLE_OFFSET
        check_index = points.index(checkpoint)
        vec_p = [points[check_index + offset][1] - points[check_index][1],
                 -(points[check_index + offset][0] - points[check_index][0])]
        n_vec_p = [vec_p[0] / math.hypot(vec_p[0], vec_p[1]), 
                   vec_p[1] / math.hypot(vec_p[0], vec_p[1])]
        angle = math.degrees(math.atan2(n_vec_p[1], n_vec_p[0]))
        # draw checkpoint
        checkpoint_surf = self._draw_rectangle((radius * 2, 5), Constants.BLUE, line_thickness=1, fill=False)
        rot_checkpoint = pygame.transform.rotate(checkpoint_surf, -angle)

        if debug:
            rot_checkpoint.fill(Constants.RED)

        check_pos = (points[check_index][0] - math.copysign(1, n_vec_p[0]) * n_vec_p[0] * radius,
                     points[check_index][1] - math.copysign(1, n_vec_p[1]) * n_vec_p[1] * radius)
        track_surface.blit(rot_checkpoint, check_pos)

    # ------------------------
    # Internal helper methods
    # ------------------------
    def _draw_rectangle(self, dimensions, color, line_thickness=1, fill=False):
        filled = 0 if fill else line_thickness
        rect_surf = pygame.Surface(dimensions, pygame.SRCALPHA)
        pygame.draw.rect(rect_surf, color, (0, 0, dimensions[0], dimensions[1]), filled)
        return rect_surf

    def _draw_corner_kerbs(self, track_surface, corners, track_width):
        step = Constants.STEP_TO_NEXT_KERB_POINT
        offset = Constants.KERB_POINT_ANGLE_OFFSET
        correction_x = Constants.KERB_PLACEMENT_X_CORRECTION
        correction_y = Constants.KERB_PLACEMENT_Y_CORRECTION
        for corner in corners:
            temp_corner = corner + corner
            last_kerb = None
            for i in range(0, len(corner), step):
                # parallel vector
                vec_p = [temp_corner[i + offset][0] - temp_corner[i][0],
                         temp_corner[i + offset][1] - temp_corner[i][1]]
                n_vec_p = [vec_p[0] / math.hypot(vec_p[0], vec_p[1]),
                           vec_p[1] / math.hypot(vec_p[0], vec_p[1])]

                # perpendicular vector
                vec_perp = [temp_corner[i + offset][1] - temp_corner[i][1],
                            -(temp_corner[i + offset][0] - temp_corner[i][0])]
                n_vec_perp = [vec_perp[0] / math.hypot(vec_perp[0], vec_perp[1]),
                              vec_perp[1] / math.hypot(vec_perp[0], vec_perp[1])]

                angle = math.degrees(math.atan2(n_vec_p[1], n_vec_p[0]))
                kerb = self._draw_single_kerb()
                rot_kerb = pygame.transform.rotate(kerb, -angle)

                # Decide direction
                m_x, m_y = 1, 1
                if angle > 180:
                    m_x = -1

                start_pos = (
                    corner[i][0] + m_x * n_vec_perp[0] * track_width - correction_x,
                    corner[i][1] + m_y * n_vec_perp[1] * track_width - correction_y
                )
                if last_kerb is None:
                    last_kerb = start_pos
                else:
                    # space kerbs at least 'track_width' apart
                    if math.hypot(start_pos[0] - last_kerb[0], start_pos[1] - last_kerb[1]) >= track_width:
                        continue
                last_kerb = start_pos
                track_surface.blit(rot_kerb, start_pos)

    def _draw_single_kerb(self):
        tile_height = Constants.KERB_TILE_HEIGHT
        tile_width = Constants.KERB_TILE_WIDTH
        kerb_tile = pygame.image.load(Constants.KERB_TILE)
        kerb = pygame.Surface((tile_width, tile_height), pygame.SRCALPHA)
        kerb.blit(kerb_tile, (0, 0))
        return kerb

    def _draw_starting_grid(self, track_width):
        tile_height = Constants.START_TILE_HEIGHT
        tile_width = Constants.START_TILE_WIDTH
        grid_tile = pygame.image.load(Constants.STARTING_GRID_TILE)
        starting_grid = pygame.Surface((track_width, tile_height), pygame.SRCALPHA)
        for i in range(track_width // tile_height):
            position = (i * tile_width, 0)
            starting_grid.blit(grid_tile, position)
        return starting_grid


class RacetrackGame:
    """
    Main class responsible for initializing Pygame and orchestrating the
    generation and drawing of the procedural racetrack.
    """

    def __init__(self, debug=True, show_checkpoints=True):
        pygame.init()
        self.screen = pygame.display.set_mode((Constants.WIDTH, Constants.HEIGHT))
        self.debug = debug
        self.show_checkpoints = show_checkpoints
        self.track_generation = TrackGeneration()
        self.track_renderer = TrackRenderer()
        pygame.display.set_caption(Constants.TITLE)

    def run(self):
        # Fill background
        self.screen.fill(Constants.GRASS_GREEN)

        # 1. Generate track data
        points = self.track_generation.random_points()
        hull = ConvexHull(points)
        track_points = self.track_generation.get_track_points(hull, points)
        shaped_track = self.track_generation.shape_track(track_points)
        corner_points = self.track_generation.get_corners_with_kerb(shaped_track)
        smoothed_track = self.track_generation.smooth_track(shaped_track)
        full_corners = self.track_generation.get_full_corners(smoothed_track, corner_points)
        checkpoints = self.track_generation.get_checkpoints(smoothed_track)

        # 2. Draw the track
        self.track_renderer.draw_track(self.screen, Constants.GREY, smoothed_track, full_corners)

        # 3. Optionally draw checkpoints
        if self.show_checkpoints or self.debug:
            for ch in checkpoints:
                self.track_renderer.draw_checkpoint(self.screen, smoothed_track, ch, self.debug)

        # 4. Debug draws
        if self.debug:
            # Original random points
            self.track_renderer.draw_points(self.screen, Constants.WHITE, points)
            self.track_renderer.draw_convex_hull(hull, self.screen, points, Constants.RED)

            # Shaped track points and lines
            self.track_renderer.draw_points(self.screen, Constants.BLUE, shaped_track)
            self.track_renderer.draw_lines_from_points(self.screen, Constants.BLUE, shaped_track)

            # Final (smoothed) track points
            self.track_renderer.draw_points(self.screen, Constants.BLACK, smoothed_track)

        # Update display
        pygame.display.update()

        # Main loop
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()


def str2bool(v):
    """
    Helper method to parse strings into boolean values.
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # rn.seed(rn.choice(COOL_TRACK_SEEDS)) # Optional seeding
    parser = argparse.ArgumentParser(description="Procedural racetrack generator")
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False,
                        help="Show racetrack generation steps")
    parser.add_argument("--show-checkpoints", type=str2bool, nargs='?', const=True, default=False,
                        help="Show checkpoints")
    args = parser.parse_args()

    game = RacetrackGame(debug=args.debug, show_checkpoints=args.show_checkpoints)
    game.run()
