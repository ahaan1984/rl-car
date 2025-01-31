import math
import random as rn
import sys

import numpy as np
import pygame
from scipy import interpolate
from scipy.spatial import ConvexHull

from utils import Constants


class TrackGeneration(Constants):
    def __init__(self):
        super().__init__()
        self.min_points = Constants.MIN_POINTS
        self.max_points = Constants.MAX_POINTS
        self.track_width = Constants.TRACK_WIDTH
        self.margin = Constants.MARGIN
        self.spline_points = Constants.SPLINE_POINTS
        self.full_corner_num_points = Constants.FULL_CORNER_NUM_POINTS
        self.min_distance = Constants.MIN_DISTANCE
        self.difficulty = Constants.DIFFICULTY
        self.max_displacement = Constants.MAX_DISPLACEMENT
        self.distance_between_points = Constants.DISTANCE_BETWEEN_POINTS
        self.max_angle = Constants.MAX_ANGLE
        self.min_kerb_angle = Constants.MIN_KERB_ANGLE
        self.max_kerb_angle = Constants.MAX_KERB_ANGLE
        self.width = Constants.WIDTH
        self.height = Constants.HEIGHT

    def random_points(self) -> np.ndarray:
        point_count = rn.randint(self.min_points, self.max_points + 1)
        points = []
        for _ in range(point_count):
            x = rn.randrange(self.margin, self.width - self.margin + 1, 1)
            y = rn.randrange(self.margin, self.height - self.margin + 1, 1)

            distances = list(filter(lambda d: d < self.min_distance,
                [math.sqrt((p[0] - x) ** 2 + (p[1] - y) ** 2) for p in points]))

            if not distances:
                points.append((x, y))
        return np.array(points)

    def get_track_points(self, hull, points) -> np.ndarray:
        return np.array([points[hull.vertices[i]] for i in range(len(hull.vertices))])

    def shape_track(self, track_points):
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

        for _ in range(3):
            track_set = self.fix_angles(track_set, self.max_angle)
            track_set = self.push_points_apart(track_set, self.distance_between_points)

        final_set = []
        for point in track_set:
            if point[0] < self.margin:
                point[0] = self.margin
            elif point[0] > (self.width - self.margin):
                point[0] = self.width - self.margin
            if point[1] < self.margin:
                point[1] = self.margin
            elif point[1] > (self.height - self.margin):
                point[1] = self.height - self.margin
            final_set.append(point)

        return self._clamp_points(track_set)

    def get_corners_with_kerb(self, points):
        points_array = np.array(points)
        vectors = np.roll(points_array, -1, axis=0) - points_array
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angle_diffs = np.abs(np.degrees(angles - np.roll(angles, 1)))
        mask = (self.min_kerb_angle <= angle_diffs) & (angle_diffs <= self.max_kerb_angle)
        return points_array[mask]

    def smooth_track(self, track_points:np.ndarray) -> list:
        x = np.array([p[0] for p in track_points])
        y = np.array([p[1] for p in track_points])
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

        tck, _ = interpolate.splprep([x, y], s=0, per=True)
        xi, yi = interpolate.splev(np.linspace(0, 1, self.spline_points), tck)
        return [(int(xi[i]), int(yi[i])) for i in range(len(xi))]

    def get_full_corners(self, track_points, corners):
        corners_in_track = self.get_corners_from_kp(track_points, corners)
        f_corners = []
        offset = self.full_corner_num_points
        for corner in corners_in_track:
            i = track_points.index(corner)
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

    # ------------------------
    # Internal helper methods
    # ------------------------
    def _make_rand_vector(self, dims):
        vec = [rn.gauss(0, 1) for _ in range(dims)]
        mag = sum(x ** 2 for x in vec) ** 0.5
        assert mag != 0
        return [x / mag for x in vec]


    def push_points_apart(self, points, distance):
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

    def _clamp_points(self, points):
        final_set = []
        for point in points:
            clamped_x = max(self.margin, min(point[0], self.width - self.margin))
            clamped_y = max(self.margin, min(point[1], self.height - self.margin))
            final_set.append([clamped_x, clamped_y])
        return np.array(final_set)


class Render(Constants):
    def __init__(self) -> None:
        super().__init__()
        self.track_width = Constants.TRACK_WIDTH
        self.checkpoint_margin = Constants.CHECKPOINT_MARGIN
        self.checkpoint_point_angle_offset = Constants.CHECKPOINT_POINT_ANGLE_OFFSET
        self.step = Constants.STEP_TO_NEXT_KERB_POINT
        self.kerb_offset = Constants.KERB_POINT_ANGLE_OFFSET
        self.kerb_x_correction = Constants.KERB_PLACEMENT_X_CORRECTION
        self.kerb_y_correction = Constants.KERB_PLACEMENT_Y_CORRECTION
        self.tile_height = Constants.KERB_TILE_HEIGHT
        self.tile_width = Constants.KERB_TILE_WIDTH
        self.tile = Constants.KERB_TILE

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
        radius = self.track_width // 2
        self._draw_corner_kerbs(surface, corners, radius)

        chunk_dimensions = (radius * 2, radius * 2)
        for point in points:
            blit_pos = (point[0] - radius, point[1] - radius)
            track_chunk = pygame.Surface(chunk_dimensions, pygame.SRCALPHA)
            pygame.draw.circle(track_chunk, color, (radius, radius), radius)
            surface.blit(track_chunk, blit_pos)

        starting_grid = self._draw_starting_grid(radius * 2)
        offset = self.checkpoint_point_angle_offset
        vec_p = [points[offset][1] - points[0][1], -(points[offset][0] - points[0][0])]
        n_vec_p = [vec_p[0] / math.hypot(vec_p[0], vec_p[1]), 
                   vec_p[1] / math.hypot(vec_p[0], vec_p[1])]
        angle = math.degrees(math.atan2(n_vec_p[1], n_vec_p[0]))
        rot_grid = pygame.transform.rotate(starting_grid, -angle)
        start_pos = (points[0][0] - math.copysign(1, n_vec_p[0]) * n_vec_p[0] * radius,
                     points[0][1] - math.copysign(1, n_vec_p[1]) * n_vec_p[1] * radius)
        surface.blit(rot_grid, start_pos)

    def draw_checkpoint(self, track_surface, points, checkpoint, debug=False):
        margin = self.checkpoint_margin
        radius = self.track_width // 2 + margin
        offset = self.checkpoint_point_angle_offset
        check_index = points.index(checkpoint)
        vec_p = [points[check_index + offset][1] - points[check_index][1],
                 -(points[check_index + offset][0] - points[check_index][0])]
        n_vec_p = [vec_p[0] / math.hypot(vec_p[0], vec_p[1]), 
                   vec_p[1] / math.hypot(vec_p[0], vec_p[1])]
        angle = math.degrees(math.atan2(n_vec_p[1], n_vec_p[0]))
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
        step = self.step
        offset = self.kerb_offset
        correction_x = self.kerb_x_correction
        correction_y = self.kerb_y_correction
        for corner in corners:
            temp_corner = corner + corner
            last_kerb = None
            for i in range(0, len(corner), step):
                vec_p = [temp_corner[i + offset][0] - temp_corner[i][0],
                         temp_corner[i + offset][1] - temp_corner[i][1]]
                n_vec_p = [vec_p[0] / math.hypot(vec_p[0], vec_p[1]),
                           vec_p[1] / math.hypot(vec_p[0], vec_p[1])]

                vec_perp = [temp_corner[i + offset][1] - temp_corner[i][1],
                            -(temp_corner[i + offset][0] - temp_corner[i][0])]
                n_vec_perp = [vec_perp[0] / math.hypot(vec_perp[0], vec_perp[1]),
                              vec_perp[1] / math.hypot(vec_perp[0], vec_perp[1])]

                angle = math.degrees(math.atan2(n_vec_p[1], n_vec_p[0]))
                kerb = self._draw_single_kerb()
                rot_kerb = pygame.transform.rotate(kerb, -angle)

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
                    if math.hypot(start_pos[0] - last_kerb[0], start_pos[1] - last_kerb[1]) >= track_width:
                        continue
                last_kerb = start_pos
                track_surface.blit(rot_kerb, start_pos)

    def _draw_single_kerb(self):
        tile_height = self.tile_height
        tile_width = self.tile_width
        kerb_tile = pygame.image.load(self.tile)
        kerb = pygame.Surface((tile_width, tile_height), pygame.SRCALPHA)
        kerb.blit(kerb_tile, (0, 0))
        return kerb

    def _draw_starting_grid(self, track_width):
        tile_height = self.tile_height
        tile_width = self.tile_width
        grid_tile = pygame.image.load(self.tile)
        starting_grid = pygame.Surface((track_width, tile_height), pygame.SRCALPHA)
        for i in range(track_width // tile_height):
            position = (i * tile_width, 0)
            starting_grid.blit(grid_tile, position)
        return starting_grid


class RacetrackGame(Constants):
    """
    Main class responsible for initializing Pygame and orchestrating the
    generation and drawing of the procedural racetrack.
    """ 
    def __init__(self, debug=True, show_checkpoints=True):
        super().__init__()
        pygame.init()
        self.screen = pygame.display.set_mode((Constants.WIDTH, Constants.HEIGHT), 
        pygame.DOUBLEBUF | pygame.HWSURFACE)
        self.debug = True
        self.smoothed_track = []
        self.show_checkpoints = show_checkpoints
        self.track_generation = TrackGeneration()
        self.track_renderer = Render()
        pygame.display.set_caption(Constants.TITLE)

    def run(self):
        self.screen.fill(Constants.GRASS_GREEN)

        points = self.track_generation.random_points()
        hull = ConvexHull(points)
        track_points = self.track_generation.get_track_points(hull, points)
        shaped_track = self.track_generation.shape_track(track_points)
        corner_points = self.track_generation.get_corners_with_kerb(shaped_track)
        self.smoothed_track = self.track_generation.smooth_track(shaped_track)
        full_corners = self.track_generation.get_full_corners(self.smoothed_track, corner_points)
        checkpoints = self.track_generation.get_checkpoints(self.smoothed_track)

        self.track_renderer.draw_track(self.screen, Constants.GREY, self.smoothed_track, full_corners)

        if self.show_checkpoints or self.debug:
            for ch in checkpoints:
                self.track_renderer.draw_checkpoint(self.screen, self.smoothed_track, ch, self.debug)

        if self.debug:
            self.track_renderer.draw_points(self.screen, Constants.WHITE, points)
            self.track_renderer.draw_convex_hull(hull, self.screen, points, Constants.RED)

            self.track_renderer.draw_points(self.screen, Constants.BLUE, shaped_track)
            self.track_renderer.draw_lines_from_points(self.screen, Constants.BLUE, shaped_track)

            self.track_renderer.draw_points(self.screen, Constants.BLACK, self.smoothed_track)

        pygame.display.update()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()

    def raycast_collision(self, start: tuple, end: tuple) -> tuple | None:
        sx, sy = start
        ex, ey = end

        min_dist = float("inf")
        closest_point = None

        for point in self.smoothed_track:
            px, py = point
            dx = ex - sx
            dy = ey - sy
            dp = (px - sx, py - sy)

            cross = dx * (py - sy) - dy * (px - sx)
            if abs(cross) > 1e-8:
                continue

            t = (dp[0] * dx + dp[1] * dy) / (dx**2 + dy**2)
            if t < 0 or t > 1:
                continue

            proj_x = sx + t * dx
            proj_y = sy + t * dy

            dist = np.sqrt((proj_x - px)**2 + (proj_y - py)**2)
            if dist < min_dist:
                min_dist = dist
                closest_point = (proj_x, proj_y)
        if closest_point is not None and min_dist < self.track_generation.track_width:
            return closest_point
        return None

    def is_on_track(self, x, y):
        if not (0 <= x <= Constants.WIDTH and 0 <= y <= Constants.HEIGHT):
            return False

        track_width = self.track_generation.track_width

        for point in self.smoothed_track:
            dist_squared = (x - point[0])**2 + (y - point[1])**2
            if dist_squared < (track_width - self.track_generation.margin)**2:
                return True

        return False