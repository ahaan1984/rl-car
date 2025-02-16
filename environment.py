import numpy as np
import pygame
from scipy.spatial import ConvexHull

from car import Car
from track import RacetrackGame, TrackGeneration


class Environment:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()

        self.track_game = RacetrackGame(debug=False, show_checkpoints=True)
        self.track_generation = TrackGeneration()
        self.track_game.screen = self.screen
        self.generate_track()
        self.hull = None

        self.car = None
        self.reset()

        self.track_points = None

        self.num_actions = 5
        self.steering_angles = {
            0: 0,      # No steering
            1: 0.5,    # Medium right + acceleration
            2: -0.5,   # Medium left + acceleration
            3: 1.0,    # Full acceleration
            4: -1.0,   # Full brake
        }

        self.acceleration_factors = {
            0: 0,      # No acceleration
            1: 0.3,    # Medium acceleration for right turn
            2: 0.3,    # Medium acceleration for left turn
            3: 1.0,    # Full acceleration
            4: -1.0,   # Full brake
        }

        self.prev_distance = 0
        self.steps_since_progress = 0
        self.max_steps_without_progress = 100
        self.track_margin = 5

    def calculate_initial_curvature(self) -> float:
        if len(self.smoothed_track) < 3:
            return 0.0

        p1 = np.array(self.smoothed_track[0])
        p2 = np.array(self.smoothed_track[1])
        p3 = np.array(self.smoothed_track[2])

        v1 = p2 - p1
        v2 = p3 - p2

        angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
        return angle

    def generate_track(self) -> None:
        self.screen.fill((100, 200, 100))
        track_points = self.track_game.track_generation.random_points()
        if len(track_points) == 0:
            raise ValueError("No points")
        self.hull = ConvexHull(track_points)
        self.track_points = self.track_game.track_generation.get_track_points(self.hull, track_points)
        self.shaped_track = self.track_game.track_generation.shape_track(self.track_points)
        self.corner_points = self.track_game.track_generation.get_corners_with_kerb(self.shaped_track)
        self.smoothed_track = self.track_game.track_generation.smooth_track(self.shaped_track)
        self.full_corners = self.track_game.track_generation.get_full_corners(self.smoothed_track, self.corner_points)

    def reset(self) -> np.ndarray:
        if self.car is None:
            self.car = Car(self.smoothed_track[0][0], self.smoothed_track[0][1])
        else:
            self.car.x = self.smoothed_track[0][0]
            self.car.y = self.smoothed_track[0][1]

        initial_angle = self.calculate_initial_angle()
        initial_curvature = self.calculate_initial_curvature()

        self.car.angle = initial_angle
        self.car.tilt_angle = 0
        self.car.steering = np.clip(initial_curvature / np.pi, -1, 1)
        self.car.speed = 0
        self.car.acceleration = 0
        self.car.angular_velocity = 0

        self.prev_distance = 0
        self.steps_since_progress = 0

        return self.get_state()

    def get_state(self) -> np.ndarray:
        sensor_readings = self.car.get_sensors(self.track_game)
        
        distance_sensors = sensor_readings[:8]  # First 8 directional sensors
        path_distance = sensor_readings[-2]     # Distance to nearest track point
        speed_ratio = self.car.speed / self.car.max_speed
        
        # Create 3x3 spatial grid
        sensor_grid = np.array([
            [distance_sensors[0], distance_sensors[1], distance_sensors[2]],
            [distance_sensors[3], speed_ratio,         distance_sensors[4]],
            [distance_sensors[5], distance_sensors[6], distance_sensors[7]]
        ])
        
        # Get orientation components
        angle_rad = np.radians(self.car.angle)
        trig_values = np.array([np.sin(angle_rad), np.cos(angle_rad)])
        
        return np.concatenate([
            sensor_grid.flatten(),  # 9 elements (3x3 grid)
            [path_distance],        # 1 element
            trig_values             # 2 elements
        ]).astype(np.float32)


    def is_on_track(self, x, y):
        track_points = np.array(self.smoothed_track)

        distances = np.sqrt(np.sum((track_points - np.array([x, y]))**2, axis=1))

        return np.any(distances < (self.track_generation.track_width - self.track_margin))

    def step(self, action: int):
        prev_x, prev_y = self.car.x, self.car.y
        self.car.steering = self.steering_angles[action]
        self.car.acceleration = self.acceleration_factors[action]

        went_off_track = self.car.move(self)

        reward = 0
        done = False

        if went_off_track:
            self.car.x, self.car.y = prev_x, prev_y
            reward = -100
            done = True
        else:
            current_distance = self.calculate_progress()
            distance_reward = current_distance - self.prev_distance

            if distance_reward > 0:
                reward += distance_reward * 20
                self.steps_since_progress = 0

                if action in [1, 2]:
                    turn_efficiency = 1.0
                    speed_ratio = self.car.speed / self.car.max_speed

                    if 0.3 <= speed_ratio <= 0.7:
                        turn_efficiency *= 1.5

                    if abs(self.car.angular_velocity) > 0.1:
                        turn_efficiency *= 1.2 if 0.3 <= speed_ratio <= 0.7 else 0.8

                    reward *= turn_efficiency
            else:
                self.steps_since_progress += 1
                reward -= 0.2

            if action in [0, 3] and abs(self.car.angular_velocity) < 0.1:
                reward += self.car.speed * 0.05

            if self.steps_since_progress >= self.max_steps_without_progress:
                reward -= 20
                done = True

            self.prev_distance = current_distance

        new_state = self.get_state()
        self.render()

        return new_state, reward, done


    def calculate_progress(self) -> float:
        min_distance = float("inf")
        for point in self.smoothed_track:
            distance = np.sqrt((self.car.x - point[0])**2 + (self.car.y - point[1])**2)
            min_distance = min(min_distance, distance)
        return -min_distance

    def calculate_initial_angle(self) -> float:
        start_point = np.array(self.smoothed_track[0])
        next_point = np.array(self.smoothed_track[1])

        direction = next_point - start_point

        angle = np.arctan2(direction[1], direction[0])
        return np.degrees(angle)

    def render(self) -> None:
        self.screen.fill((100, 200, 100))
        self.track_game.track_renderer.draw_track(self.screen, (128, 128, 128),
                                                self.smoothed_track, self.full_corners)
        self.car.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)