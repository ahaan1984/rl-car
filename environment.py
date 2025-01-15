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

        self.num_actions = 9
        self.steering_angles = {
            0: 0,      # No steering
            1: 1,      # Full acceleration
            2: -1,     # Full brake
            3: 0.25,   # Slight right + some acceleration
            4: 0.5,    # Medium right + some acceleration
            5: 1.0,    # Sharp right + minimal acceleration
            6: -0.25,  # Slight left + some acceleration
            7: -0.5,   # Medium left + some acceleration
            8: -1.0    # Sharp left + minimal acceleration
        }

        self.acceleration_factors = {
            0: 0,    # No acceleration
            1: 1.0,  # Full acceleration
            2: -1.0, # Full brake
            3: 0.4,  # More acceleration for gentle turns
            4: 0.3,  # Moderate acceleration for medium turns
            5: 0.2,  # Minimal acceleration for sharp turns
            6: 0.4,  # Mirror of right turns
            7: 0.3,
            8: 0.2
        }

        self.prev_distance = 0
        self.steps_since_progress = 0
        self.max_steps_without_progress = 100
        self.track_margin = 5

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
            self.car.angle = 0
            self.car.speed = 0
            self.car.acceleration = 0
            self.car.steering = 0

        self.prev_distance = 0
        self.steps_since_progress = 0
        return self.get_state()

    def get_state(self) -> np.ndarray:
        sensor_readings = self.car.get_sensors(self.track_game)

        state = np.concatenate([
            sensor_readings,
            [self.car.speed / self.car.max_speed],
            [np.sin(np.radians(self.car.angle))],
            [np.cos(np.radians(self.car.angle))]
        ], dtype=np.float32)
        return state

    def is_on_track(self, x, y):
        for point in self.smoothed_track:
            if np.sqrt((x - point[0])**2 + (y - point[1])**2) < (self.track_generation.track_width - self.track_margin):
                return True
        return False

    def step(self, action: int):
        prev_x, prev_y = self.car.x, self.car.y
        # Apply steering and acceleration based on action
        self.car.steering = self.steering_angles[action]
        self.car.acceleration = self.acceleration_factors[action]

        went_off_track = self.car.move(self)

        # Modified reward structure for granular steering
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
                # Base progress reward
                reward += distance_reward * 20
                self.steps_since_progress = 0

                # Additional reward for smooth turning
                if 3 <= action <= 8:  # Turning actions
                    turn_efficiency = 1.0
                    # Reward more for appropriate speed during turns
                    speed_ratio = self.car.speed / self.car.max_speed
                    if 0.2 <= speed_ratio <= 0.6:  # Sweet spot for turning
                        turn_efficiency *= 1.5
                    # Reward more for using appropriate turn angle
                    if abs(self.car.angular_velocity) > 0.1:  # Actually turning
                        if action in [3, 6]:  # Gentle turns
                            turn_efficiency *= 1.2 if speed_ratio > 0.5 else 0.8
                        elif action in [4, 7]:  # Medium turns
                            turn_efficiency *= 1.2 if 0.3 <= speed_ratio <= 0.5 else 0.8
                        elif action in [5, 8]:  # Sharp turns
                            turn_efficiency *= 1.2 if speed_ratio < 0.3 else 0.8
                    reward *= turn_efficiency
            else:
                self.steps_since_progress += 1
                reward -= 0.5

                # Less severe penalty for slight course corrections
                if action in [3, 6]:  # Gentle turns
                    reward += 0.1  # Encourage exploration of minor adjustments

            # Add small incentive for maintaining speed during straight sections
            if action in [0, 1] and abs(self.car.angular_velocity) < 0.1:
                reward += self.car.speed * 0.05

            if self.steps_since_progress >= self.max_steps_without_progress:
                reward -= 50
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

    def render(self) -> None:
        self.screen.fill((100, 200, 100))
        self.track_game.track_renderer.draw_track(self.screen, (128, 128, 128), 
                                                self.smoothed_track, self.full_corners)
        self.car.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)