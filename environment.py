import pygame
import numpy as np
from track import RacetrackGame, TrackGeneration, Render
from scipy.spatial import ConvexHull
from car import Car

class Environment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        
        self.track_game = RacetrackGame(debug=False, show_checkpoints=False)
        self.track_generation = TrackGeneration()
        self.track_game.screen = self.screen
        self.generate_track()
        
        self.car = None
        self.reset()

        self.track_points = None
        
        self.num_actions = 5
        
        self.prev_distance = 0
        self.steps_since_progress = 0
        self.max_steps_without_progress = 100
        
    def generate_track(self):
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
        
    def reset(self):
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
    
    def get_state(self):
        sensor_readings = self.car.get_sensors(self.track_game)
        
        state = np.concatenate([
            sensor_readings,
            [self.car.speed / self.car.max_speed],
            [np.sin(np.radians(self.car.angle))],
            [np.cos(np.radians(self.car.angle))]
        ])
        return state
    
    def is_on_track(self, x, y):
        for point in self.smoothed_track:
            if np.sqrt((x - point[0])**2 + (y - point[1])**2) < self.track_generation.track_width:
                return True
        return False
    
    def step(self, action):
        if action == 0:  # No action
            self.car.acceleration = 0
            self.car.steering = 0
        elif action == 1:  # Accelerate
            self.car.acceleration = 1
            self.car.steering = 0
        elif action == 2:  # Brake
            self.car.acceleration = -1
            self.car.steering = 0
        elif action == 3:  # Steer left
            self.car.steering = 1
        elif action == 4:  # Steer right
            self.car.steering = -1
            
        self.car.move()
        
        reward = 0
        done = False
        
        if not self.is_on_track(self.car.x, self.car.y):
            reward = -100
            done = True
        else:
            current_distance = self.calculate_progress()
            distance_reward = current_distance - self.prev_distance
            reward += distance_reward * 10
            
            reward -= 0.1
            
            if distance_reward <= 0:
                self.steps_since_progress += 1
            else:
                self.steps_since_progress = 0
                
            if self.steps_since_progress >= self.max_steps_without_progress:
                done = True
            
            self.prev_distance = current_distance
        
        new_state = self.get_state()
        
        self.render()
        
        return new_state, reward, done
    
    def calculate_progress(self):
        min_distance = float('inf')
        for point in self.smoothed_track:
            distance = np.sqrt((self.car.x - point[0])**2 + (self.car.y - point[1])**2)
            min_distance = min(min_distance, distance)
        return -min_distance
    
    def render(self):
        self.screen.fill((100, 200, 100))
        self.track_game.track_renderer.draw_track(self.screen, (128, 128, 128), 
                                                self.smoothed_track, self.full_corners)
        self.car.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)