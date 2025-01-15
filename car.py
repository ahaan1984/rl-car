import pygame
import numpy as np

class Car:
    def __init__(self, start_x=400, start_y=300):
        self.CAR_LENGTH = 40
        self.CAR_WIDTH = 20
        self.SENSOR_RANGE = 200
        self.NUM_SENSORS = 8
        self.x = start_x
        self.y = start_y
        self.angle = 0
        self.speed = 0
        self.acceleration = 0
        self.steering = 0
        self.sensor_readings = np.zeros(self.NUM_SENSORS + 2)
        self.min_speed = -2
        self.max_speed = 20
        self.steering_factor = 4
        self.friction = 0.1
        self.acceleration_factor = 4
        
        self.surface = pygame.Surface((self.CAR_LENGTH, self.CAR_WIDTH), pygame.SRCALPHA)
        pygame.draw.rect(self.surface, (255, 0, 0), (0, 0, self.CAR_LENGTH, self.CAR_WIDTH))
    
    def move(self):
        self.speed += self.acceleration * self.acceleration_factor
        
        if abs(self.speed) > 0:
            self.speed *= (1 - self.friction)
            
        self.speed = np.clip(self.speed, self.min_speed, self.max_speed)
        
        min_speed_for_steering = 0.1
        speed_factor = abs(self.speed) / self.max_speed if abs(self.speed) > min_speed_for_steering else 0
        effective_steering = self.steering * self.steering_factor * speed_factor
        
        self.angle += effective_steering

        self.x += np.cos(np.radians(self.angle)) * self.speed
        self.y += np.sin(np.radians(self.angle)) * self.speed

    def draw(self, screen):
        rotated_surface = pygame.transform.rotate(self.surface, -self.angle)
        
        rect = rotated_surface.get_rect(center=(self.x, self.y))
        
        screen.blit(rotated_surface, rect)
        
        if hasattr(self, 'sensor_readings'):
            for i, reading in enumerate(self.sensor_readings[:self.NUM_SENSORS]):
                sensor_angle = self.angle + np.linspace(-120, 120, self.NUM_SENSORS)[i]
                end_x = self.x + np.cos(np.radians(sensor_angle)) * self.SENSOR_RANGE * reading
                end_y = self.y + np.sin(np.radians(sensor_angle)) * self.SENSOR_RANGE * reading
                pygame.draw.line(screen, (0, 255, 0), (self.x, self.y), (end_x, end_y), 1)

    def get_sensors(self, track):
        sensor_angles = np.linspace(-120, 120, self.NUM_SENSORS)
        for i, angle in enumerate(sensor_angles):
            sensor_angle = self.angle + angle
            sensor_x = self.x
            sensor_y = self.y
            
            for distance in range(self.SENSOR_RANGE):
                sensor_x += np.cos(np.radians(sensor_angle))
                sensor_y += np.sin(np.radians(sensor_angle))
                
                if not track.is_on_track(sensor_x, sensor_y):
                    self.sensor_readings[i] = distance / self.SENSOR_RANGE
                    break
            else:
                self.sensor_readings[i] = 1.0

        if hasattr(track, 'get_nearest_path_point'):
            nearest_point, distance = track.get_nearest_path_point(self.x, self.y)
            self.sensor_readings[-2] = np.clip(distance / (track.track_width/2), 0, 1)
            point_angle = np.degrees(np.arctan2(nearest_point[1] - self.y, nearest_point[0] - self.x))
            angle_diff = (point_angle - self.angle + 180) % 360 - 180
            self.sensor_readings[-1] = angle_diff / 180
            
        return self.sensor_readings