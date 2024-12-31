import pygame
import numpy as np
from scipy.interpolate import splprep, splev

class Car:
    def __init__(self, start_x=None, start_y=None):
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
        self.max_speed = 8
        self.steering_factor = 8
        self.friction = 0.1
        self.acceleration_factor = 0.5
    
    def move(self):
        self.speed += self.acceleration * self.acceleration_factor
        if abs(self.speed) > 0:
            self.speed *= (1 - self.friction)
        self.speed = np.clip(self.speed, self.min_speed, self.max_speed)
        effective_steering = self.steering * self.steering_factor * (abs(self.speed) / self.max_speed)
        self.angle += effective_steering
        self.x = self.speed * np.cos(np.radians(self.angle))
        self.y = self.speed * np.cos(np.radians(self.angle))

    def get_sensors(self, track):
        sensor_angles = np.linspace(-120, 120, self.NUM_SENSORS)
        for i, angle in enumerate(sensor_angles):
            sensor_angle = self.angle + angle
            sensor_x = self.x
            sensor_y = self.y
            for distance in range(self.SENSOR_RANGE):
                sensor_x += np.cos(np.radians(sensor_angle))
                sensor_y += np.sin(np.radians(sensor_angle))
            
                # implement is_on_track() method in class Track
                if not track.is_on_track(sensor_x, sensor_y):
                    self.sensor_readings[i] = distance / self.SENSOR_RANGE
                    break

            else:
                self.sensor_reading[i] = 1.0

        # implement get_nearest_path_point in class Track
        nearest_point, distance = track.get_nearest_path_point(self.x, self.y)
        self.sensor_readings[-2] = np.clip(distance / (track.track_width/2), 0, 1)
        point_angle = np.degrees(np.arctan2(nearest_point[1] - self.y, nearest_point[0] - self.x))
        angle_diff = (point_angle - self.angle + 180) % 360 - 180
        self.sensor_readings[-1] = angle_diff / 180
        return self.sensor_readings


