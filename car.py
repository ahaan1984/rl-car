import numpy as np
import pygame


class Car:
    def __init__(self, start_x:int=400, start_y:int=300) -> None:
        self.CAR_LENGTH = 40
        self.CAR_WIDTH = 20
        self.SENSOR_RANGE = 200
        self.NUM_SENSORS = 8
        self.x = start_x
        self.y = start_y
        self.angle = 0
        self.initial_angle = 0
        self.angle_offset = 0

        self.speed = 0
        self.acceleration = 0
        self.steering = 0

        self.tilt_angle = 0
        self.initial_tilt = 0
        self.tilt_interpolation = 0.1
        self.max_tilt_angle = 15
        self.tilt_damping = 0.1

        self.angular_velocity = 0
        self.angular_dampening = 0.1
        self.min_speed_for_steering = 0.05
        self.steering_response = 0.4
        self.max_angular_velocity = 5.0
        self.steering_speed_factor = 0.3

        self.turn_precision = 0.05
        self.steering_deadzone = 0.02

        self.sensor_readings = np.zeros(self.NUM_SENSORS + 2)
        self.min_speed = -2
        self.max_speed = 20
        self.steering_factor = 4
        self.friction = 0.1
        self.acceleration_factor = 4
        self.show_sensors = False

        self.sprite = pygame.image.load('static/simple-car.png').convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, 
                                    (self.CAR_LENGTH, self.CAR_WIDTH))


    def move(self, environment):
        self.speed += self.acceleration * self.acceleration_factor

        base_friction = self.friction
        if abs(self.steering) > 0:

            turn_friction = base_friction * (1 + abs(self.steering) * 0.5)
            self.speed *= (1 - turn_friction)
        else:
            self.speed *= (1 - base_friction)

        self.speed = np.clip(self.speed, self.min_speed, self.max_speed)

        if abs(self.steering) > self.steering_deadzone:
            speed_steering_factor = max(0.2, abs(self.speed) / self.max_speed)
            if abs(self.speed) < self.min_speed_for_steering:
                speed_steering_factor = 0.15

            steering_power = abs(self.steering)
            if steering_power < 0.3:
                steering_multiplier = 0.8
            elif steering_power < 0.7:
                steering_multiplier = 1.0
            else:
                steering_multiplier = 1.2

            target_angular_velocity = (self.steering *
                                     self.max_angular_velocity *
                                     speed_steering_factor *
                                     steering_multiplier)

            angular_diff = target_angular_velocity - self.angular_velocity
            self.angular_velocity += angular_diff * self.steering_response
            self.angular_velocity = np.clip(self.angular_velocity,
                                          -self.max_angular_velocity,
                                          self.max_angular_velocity)
        else:
            self.angular_velocity *= (1 - self.angular_dampening)

        new_angle = self.angle + self.angular_velocity

        self.tilt_angle = -self.steering * min(abs(self.speed), self.max_speed) / self.max_speed * self.max_tilt_angle

        self.tilt_angle *= (1 - self.tilt_damping)
        target_tilt = -self.steering * min(abs(self.speed), self.max_speed) / self.max_speed * self.max_tilt_angle
        self.tilt_angle += (target_tilt - self.tilt_angle) * self.tilt_interpolation

        new_x = self.x + np.cos(np.radians(new_angle)) * self.speed
        new_y = self.y + np.sin(np.radians(new_angle)) * self.speed

        if abs(self.steering) > 0.8:
            self.speed = max(self.speed * 0.8, 0)

        if environment.is_on_track(new_x, new_y):
            self.angle = new_angle
            self.x = new_x
            self.y = new_y
            return False
        else:
            self.speed = max(self.speed * 0.5, 0)
            self.angular_velocity = 0
            self.acceleration = 0
            return True

    def draw(self, screen) -> None:
    # Rotate sprite based on car angle and tilt
        rotated = pygame.transform.rotate(self.sprite, -self.angle)
        tilted = pygame.transform.rotate(rotated, self.tilt_angle)
        rect = tilted.get_rect(center=(self.x, self.y))
        screen.blit(tilted, rect)

        if self.show_sensors:
            for i, reading in enumerate(self.sensor_readings[:self.NUM_SENSORS]):
                sensor_angle = self.angle + np.linspace(-120, 120, self.NUM_SENSORS)[i]
                max_x = self.x + np.cos(np.radians(sensor_angle)) * self.SENSOR_RANGE
                max_y = self.y + np.sin(np.radians(sensor_angle)) * self.SENSOR_RANGE
                pygame.draw.line(screen, (0, 255, 0, 30), (self.x, self.y), (max_x, max_y), 1)

                end_x = self.x + np.cos(np.radians(sensor_angle)) * self.SENSOR_RANGE * reading
                end_y = self.y + np.sin(np.radians(sensor_angle)) * self.SENSOR_RANGE * reading
                pygame.draw.line(screen, (0, 255, 0, 255), (self.x, self.y), (end_x, end_y), 2)
                pygame.draw.circle(screen, (255, 255, 0), (int(end_x), int(end_y)), 3)


    def get_sensors(self, track) -> np.ndarray:
            sensor_angles = np.linspace(-120, 120, self.NUM_SENSORS)
            for i, angle in enumerate(sensor_angles):
                sensor_angle = self.angle + angle
                sensor_x = self.x
                sensor_y = self.y

                end_x = sensor_x + np.cos(np.radians(sensor_angle)) * self.SENSOR_RANGE
                end_y = sensor_y + np.sin(np.radians(sensor_angle)) * self.SENSOR_RANGE

                collision = track.raycast_collision((self.x, self.y), (end_x, end_y))
                if collision:
                    distance = np.sqrt((collision[0] - self.x)**2 + (collision[1] - self.y)**2)
                    self.sensor_readings[i] = distance / self.SENSOR_RANGE
                else:
                    self.sensor_readings[i] = 1.0

            if hasattr(track, "get_nearest_path_point"):
                nearest_point, distance = track.get_nearest_path_point(self.x, self.y)
                self.sensor_readings[-2] = np.clip(distance / (track.track_width / 2), 0, 1)
                point_angle = np.degrees(np.arctan2(nearest_point[1] - self.y, nearest_point[0] - self.x))
                angle_diff = (point_angle - self.angle + 180) % 360 - 180
                self.sensor_readings[-1] = angle_diff / 180

            return self.sensor_readings
