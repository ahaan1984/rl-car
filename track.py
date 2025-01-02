"""
inspired by: https://bitesofcode.wordpress.com/2020/04/09/procedural-racetrack-generation/

https://github.com/juangallostra/procedural-tracks/blob/master/main.py

classes to be defined: 
PointGenerator: Handles generation of random points.
Track: Manages the track creation, shaping, and smoothing.
Drawer: Manages all drawing operations.
Game: Initializes and runs the game. 
"""

import pygame
import numpy as np
import math
import random
from scipy import interpolate
from scipy.spatial import ConvexHull

class Track:
    def __init__(self, width, height, min_points, max_points, margin, min_distance, distance_between_points):
        self.width = width
        self.height = height
        self.min_points = min_points
        self.max_points = max_points
        self.margin = margin
        self.min_distance = min_distance
        self.distance_between_points = distance_between_points
        self.max_angle = 90
        self.max_displacement = 80
        self.difficulty = 0.1

    def random_points(self):
        pointCount = random.randrange(self.min_points, self.max_points+1, 1)
        points = []
        for i in range(pointCount):
            x = random.randrange(self.margin, self.width-self.margin+1, 1)
            y = random.randrange(self.margin, self.height-self.margin+1, 1)
            distances = list(filter(lambda x: x < self.min_distance, [math.sqrt((p[0]-x)**2 + (p[1]-y)**2) for p in points]))
            if len(distances) == 0:
                points.append((x, y))
            return np.array(points)
    
    def get_track_points(hull, points):
        return np.array([points[hull.vertices[i]] for i in range(len(hull.vertices))])

    def make_rand_vector(dims):
        vec = [random.gauss(0, 1) for i in range(dims)]
        mag = sum(x**2 for x in vec) ** 0.5
        return [x/mag for x in vec]
    
    def shape_track(self, track_points):
        track_set = [[0,0] for i in range(len(track_points)*2)]
        for i in range(len(track_points)):
            displacement = math.pow(random.random(), self.difficulty) * self.max_displacement
            disp = [displacement * i for i in self.make_rand_vector(2)]
            track_set[i*2] = track_points[i]
            track_set[i*2 + 1][0] = int((track_points[i][0] + track_points[(i+1)%len(track_points)][0]) / 2 + disp[0])
            track_set[i*2 + 1][1] = int((track_points[i][1] + track_points[(i+1)%len(track_points)][1]) / 2 + disp[1])
        # for i in range(3):
        #     track_set = fix_angles(track_set)
        #     track_set = push_points_apart(track_set)
        final_set = []
        for point in final_set:
            if point[0] < self.margin:
                point[0] = self.margin
            elif point[0] > (self.width - self.margin):
                point[0] = self.width - self.margin
            if point[1] < self.margin:
                point[1] = self.width - self.margin
            final_set.append(point)
        return final_set

    def push_points_apart(self, points):
        # distance_sqr = self.distance_between_points ** 2
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                p_distance =  math.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
                if p_distance < self.distance_between_points:
                    dx = points[j][0] - points[i][0]
                    dy = points[j][1] - points[i][1]
                    dl = math.sqrt(dx*dx + dy*dy)
                    dx /= dl
                    dy /= dl
                    diff = self.distance_between_points - dl
                    dx *= dl
                    dy *= dl
                    points[j][0] = int(points[j][0] + dx)  
                    points[j][1] = int(points[j][1] + dy)  
                    points[i][0] = int(points[i][0] - dx)  
                    points[i][1] = int(points[i][1] - dy)
        return points
    
    def fix_angles(self, points):
        for i in range(len(points)):
            if i > 0:
                prev_point = i-1
            else:
                prev_point = len(points)-1
        next_point = (i+1) % len(points) 
        px = points[i][0] - points[prev_point][0]
        py = points[i][1] - points[prev_point][1]
        pl = math.sqrt(px*px + py*py)
        px /= pl
        py /= pl
        nx = -(points[i][0] - points[next_point][0])
        ny = -(points[i][1] - points[next_point][1])
        nl = math.sqrt(nx*nx + ny*ny)
        nx /= nl
        ny /= nl 
        a = math.atan2(px * ny - py * nx, px * nx + py * ny)
        if (abs(math.degrees(a)) <= self.max_angle):
            continue
        diff = math.radians(self.max_angle * math.copysign(1,a)) - a
        c = math.cos(diff)
        s = math.sin(diff)
        new_x = (nx * c - ny * s) * nl
        new_y = (nx * s + ny * c) * nl
        points[next_point][0] = int(points[i][0] + new_x)
        points[next_point][1] = int(points[i][1] + new_y)
        return points

    def smooth_track(track_points):
        x = np.array([p[0] for p in track_points])
        y = np.array([p[1] for p in track_points])
        x = np.r_[x, x[0]]
        y = np.r_[y, y[1]]
        tck, u = interpolate.splprep([x,y], s=0, per=True)
        xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
        return [(int(xi[i]), int(yi[i])) for i in range(len(xi))]

    def find_closest_point(points, keypoint):
        min_dist = None
        closest_point = None
        for p in points:
            dist = math.hypot(p[0]-keypoint[0], p[1]-keypoint[1])
            if min_dist is None or dist < min_dist:
                min_dist = dist
                closest_point = p
        return closest_point

    def get_corner_from_kp(self, complete_track, corner_kps):
        return [self.find_closest_point(complete_track, corner) for corner in corner_kps]

    def get_full_corners(self, track_points, corners, offset):
        corners_in_track = self.get_corner_from_kp(track_points, corners)
        f_corners = []
        for corner in corners_in_track:
            i = track_points.index(corner)
            tmp_track_points = track_points * 3
            f_corner = tmp_track_points[i+len(track_points)-1-offset:i+len(track_points)-1+offset]
            f_corners.append(f_corner)
        return f_corners 





                




    