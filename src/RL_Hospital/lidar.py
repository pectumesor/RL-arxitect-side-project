# Lidar.py
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from constants import NEARBY_ZONE

class Lidar:
    def __init__(self, walls, max_range=5.0, num_rays=360):
        self.walls = walls
        self.max_range = max_range
        self.num_rays = num_rays
        self.angles = None
        self.wall_tree = STRtree(self.walls)

    def get_readings(self, position, facing_angle, goal_position):
        readings = np.zeros((self.num_rays,1), dtype=np.float32)
        reading_types = np.zeros((self.num_rays,1), dtype=np.float32)

        origin = Point(position)
        goal = Point(goal_position)

        angle_min = facing_angle - np.pi/2
        angle_max = facing_angle + np.pi/2
        self.angles = np.linspace(angle_min, angle_max, self.num_rays)

        for i, angles in enumerate(self.angles):
            dx = np.cos(angles) * self.max_range
            dy = np.sin(angles) * self.max_range
            ray = LineString([origin, (origin.x + dx, origin.y + dy)])
            intersections = [ray.intersection(walls) for walls in self.walls]
            min_distance = self.max_range

            for intersection in intersections:
                if origin.distance(intersection) < min_distance:
                    min_distance = origin.distance(intersection)
                    if goal.distance(intersection) <= NEARBY_ZONE:
                        # Our ray intersected the goal, so we saw the goal
                        reading_types[i] = 1
                    else:
                        # Our ray intersected a wall or some obstacle
                        reading_types[i] = 0


            readings[i] = min_distance

        return readings, reading_types

    def check_collision(self, position, angle, speed):
        """
        Checks if moving in the given angle direction from the given position would result in a collision.

        :param speed: Step size
        :param position: Tuple or array-like representing the (x, y) position of the agent.
        :param angle: Angle in radians indicating the direction to check.
        :return: True if a collision would occur, False otherwise.
        """
        # Calculate the end point of the movement
        dx = np.cos(angle) * speed
        dy = np.sin(angle) * speed
        movement_line = LineString([Point(position), Point(position[0] + dx, position[1] + dy)])
        for wall in self.walls:
            if movement_line.intersects(wall):
                return True, movement_line.intersection(wall)

        return False, None
