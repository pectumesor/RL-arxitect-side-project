# Lidar.py

import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

class Lidar:
    def __init__(self, walls, max_range=5.0, num_rays=360):

        self.walls = walls
        self.max_range = max_range
        self.num_rays = num_rays
        self.angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        self.clear_angles = []  # Angles with no obstacle detected


        self.wall_tree = STRtree(self.walls)

    def get_readings(self, position):

        readings = np.full(self.num_rays, self.max_range, dtype=np.float32)
        self.clear_angles = []
        origin = Point(position)

        for i, angle in enumerate(self.angles):
            # Calculate the end point of the ray

            relative_angle = np.clip(angle, -2 * np.pi, 2 * np.pi)
            # Compute global movement direction

            dx = np.cos(relative_angle) * self.max_range
            dy = np.sin(relative_angle) * self.max_range
            ray = LineString([origin, (origin.x + dx, origin.y + dy)])

            intersected_walls_tree = [ray.intersection(self.walls[idx]) for idx in self.wall_tree.query(ray)]

            # Initialize minimum distance for this ray
            min_distance = self.max_range

            for intersection in intersected_walls_tree:
                if not intersection.is_empty:
                    if isinstance(intersection, Point):
                        distance = origin.distance(intersection)
                        if distance < min_distance:
                            min_distance = distance
                    elif isinstance(intersection, LineString):
                        # If the intersection is a LineString, take the first point
                        distance = origin.distance(Point(intersection.coords[0]))
                        if distance < min_distance:
                            min_distance = distance
                    elif hasattr(intersection, 'geoms'):
                        # Handle MultiPoint or other geometry collections
                        for geom in intersection.geoms:
                            if isinstance(geom, Point):
                                distance = origin.distance(geom)
                                if distance < min_distance:
                                    min_distance = distance

            readings[i] = min_distance

            # Save angles where no obstacle was detected (distance == max_range)
            if min_distance == self.max_range:
                self.clear_angles.append(angle)

        return readings

    def check_collision(self, position, angle, speed):
        """
        Checks if moving in the given angle direction from the given position would result in a collision.

        :param position: Tuple or array-like representing the (x, y) position of the agent.
        :param angle: Angle in radians indicating the direction to check.
        :return: True if a collision would occur, False otherwise.
        """
        # Calculate the end point of the movement
        dx = np.cos(angle) * speed
        dy = np.sin(angle) * speed
        movement_line = LineString([Point(position), Point(position[0] + dx, position[1] + dy)])
        intersected_walls_tree = [movement_line.intersection(self.walls[idx]) for idx in self.wall_tree.query(movement_line)]
        # Check if the movement line intersects any wall
        return len(intersected_walls_tree) > 0

    def find_door(self, position):
        """
        Find the furthest reachable position from the current position without colliding with any walls.

        :param position: Current position of the agent as a NumPy array or similar.
        :return: New position of the agent after moving towards the furthest reachable point.
        """
        # Retrieve Lidar readings based on current position
        readings = self.get_readings(position)  # Returns a NumPy array of distances

        # Find the direction with the maximum reading
        idx = np.argmax(readings)
        angle_rad = self.angles[idx]
        distance = readings[idx]

        # To avoid being exactly on the wall, subtract a small epsilon
        epsilon = 1e-3
        move_distance = max(distance - epsilon, 0.0)  # Ensure non-negative

        # Compute new position by moving move_distance in the chosen direction
        dx = np.cos(angle_rad) * move_distance
        dy = np.sin(angle_rad) * move_distance
        new_position = position + np.array([dx, dy], dtype=np.float32)

        # Ensure the new position is not on top of a wall or colliding
        if self.on_top_of_wall(new_position) or self.check_collision(position, angle_rad, move_distance):
            # Find all clear directions (readings equal to max_range)
            clear_indices = np.where(readings >= self.max_range)[0]

            if len(clear_indices) > 0:
                # Choose the direction with the maximum reading among clear directions
                chosen_idx = clear_indices[np.argmax(readings[clear_indices])]
                angle_rad = self.angles[chosen_idx]
                move_distance = self.max_range - epsilon  # Move as far as possible without collision

                # Compute the new position in the chosen clear direction
                dx = np.cos(angle_rad) * move_distance
                dy = np.sin(angle_rad) * move_distance
                new_position = position + np.array([dx, dy], dtype=np.float32)

                # Final collision check
                if self.on_top_of_wall(new_position) or self.check_collision(position, angle_rad,move_distance):
                    # If still colliding, do not move
                    new_position = position
            else:
                # No clear directions available, do not move
                new_position = position

        return new_position

    def on_top_of_wall(self, position):
        origin = Point(position)
        contained_in_walls_tree = [origin.intersection(self.walls[idx]) for idx in self.wall_tree.query(origin, predicate="contains")]
        return len(contained_in_walls_tree) > 0


