# Lidar.py
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

class Lidar:
    def __init__(self, walls, max_range=5.0, num_rays=360):
        self.walls = walls
        self.max_range = max_range
        self.num_rays = num_rays
        self.clear_angles = []  # Angles with no obstacle detected
        self.angles = None
        self.wall_tree = STRtree(self.walls)

    def get_readings(self, position, facing_angle):
        readings = np.zeros((self.num_rays, 2), dtype=np.float32)
        self.clear_angles = []

        origin = Point(position)

        angle_min = facing_angle - np.pi / 2
        angle_max = facing_angle + np.pi / 2
        self.angles = np.linspace(angle_min, angle_max, self.num_rays)

        for i, angle in enumerate(self.angles):
            dx = np.cos(angle) * self.max_range
            dy = np.sin(angle) * self.max_range
            ray = LineString([origin, (origin.x + dx, origin.y + dy)])

            try:
                intersections = [ray.intersection(self.walls[idx]) for idx in self.wall_tree.query(ray)]
            except Exception:
                intersections = []

            min_distance = self.max_range

            for intersection in intersections:
                if not intersection.is_empty:
                    if isinstance(intersection, Point):
                        distance = origin.distance(intersection)
                        if distance < min_distance:
                            min_distance = distance
                    elif isinstance(intersection, LineString):
                        first_coord = intersection.coords[0]
                        if first_coord is not None:
                            distance = origin.distance(Point(first_coord))
                            if distance < min_distance:
                                min_distance = distance
                    elif hasattr(intersection, 'geoms'):
                        for geom in intersection.geoms:
                            if isinstance(geom, Point):
                                distance = origin.distance(geom)
                                if distance < min_distance:
                                    min_distance = distance

            if np.isnan(min_distance) or np.isinf(min_distance):
                min_distance = self.max_range

            # ✅ Safer calculation
            endpoint_x = origin.x + np.clip(np.cos(angle), -1.0, 1.0) * np.clip(min_distance, 0.0, self.max_range)
            endpoint_y = origin.y + np.clip(np.sin(angle), -1.0, 1.0) * np.clip(min_distance, 0.0, self.max_range)

            # 🔥 Final check: if still invalid, fallback
            if np.isnan(endpoint_x) or np.isinf(endpoint_x):
                endpoint_x = origin.x
            if np.isnan(endpoint_y) or np.isinf(endpoint_y):
                endpoint_y = origin.y

            readings[i] = np.array([endpoint_x, endpoint_y], dtype=np.float32)

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

    def find_door(self, position, facing_angle):
        endpoints = self.get_readings(position, facing_angle)  # (num_rays, 2)

        origin = np.array(position, dtype=np.float32)
        max_distance = -np.inf
        best_point = None

        for point in endpoints:
            if isinstance(point, np.ndarray) and point.shape == (2,) and not np.any(np.isnan(point)) and not np.any(
                    np.isinf(point)):
                dist = np.linalg.norm(point - origin)
                if dist > max_distance:
                    max_distance = dist
                    best_point = point

        # Safety fallback
        if best_point is None:
            best_point = origin

        # Safety checks again
        if not isinstance(best_point, np.ndarray) or best_point.shape != (2,) or np.any(np.isnan(best_point)) or np.any(
                np.isinf(best_point)):
            best_point = origin

        # Now SAFE to use best_point
        direction = best_point - origin
        if np.linalg.norm(direction) == 0:
            direction = np.array([1.0, 0.0], dtype=np.float32)  # Just to avoid atan2 issues

        angle_to_best = np.arctan2(direction[1], direction[0])

        if self.on_top_of_wall(best_point) or self.check_collision(origin, angle_to_best, max_distance):
            return origin  # fallback to current position if unsafe

        return best_point

    def on_top_of_wall(self, position):
        origin = Point(position)
        contained_in_walls_tree = [origin.intersection(self.walls[idx]) for idx in self.wall_tree.query(origin, predicate="contains")]
        return len(contained_in_walls_tree) > 0


