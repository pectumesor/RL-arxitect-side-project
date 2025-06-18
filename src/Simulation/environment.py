import shapely
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely import polygonize, multipolygons
from Nurse import  NurseAgent
import random
from lidar import Lidar
import pygame
from constants import BUFFER_SIZE

class MyMultiGoalEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, poi, graph_data, node_positions, box_size, bounding_box, render_mode=None):
        self.render_mode = render_mode
        self.poi = poi
        self.graph_data = graph_data
        self.node_positions = node_positions
        self.box_size = box_size
        self.bounding_box = bounding_box
        self.graph = []
        self.walls = []

        self.box_center = np.array(box_size) / 2

        # Load graph edges as walls, offsetting positions
        for edge in graph_data["edges"]:
            node_v = edge["v"]
            node_w = edge["w"]
            pos_v = self._get_node_position(graph_data, node_v)
            pos_w = self._get_node_position(graph_data, node_w)
            self.graph.append((pos_v, pos_w))
            self.walls.append(LineString([pos_v, pos_w]))

        # Define free space (white space)
        bounding_polygon = Polygon([
            (self.bounding_box[0], self.bounding_box[1]),
            (self.bounding_box[2], self.bounding_box[1]),
            (self.bounding_box[2], self.bounding_box[3]),
            (self.bounding_box[0], self.bounding_box[3])
        ])
        self.white_space = bounding_polygon.difference(unary_union(self.walls))

        # Find all polygons in the drawing
        drawing_polygons = shapely.multipolygons(shapely.get_parts(polygonize(self.walls)))
        # Put them in an iterable list
        self.inner_polygons = [geom for geom in drawing_polygons.geoms]
        # Find the polygon with the largest area, this will be the outer walls of the drawing
        self.shape = self.extract_outer_polygon(self.inner_polygons)
        # List only holds the inner polygons
        self.inner_polygons.remove(self.shape)
        # Turn list back to Multipolygon to be able to use shapely.Multipolygon functions
        self.inner_polygons = shapely.multipolygons(shapely.get_parts(self.inner_polygons))

        # Initialize Lidar with max_range equal to speed
        self.lidar = Lidar(walls=self.walls, max_range=200, num_rays=360)

        nurse_station =  [np.array([station['x'], station['y']]) for station
                                    in poi['nurse_station'].values()]
        patient_room = [np.array([station['x'], station['y']]) for station
                                    in poi['patient_room'].values()]

        amount_patients = 1
        random_nurse_station = random.choice(nurse_station)
        random_patients = random.choices(patient_room, k=amount_patients)
        self.agent = NurseAgent(
            {
                'nurse_station': {'position': random_nurse_station},
                f"patient_room_{0}": {'position': random_patients[0]}
            },
                self.walls, amount_patients=amount_patients,
                bounding_box=self.bounding_box)

        # Define observation and action space
        lidar_dim = self.agent.lidar.num_rays
        goal_dim = 2
        position_dim = 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lidar_dim + goal_dim + position_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(self.agent.num_actions)

    def _get_node_position(self, graph_data, node_id):
        for node in graph_data["nodes"]:
            if node["v"] == node_id:
                return (node["value"]["x"], node["value"]["y"])
        raise ValueError(f"Node {node_id} not found in graph data.")

    def extract_outer_polygon(self, multipolygon):
        return max(multipolygon, key=lambda p: p.area)

    def reset(self, seed=None, options=None):
        # Reset agents position and return his observation (i.e., position)
        super().reset(seed=seed)
        obs = self.agent.reset()
        return obs, {}


    def step(self, action):
        obs, reward, done, truncated, info = self.agent.update(action)
        return obs, reward, done, truncated, info

    def to_pygame_coords(self, point):
        """Convert world (x, y) to pygame screen coordinates."""
        x, y = point
        return int(x - self.bounding_box[0]), int(y - self.bounding_box[1])

    def render(self, mode="human"):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((0, 0))  # Fullscreen
            pygame.display.set_caption(f"Nurse Agent Environment, Current State: {self.agent.state}")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.fill((255, 255, 255))  # White background

        # 🧱 Draw walls
        for wall in self.walls:
            coords = list(wall.coords)
            for i in range(len(coords) - 1):
                p1 = self.to_pygame_coords(coords[i])
                p2 = self.to_pygame_coords(coords[i + 1])
                pygame.draw.line(self.screen, (0, 0, 0), p1, p2, 2)

        # 🏥 Draw nurse stations
        for station in self.poi['nurse_station'].values():
            pos = self.to_pygame_coords((station['x'], station['y']))
            pygame.draw.circle(self.screen, (255, 0, 0), pos, 6)

        # 🛏️ Draw patient rooms
        for room in self.poi['patient_room'].values():
            pos = self.to_pygame_coords((room['x'], room['y']))
            pygame.draw.circle(self.screen, (255, 0, 0), pos, 6)

        # 👩‍⚕️ Draw agent
        nurse_pos = self.to_pygame_coords(self.agent.position)
        pygame.draw.circle(self.screen, (0, 100, 255), nurse_pos, 6)

        # ✅ Draw current target
        goal_pos = self.to_pygame_coords(self.agent.current_target)
        pygame.draw.circle(self.screen, (0, 255, 0), goal_pos, 8)

        # 🔷 Visualize 180° forward movement cone
        move_angles = [-np.pi / 2, -np.pi / 4, 0.0, np.pi / 4, np.pi / 2]
        ray_length = 30
        cx, cy = self.agent.position
        facing = self.agent.facing

        for rel_angle in move_angles:
            angle = facing + rel_angle
            dx = np.cos(angle) * ray_length
            dy = np.sin(angle) * ray_length
            start = self.to_pygame_coords((cx, cy))
            end = self.to_pygame_coords((cx + dx, cy + dy))
            pygame.draw.line(self.screen, (150, 150, 255), start, end, 1)

        # 🔵 Bold line for current facing direction
        dx = np.cos(facing) * ray_length
        dy = np.sin(facing) * ray_length
        end_facing = self.to_pygame_coords((cx + dx, cy + dy))
        pygame.draw.line(self.screen, (0, 0, 255), nurse_pos, end_facing, 2)

        pygame.display.flip()
        self.clock.tick(10)


