import gymnasium as gym
from gymnasium import spaces
import numpy as np
from shapely.geometry import LineString
from Nurse import  NurseAgent
import pygame
from constants import RAYS

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

        nurse_station =  [np.array([station['x'], station['y']]) for station
                                    in poi['nurse_station'].values()]
        patient_room = [np.array([station['x'], station['y']]) for station
                                    in poi['patient_room'].values()]


        self.agent = NurseAgent(
            {
                'nurse_stations': nurse_station,
                "patient_rooms": patient_room
            },
                self.walls,bounding_box=self.bounding_box, box_size=self.box_size)

        # ---- OBSERVATION SPACE ---- #
        '''
            Our observation space consists of:
                - An array of size 2 * RAYS
                - RAYS entries define the distances each ray reaches until first obstacle
                - The rest RAYS entries define which type of object the ray collided with:
                    - 1 if it was the goal
                    - 0 else
        '''

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * RAYS,),
            dtype=np.float32
        )

        # ---- ACTION SPACE ---- #

        '''
            This is define in the Nurse.py class. We have in total 3 actions:
                - Action 0: Turn leftwards with an angle of np.pi / 16
                - Action 1: Move straight ahead
                - Action 2: Turn rightwards with an angle of np.pi / 16
        '''

        self.action_space = spaces.Discrete(self.agent.num_actions)

    def _get_node_position(self, graph_data, node_id):
        for node in graph_data["nodes"]:
            if node["v"] == node_id:
                return (node["value"]["x"], node["value"]["y"])
        raise ValueError(f"Node {node_id} not found in graph data.")

    def reset(self, seed=None, options=None):
        # Reset agents position and return his observation (i.e., position)
        super().reset(seed=seed)
        obs = self.agent.reset()
        return obs.flatten(), {}

    def step(self, action):
        obs, reward, done, truncated, info = self.agent.update(action)
        return obs.flatten(), reward, done, truncated, info

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
        ray_length = 1 * self.agent.lidar.max_range
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
        self.clock.tick(20)


