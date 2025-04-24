import shapely
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely import polygonize, multipolygons
from .Doctor import DoctorAgent
from .Nurse import NurseAgent
from .Patient import PatientAgent
from .lidar import Lidar
import random
from .constants import NEARBY_ZONE
from scipy.spatial import KDTree
"""
    Class that defines the environment of our simulation.
"""
class MultiAgentFreeSpaceEnv(ParallelEnv):
    def __init__(self, poi, graph_data, node_positions, bounding_box, box_size,
                 num_doctors, num_nurses, num_patients):
        self.poi = poi
        self.graph_data = graph_data
        self.node_positions = node_positions
        self.box_size = box_size
        self.bounding_box = bounding_box
        self.num_doctors = num_doctors
        self.num_nurses = num_nurses
        self.num_patients = num_patients
        self.graph = []
        self.walls = []

        # Initialize Lidar with max_range equal to speed
        self.lidar = Lidar(walls=self.walls, max_range=100, num_rays=360)

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

        # Define observation and action spaces
        # Assuming each agent observes its (x, y) position
        self.observation_space = spaces.Dict({
            'doctors': spaces.Box(low=np.array([self.bounding_box[0], self.bounding_box[1]]),
                                  high=np.array([self.bounding_box[2], self.bounding_box[3]]),
                                  dtype=np.float64),
            'nurses': spaces.Box(low=np.array([self.bounding_box[0], self.bounding_box[1]]),
                                 high=np.array([self.bounding_box[2], self.bounding_box[3]]),
                                 dtype=np.float64),
            'patients': spaces.Box(low=np.array([self.bounding_box[0], self.bounding_box[1]]),
                                   high=np.array([self.bounding_box[2], self.bounding_box[3]]),
                                   dtype=np.float64),
        })

        self.action_space = spaces.Box(low=-np.pi / 2, high=np.pi / 2, shape=(1,), dtype=np.float32)

        # Define the agents

        medical_clinic = [np.array([clinic['x'], clinic['y']]) for clinic
                                    in poi['medical_clinic'].values()]
        medication_room = [np.array([station['x'], station['y']]) for station
                                    in poi['medication_room'].values()]
        nurse_station =  [np.array([station['x'], station['y']]) for station
                                    in poi['nurse_station'].values()]
        patient_room = [np.array([station['x'], station['y']]) for station
                                    in poi['patient_room'].values()]
        elevator_lobby = [np.array([station['x'], station['y']]) for station
                                    in poi['elevator_lobby'].values()]

        # Add random noise, such that the agents are not exactly on top of each other

        doctors_data = []
        for i in range(num_doctors):
            noise = np.random.uniform(-1, 1, size=(2,))
            random_med_clinic = random.choice(medical_clinic) + noise
            random_med_station = random.choice(medication_room) + noise
            random_nurse_station = random.choice(nurse_station) + noise
            doctors_data.append({'name': f"doctor_{i}",
                                 'agent': DoctorAgent(
                                     {
                                         'clinic': {'position': random_med_clinic, 'door': self.lidar.find_door(random_med_clinic)},
                                         'med_station': {'position': random_med_station, 'door': self.lidar.find_door(random_med_station)},
                                         'nurse_station': {'position': random_nurse_station, 'door':self.lidar.find_door(random_nurse_station)}
                                             },
                                     self.walls)})


        nurses_data = []
        amount_patients = 3
        for i in range(num_nurses):
            noise = np.random.uniform(-1, 1, size=(2,))
            random_nurse_station = random.choice(nurse_station) + noise
            random_patients = random.choices(patient_room,k=amount_patients)
            nurses_data.append({'name': f"nurse_{i}",
                                'agent': NurseAgent(
                                    {
                                'nurse_station': {'position': random_nurse_station,   'door': self.lidar.find_door(random_nurse_station)},
                                f"patient_room_{0}": {'position': random_patients[0] + noise, 'door': self.lidar.find_door(random_patients[0])},
                                f"patient_room_{1}": {'position': random_patients[1] + noise, 'door': self.lidar.find_door(random_patients[1])},
                                f"patient_room_{2}": {'position': random_patients[2] + noise, 'door': self.lidar.find_door(random_patients[2])},
                                    },
                                self.walls,amount_patients=amount_patients)})

        patients_data = []
        for i in range(num_patients):
            noise = np.random.uniform(-1, 1, size=(2,))
            random_elevator_lobby = random.choice(elevator_lobby) + noise
            random_med_clinic = random.choice(medical_clinic) + noise
            patients_data.append({'name': f"patient_{i}",
                                  'agent': PatientAgent(
                                      {

                                          'elevator_lobby': {'position': random_elevator_lobby, 'door': self.lidar.find_door(random_elevator_lobby)},
                                          'medical_clinic': {'position': random_med_clinic, 'door': self.lidar.find_door(random_med_clinic)},
                                      },
                                      self.walls)})
        self.agents = {
            'doctors': doctors_data,
            'nurses': nurses_data,
            'patients': patients_data
        }

                                #### Functions ###
    def _get_node_position(self, graph_data, node_id):
        for node in graph_data["nodes"]:
            if node["v"] == node_id:
                return (node["value"]["x"], node["value"]["y"])
        raise ValueError(f"Node {node_id} not found in graph data.")


    def extract_outer_polygon(self, multipolygon):
        return max(multipolygon, key=lambda p: p.area)


    def reset(self):
        for doctors in self.agents['doctors']:
            doctors['agent'].reset()

        for nurses in self.agents['nurses']:
            nurses['agent'].reset()

        for patients in self.agents['patients']:
            patients['agent'].reset()


    def step(self, actions):
        observations = []
        observations.append({})
        observations.append({})

        for agent_type in self.agents:
            for agent in self.agents[agent_type]:
                agent['agent'].update(actions[agent['name']])
                observations[0][agent['name']] = agent['agent'].position

        # Initialize an empty set to store unique collision positions
        collisions = set()
        # Extract agents by type
        doctors = self.agents['doctors']
        nurses = self.agents['nurses']
        patients = self.agents['patients']

        # Combine all agents into a single list with their type and name
        all_agents = []
        for agent_type, agents in [('doctor', doctors), ('nurse', nurses), ('patient', patients)]:
            for agent in agents:
                all_agents.append({
                    'type': agent_type,
                    'name': agent['name'],
                    'position': agent['agent'].position
                })

        # Create a NumPy array of positions for KDTree
        positions = np.array([agent['position'] for agent in all_agents])

        # Build the KDTree
        tree = KDTree(positions)

        # Query all unique pairs within NEARBY_ZONE
        pairs = tree.query_pairs(r=NEARBY_ZONE)

        for i, j in pairs:
            agent_i = all_agents[i]
            agent_j = all_agents[j]

            collisions.add(tuple(agent_i['position']))
            collisions.add(tuple(agent_j['position']))

        # Convert the set of tuples back to a list of lists
        collisions = [list(pos) for pos in collisions]

        observations[1]['collisions'] = collisions

        return observations

    def render(self, mode="human"):
        plt.figure(figsize=(10, 10))
        print(self.box_size[0])
        plt.xlim(0, self.box_size[0])
        plt.ylim(0, self.box_size[1])

        # Draw walls
        for wall in self.walls:
            x, y = wall.xy
            plt.plot(x, y, color="black", linewidth=1)

        # Draw agents and goal
        for agent, position in self.agent_positions.items():
            plt.scatter(*position, label=agent, s=20)
        plt.scatter(*self.goal_position, color="green", label="Goal", s=100)

        plt.legend()
        plt.title("Multi-Agent Environment with Centered Graph Walls")
        plt.show()
