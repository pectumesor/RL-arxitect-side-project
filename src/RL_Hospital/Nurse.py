import numpy as np
from constants import NEARBY_ZONE, TRAINING_STEPS, RAYS
from lidar import Lidar
import random
from shapely import Point

class FSMNurseState:
    CHECKING_PATIENT_LIST = 'CheckingPatientList'
    PERFORMING_ROUTE = 'PerformingRoute'
    TREATING_PATIENT = 'TreatingPatient'
    RETURNING_TO_STATION = 'ReturningToStation'
    COMPLETED = 'Completed'
    OUTSIDE = 'Outside'

class NurseAgent:
    def __init__(self, poi: dict, walls, bounding_box, box_size):
        self.poi = poi
        self.current_target = [0.0, 0.0]
        self.patients_left = self.poi["patient_rooms"]
        self.state = FSMNurseState.CHECKING_PATIENT_LIST
        self.position = None
        self.speed = 5
        self.facing = np.pi / 2  # Facing North initially
        self.walls = walls
        self.bounding_box = bounding_box
        self.box_size = box_size

        self.epsilon = 0.8
        self.treatment_delay = 10

        self.actions = {
            0: ("turn", -np.pi / 16),  # Turn left
            1: ("move", 0.0),         # Forward
            2: ("turn", np.pi / 16),   # Turn right
        }

        self.action_mask = np.pi / 2

        self.num_actions = len(self.actions)

        self.lidar = Lidar(walls=self.walls,
                           max_range=max(box_size[0], box_size[1]),
                           num_rays=RAYS)


    def reset(self):
        self.position = random.choice(self.poi['nurse_stations'])
        self.state = FSMNurseState.CHECKING_PATIENT_LIST
        self.facing = random.choice(np.linspace(0.0, 2* np.pi, self.num_actions))
        self.patients_left = self.poi["patient_rooms"]
        self.current_target = random.choice(self.patients_left)
        return self.get_observation()

    def update(self, action):
        prev_state = self.state

        collided = self._fsm_step(action)

        ''' 
         To keep rewards in the [-1, 1] range, 
         we will subtract all of them by the amount of training steps
        '''

        reward = -1 / TRAINING_STEPS  # Time Penalty
        done = False

        # Bonus for key state transitions
        if self.state != prev_state:
            if self.state == FSMNurseState.TREATING_PATIENT:
                reward += 1 / TRAINING_STEPS
            elif self.state == FSMNurseState.COMPLETED:
                reward += 1 / TRAINING_STEPS
                done = True # Completed all the goals

        obs = self.get_observation()
        info = {'state': self.state, 'collided': collided}

        return obs, reward, done, False, info

    def _fsm_step(self, action):
        match self.state:
            case FSMNurseState.CHECKING_PATIENT_LIST:
                self.current_target = random.choice(self.patients_left)
                self.state = FSMNurseState.PERFORMING_ROUTE
                return False

            case FSMNurseState.PERFORMING_ROUTE:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    return self.step(action)
                else:
                    self.state = FSMNurseState.TREATING_PATIENT
                    return False

            case FSMNurseState.TREATING_PATIENT:
                if self.treatment_delay > 0:
                    self.treatment_delay -= 1
                    return False
                else:
                    self.patients_left = [p for p in self.patients_left if not np.array_equal(p, self.current_target)]
                    if len(self.patients_left) <= 0:
                        self.current_target = random.choice(self.poi['nurse_stations'])
                        self.state = FSMNurseState.RETURNING_TO_STATION
                    else:
                        self.current_target = random.choice(self.patients_left)
                        self.treatment_delay = 100
                        self.state = FSMNurseState.PERFORMING_ROUTE
                    return False

            case FSMNurseState.RETURNING_TO_STATION:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    return self.step(action)
                else:
                    self.state = FSMNurseState.COMPLETED
                    return False

            case FSMNurseState.COMPLETED:
                return False

            case FSMNurseState.OUTSIDE:
                return True

    def step(self, action):

        action_type, angle = self.actions[int(action)]
        movement_angle = (self.facing + angle) % (2 * np.pi)

        if action_type == "turn":
            self.facing = movement_angle
            return False # No Movement

        collided, wall = self.lidar.check_collision(self.position, movement_angle, self.speed)

        if collided:
            distance_to_collision = Point(self.position).distance(wall)
            # Stop a small distance from obstacle
            dx = np.cos(movement_angle) * max(0.0, distance_to_collision - NEARBY_ZONE)
            dy = np.sin(movement_angle) * max (0.0, distance_to_collision - NEARBY_ZONE)
        else:
            dx = np.cos(movement_angle) * self.speed
            dy = np.sin(movement_angle) * self.speed

        movement = np.array([dx, dy], dtype=np.float32)
        new_position = self.position + movement

        self.position = new_position
        self.facing = movement_angle

        return collided

    def get_observation(self):
        distances, obstacle_types = self.lidar.get_readings(self.position, self.facing, self.current_target)
        obs = np.concatenate([distances, obstacle_types])
        return obs
