import numpy as np
from constants import NEARBY_ZONE
from lidar import Lidar
import random

class FSMNurseState:
    CHECKING_PATIENT_LIST = 'CheckingPatientList'
    PERFORMING_ROUTE = 'PerformingRoute'
    TREATING_PATIENT = 'TreatingPatient'
    RETURNING_TO_STATION = 'ReturningToStation'
    COMPLETED = 'Completed'
    OUTSIDE = 'Outside'

class NurseAgent:
    def __init__(self, poi: dict, walls, amount_patients: int, bounding_box):
        self.poi = poi
        self.curr_patient = 0
        self.current_target = None
        self.state = FSMNurseState.CHECKING_PATIENT_LIST
        self.position = None
        self.speed = 20.0
        self.facing = np.pi / 2  # Facing North initially
        self.walls = walls
        self.amount_patients = amount_patients
        self.bounding_box = bounding_box

        self.epsilon = 0.8
        self.treatment_delay = 10

        self.actions = {
            0: ("face", 0),          # East
            1: ("face", np.pi / 2),  # North
            2: ("face", np.pi),      # West
            3: ("face", 3 * np.pi / 2),  # South
            4: ("move", -np.pi / 2),  # Left
            5: ("move", -np.pi / 4),  # Forward-left
            6: ("move", 0.0),         # Forward
            7: ("move", np.pi / 4),   # Forward-right
            8: ("move", np.pi / 2),   # Right
        }
        self.num_actions = len(self.actions)

        self.lidar = Lidar(walls=self.walls, max_range=self.speed, num_rays=self.num_actions)

        self.position_history = []
        self.position_history_maxlen = 20
        self.revisit_distance_threshold = 20.0

    def reset(self):
        self.position = self.poi['nurse_station']['position']
        self.curr_patient = 0
        self.state = FSMNurseState.CHECKING_PATIENT_LIST
        self.facing = np.pi / 2
        self.current_target = self.poi[f'patient_room_{self.curr_patient}']['position']
        self.position_history.clear()
        return self.get_observation()

    def update(self, action):
        prev_position = self.position.copy()
        prev_state = self.state

        collided = self._fsm_step(action)

        reward = -1  # Step penalty
        done = False

        if collided:
            reward -= 50  # Flat collision penalty

        # Reward for moving closer to goal
        dist_prev = np.linalg.norm(prev_position - self.current_target)
        dist_now = np.linalg.norm(self.position - self.current_target)
        if dist_now < dist_prev:
            reward += 100  # Getting closer

        # Bonus for key state transitions
        if self.state != prev_state:
            if self.state == FSMNurseState.TREATING_PATIENT:
                reward += 50
            elif self.state == FSMNurseState.COMPLETED:
                reward += 100
                done = True
            elif self.state == FSMNurseState.OUTSIDE: # Outside the drawing, very bad
                reward -=100
                done = True

        obs = self.get_observation()
        info = {'state': self.state}

        return obs, reward, done, False, info

    def _fsm_step(self, action):
        match self.state:
            case FSMNurseState.CHECKING_PATIENT_LIST:
                self.current_target = self.poi[f'patient_room_{self.curr_patient}']['position']
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
                    self.curr_patient += 1
                    if self.curr_patient >= self.amount_patients:
                        self.current_target = self.poi['nurse_station']['position']
                        self.state = FSMNurseState.RETURNING_TO_STATION
                    else:
                        self.current_target = self.poi[f'patient_room_{self.curr_patient}']['position']
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
        action = int(action)
        print("Action:", action, self.actions[action])
        action_type, angle = self.actions[action]

        if action_type == "face":
            self.facing = angle % (2 * np.pi)
            return False  # No movement

        elif action_type == "move":
            movement_angle = (self.facing + angle) % (2 * np.pi)

            dx = np.cos(movement_angle) * self.speed
            dy = np.sin(movement_angle) * self.speed
            movement = np.array([dx, dy], dtype=np.float32)

            new_position = self.position + movement

            x_out = new_position[0] < self.bounding_box[0] or new_position[0] > self.bounding_box[2]
            y_out = new_position[1] < self.bounding_box[1] or new_position[1] > self.bounding_box[3]

            if x_out or y_out:
                self.state = FSMNurseState.OUTSIDE
                return True

            collided = self.lidar.check_collision(self.position, movement_angle, self.speed)
            if collided:
                return True

            self.position = new_position
            self.facing = movement_angle
            return False

    def get_observation(self):
        lidar_obs = self.lidar.get_readings(self.position, self.facing).flatten()
        assert not np.any(np.isnan(lidar_obs)), "Lidar readings contain NaNs!"
        assert not np.any(np.isinf(lidar_obs)), "Lidar readings contain Infs!"

        goal_vector = self.current_target - self.position
        norm = np.linalg.norm(goal_vector)
        if norm > 1e-6:
            goal_vector /= norm
        else:
            goal_vector = np.zeros_like(goal_vector)

        obs = np.concatenate([lidar_obs, goal_vector, self.position])
        return obs.astype(np.float32)
