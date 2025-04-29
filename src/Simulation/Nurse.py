import numpy as np
from constants import NEARBY_ZONE, LARGE_VALUE, FRACTION
from lidar import Lidar
import random

class FSMNurseState:
    CHECKING_PATIENT_LIST = 'CheckingPatientList'
    LEAVING_NURSE_STATION = 'LeavingNurseStation'
    PERFORMING_ROUTE = 'PerformingRoute'
    REACHED_PATIENT = 'ReachedPatient'
    TREATING_PATIENT = 'TreatingPatient'
    LEAVING_PATIENT_ROOM = 'LeavingPatientRoom'
    RETURNING_TO_STATION = 'ReturningToStation'
    REACHED_STATION = 'ReachedStation'
    COMPLETED = 'Completed'


class NurseAgent:
    def __init__(self, poi: dict, walls, amount_patients: int, bounding_box):
        self.poi = poi
        self.curr_patient = 0
        self.current_target = None
        self.state = FSMNurseState.CHECKING_PATIENT_LIST
        self.position = None  # Starting position
        self.speed = 1.0  # Units per step
        self.facing = np.pi/2
        self.walls = walls
        self.amount_patients = amount_patients
        self.bounding_box = bounding_box

        self.epsilon = 0.8
        self.treatment_delay = 100

        # Set up angle range
        self.num_actions = 20
        self.angle_range = np.linspace(-np.pi / 2, np.pi / 2, self.num_actions)

        # Initialize Lidar with max_range equal to speed
        self.lidar = Lidar(walls=self.walls, max_range=50, num_rays=self.num_actions)

    def reset(self):
        self.position = self.poi['nurse_station']['position']
        self.curr_patient = 0
        self.state = FSMNurseState.CHECKING_PATIENT_LIST
        self.facing = np.pi/2
        self.current_target = self.poi['nurse_station']['door']
        return self.get_observation()

    def update(self, action):
        prev_position = self.position.copy()
        prev_state = self.state

        collided = self._fsm_step(action)  # Your existing FSM logic (moved to helper for clarity)

        reward = -1  # Step penalty to encourage faster completion
        done = False

        if collided: # Collided with wall or walked over one, very big penalization
            reward -= 100

        # Penalize no movement (e.g., stuck or bumping walls)
        movement = np.linalg.norm(self.position - prev_position)
        if movement < 0.1:
            reward -= 10 # Encourage not moving at all
        else:
            reward += 5 # Motivate movement

            # Bonus rewards for reaching goals
        if self.state != prev_state:
            if self.state in [FSMNurseState.REACHED_PATIENT, FSMNurseState.REACHED_STATION]:
                reward += 50
            elif self.state == FSMNurseState.COMPLETED:
                reward += 100
                done = True

        obs = self.get_observation()
        info = {'state': self.state }

        return obs, reward, done, False, info

    def _fsm_step(self, action):

        match self.state:
            case FSMNurseState.CHECKING_PATIENT_LIST:
                self.current_target = self.poi['nurse_station']['door']
                self.state = FSMNurseState.LEAVING_NURSE_STATION
                return False

            case FSMNurseState.LEAVING_NURSE_STATION:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    return self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi[f"patient_room_{self.curr_patient}"]['door']
                    self.state = FSMNurseState.PERFORMING_ROUTE
                    return False

            case FSMNurseState.PERFORMING_ROUTE:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    return self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi[f"patient_room_{self.curr_patient}"]['position']
                    self.state = FSMNurseState.REACHED_PATIENT
                    return False

            case FSMNurseState.REACHED_PATIENT:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    return self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.state = FSMNurseState.TREATING_PATIENT
                    return False

            case FSMNurseState.TREATING_PATIENT:
                if self.treatment_delay > 0:
                    self.treatment_delay -= 1
                    return False
                elif self.treatment_delay <= 0:
                    self.current_target = self.poi[f"patient_room_{self.curr_patient}"]['door']
                    self.curr_patient +=1
                    self.treatment_delay = 100
                    self.state = FSMNurseState.LEAVING_PATIENT_ROOM
                    return False

            case FSMNurseState.LEAVING_PATIENT_ROOM:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    return self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    if self.curr_patient >= self.amount_patients - 1:
                        self.current_target = self.poi['nurse_station']['door']
                        self.state = FSMNurseState.RETURNING_TO_STATION
                    else:
                        self.current_target = self.poi[f"patient_room_{self.curr_patient}"]['door']
                        self.state = FSMNurseState.PERFORMING_ROUTE
                    return False

            case FSMNurseState.RETURNING_TO_STATION:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target < NEARBY_ZONE:
                    self.speed = distance_to_target
                    return self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['nurse_station']['position']
                    self.state = FSMNurseState.REACHED_STATION
                    return False

            case FSMNurseState.REACHED_STATION:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target < NEARBY_ZONE:
                    self.speed = distance_to_target
                    return self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.state = FSMNurseState.COMPLETED
                    return False

            case FSMNurseState.COMPLETED:
                return False

    def step(self, action):
        """
        Executes a movement step based on the action and keeps the agent inside bounds.

        :param action: Discrete action index (0 to num_actions-1)
        """
        collided = False

        # Calculate movement vector
        relative_angle = self.angle_range[action]
        movement_angle = self.facing + relative_angle

        # Check collision
        collided = self.lidar.check_collision(self.position, movement_angle, self.speed)

        # Movement deltas
        dx = np.cos(movement_angle) * self.speed
        dy = np.sin(movement_angle) * self.speed
        movement = np.array([dx, dy], dtype=np.float32)

        if not np.any(np.isnan(movement)) and not np.any(np.isinf(movement)):
            new_position = self.position + movement

            # ✨ CLAMP to bounding box
            new_position[0] = np.clip(new_position[0], self.bounding_box[0] + 1, self.bounding_box[2] - 1)
            new_position[1] = np.clip(new_position[1], self.bounding_box[1] + 1, self.bounding_box[3] - 1)

            self.position = new_position
            self.facing = movement_angle % (2 * np.pi)

        return collided

    def compute_distance(self,action):
        # Determine movement angle based on action and Lidar data
        relative_angle = np.clip(action, -np.pi / 2, np.pi / 2)
        desired_angle = self.facing + relative_angle

        # Convert desired_angle to index in Lidar readings
        desired_angle_deg = int(np.degrees(desired_angle) % 10)
        distance_ahead = self.lidar_readings[desired_angle_deg]

        # Simple obstacle avoidance: if obstacle is too close, adjust the movement angle
        if distance_ahead < self.speed:
            # Obstacle detected ahead, choose a new direction among the clear angles
            desired_angle = random.choice(self.lidar.clear_angles)
            desired_angle %= 2 * np.pi

        # Calculate movement deltas
        dx = np.cos(desired_angle) * self.speed
        dy = np.sin(desired_angle) * self.speed
        return dx, dy, desired_angle

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

        obs = np.concatenate([lidar_obs, goal_vector])
        return obs.astype(np.float32)


