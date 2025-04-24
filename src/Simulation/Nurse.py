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
    def __init__(self, poi: dict, walls, amount_patients: int):
        self.poi = poi
        self.curr_patient = 0
        self.current_target = None
        self.state = FSMNurseState.CHECKING_PATIENT_LIST
        self.position = None  # Starting position
        self.speed = 10  # Units per step
        self.facing = 0.0
        self.walls = walls
        self.amount_patients = amount_patients

        self.epsilon = 0.8
        self.treatment_delay = 100

        # Initialize Lidar with max_range equal to speed
        self.lidar = Lidar(walls=self.walls, max_range=50, num_rays=80)

        # Set up angle range
        self.num_actions = 20
        self.angle_range = np.linspace(-np.pi / 2, np.pi / 2, self.num_actions)

    def reset(self):
        self.position = self.poi['nurse_station']['position']
        self.curr_patient = 0
        self.state = FSMNurseState.CHECKING_PATIENT_LIST
        self.facing = 0.0
        self.current_target = self.poi['nurse_station']['door']
        return self.position

    def update(self, action):
        prev_position = self.position.copy()
        prev_state = self.state

        self._fsm_step(action)  # Your existing FSM logic (moved to helper for clarity)

        reward = -1  # Step penalty to encourage faster completion
        done = False

        # Reward reaching key locations
        if self.state != prev_state:
            if self.state in [FSMNurseState.REACHED_PATIENT, FSMNurseState.REACHED_STATION]:
                reward += 10
            elif self.state == FSMNurseState.COMPLETED:
                reward += 50
                done = True

        # Penalize no movement (e.g., stuck or bumping walls)
        movement = np.linalg.norm(self.position - prev_position)
        if movement < 0.1:
            reward -= 2
        else:
            reward += 5 # Motivate movement

        obs = self.position
        info = {'state': self.state, 'step_penalty': -1}

        return obs, reward, done, False, info

    def _fsm_step(self, action):

        match self.state:
            case FSMNurseState.CHECKING_PATIENT_LIST:
                self.current_target = self.poi['nurse_station']['door']
                self.state = FSMNurseState.LEAVING_NURSE_STATION
                return

            case FSMNurseState.LEAVING_NURSE_STATION:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                    return
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi[f"patient_room_{self.curr_patient}"]['door']
                    self.state = FSMNurseState.PERFORMING_ROUTE
                    return

            case FSMNurseState.PERFORMING_ROUTE:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                    return
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi[f"patient_room_{self.curr_patient}"]['position']
                    self.state = FSMNurseState.REACHED_PATIENT
                    return

            case FSMNurseState.REACHED_PATIENT:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                    return
                elif distance_to_target <= NEARBY_ZONE:
                    self.state = FSMNurseState.TREATING_PATIENT
                    return

            case FSMNurseState.TREATING_PATIENT:
                if self.treatment_delay > 0:
                    self.treatment_delay -= 1
                    return
                elif self.treatment_delay <= 0:
                    self.current_target = self.poi[f"patient_room_{self.curr_patient}"]['door']
                    self.curr_patient +=1
                    self.treatment_delay = 100
                    self.state = FSMNurseState.LEAVING_PATIENT_ROOM
                    return

            case FSMNurseState.LEAVING_PATIENT_ROOM:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                    return
                elif distance_to_target <= NEARBY_ZONE:
                    if self.curr_patient >= self.amount_patients - 1:
                        self.current_target = self.poi['nurse_station']['door']
                        self.state = FSMNurseState.RETURNING_TO_STATION
                    else:
                        self.current_target = self.poi[f"patient_room_{self.curr_patient}"]['door']
                        self.state = FSMNurseState.PERFORMING_ROUTE
                    return

            case FSMNurseState.RETURNING_TO_STATION:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target < NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                    return
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['nurse_station']['position']
                    self.state = FSMNurseState.REACHED_STATION
                    return

            case FSMNurseState.REACHED_STATION:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target < NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                    return
                elif distance_to_target <= NEARBY_ZONE:
                    self.state = FSMNurseState.COMPLETED
                    return

            case FSMNurseState.COMPLETED:
                pass

    def step(self, action):
        """
        Executes a movement step based on the action and Lidar readings.

        :param action: Action index determining the movement angle (0 to 359 degrees).
        """

        #Apply fraction to control the speed
        self.speed *= FRACTION

        relative_angle = self.angle_range[action]
        movement_angle = self.facing + relative_angle
        # Calculate movement vector

        while self.speed >= 1:
            if self.lidar.check_collision(self.position, movement_angle,self.speed):
                self.speed = max(self.speed/2, 0.9)
            else:
                break


        # Check if the desired movement direction is clear
        if not self.lidar.check_collision(self.position, movement_angle, self.speed):
            # Move in the desired direction
            dx = np.cos(movement_angle) * self.speed
            dy = np.sin(movement_angle) * self.speed
            self.position += np.array([dx, dy], dtype=np.float32).flatten()
            self.facing = movement_angle % (2 * np.pi)
        else:
            # Obstacle detected in the desired direction, attempt to find an alternative direction
            # For simplicity, try to turn right by 45 degrees
            alternative_angle_deg = np.clip(action + 45, -np.pi / 2, np.pi / 2)
            alternative_angle_rad = np.deg2rad(alternative_angle_deg)

            if not self.lidar.check_collision(self.position, alternative_angle_rad, self.speed):
                dx = np.cos(alternative_angle_rad) * self.speed
                dy = np.sin(alternative_angle_rad) * self.speed
                self.position += np.array([dx, dy], dtype=np.float32).flatten()
                self.facing = alternative_angle_rad
            else:
                # If right turn is blocked, try to turn left by 45 degrees
                alternative_angle_deg = np.clip(action + 45, -np.pi / 2, np.pi / 2)
                alternative_angle_rad = np.deg2rad(alternative_angle_deg)

                if not self.lidar.check_collision(self.position, alternative_angle_rad, self.speed):
                    dx = np.cos(alternative_angle_rad) * self.speed
                    dy = np.sin(alternative_angle_rad) * self.speed
                    self.position += np.array([dx, dy], dtype=np.float32).flatten()
                    self.facing = alternative_angle_rad
                else:
                    # If all directions are blocked, go back
                    backward_angle = (self.facing + np.pi) % (2 * np.pi)
                    dx = np.cos(backward_angle) * self.speed
                    dy = np.sin(backward_angle) * self.speed
                    self.position += np.array([dx, dy], dtype=np.float32).flatten()
                    self.facing = backward_angle


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

    def get_action(self, chance):
        # Update Lidar readings
        self.lidar_readings = self.lidar.get_readings(self.position)

        # Compute distance to goal among the clear angles
        min_distance_to_goal = LARGE_VALUE
        desired_action = None

        if len(self.lidar.clear_angles) <= 0:
            desired_action = np.argmax(self.lidar_readings)
            return desired_action

        # Introduce some randomness in the agent behavior
        if chance > self.epsilon:
            self.speed = 3
            return random.choice(self.lidar.clear_angles)

        for action in self.lidar.clear_angles:
            dx, dy, _ = self.compute_distance(action)
            new_position = self.position + np.array([dx, dy])
            distance = np.linalg.norm(new_position - self.current_target)
            if distance < min_distance_to_goal:
                min_distance_to_goal = distance
                desired_action = action

        return desired_action

