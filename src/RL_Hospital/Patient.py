import numpy as np
from .constants import NEARBY_ZONE, LARGE_VALUE, FRACTION
import random
from .lidar import Lidar
from gym import spaces

class FSMPatientState:
    ARRIVE_AT_FLOOR = 'ArriveAtFloor'
    LEAVE_ELEVATOR_LOBBY = 'LeaveElevatorLobby'
    MOVING_TO_CLINIC = 'MovingToClinic'
    REACHED_CLINIC = 'ReachedClinic'
    CONSULTATION = 'Consultation'
    LEAVING_CLINIC = 'LeaveClinic'
    MOVING_TO_ELEVATOR = 'MovingToElevator'
    REACHED_ELEVATOR = 'ReachedElevator'
    COMPLETED = 'Completed'

class PatientAgent:
    def __init__(self, poi, walls):
        self.poi = poi
        self.state = FSMPatientState.ARRIVE_AT_FLOOR
        self.position = None # Starting position
        self.speed = 3.0  # Units per step
        self.facing = 0.0
        self.walls = walls
        self.current_target = None

        self.epsilon = 0.8

        self.consult_delay = 5

        self.lidar = Lidar(walls=self.walls, max_range=50, num_rays=80)

    def reset(self):
        self.state = FSMPatientState.ARRIVE_AT_FLOOR
        self.facing = 0.0
        self.current_target = self.poi['elevator_lobby']['door']
        self.position = self.poi['elevator_lobby']['position']

    def update(self, action):
        """
                Updates the agent's state based on the current state and action.

                :param action: Action taken by the agent (relative angle in radians).
                """
        match self.state:
            case FSMPatientState.ARRIVE_AT_FLOOR:
                self.current_target = self.poi['elevator_lobby']['door']
                self.state = FSMPatientState.LEAVE_ELEVATOR_LOBBY

            case FSMPatientState.LEAVE_ELEVATOR_LOBBY:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['medical_clinic']['door']
                    self.state = FSMPatientState.MOVING_TO_CLINIC

            case FSMPatientState.MOVING_TO_CLINIC:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['medical_clinic']['position']
                    self.state = FSMPatientState.REACHED_CLINIC

            case FSMPatientState.REACHED_CLINIC:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.state = FSMPatientState.CONSULTATION

            case FSMPatientState.CONSULTATION:
                if self.consult_delay > 0:
                    self.consult_delay -= 1
                elif self.consult_delay <= 0:
                    self.state = FSMPatientState.LEAVING_CLINIC
                    self.current_target = self.poi['medical_clinic']['door']

            case FSMPatientState.LEAVING_CLINIC:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['elevator_lobby']['door']
                    self.state = FSMPatientState.MOVING_TO_ELEVATOR

            case FSMPatientState.MOVING_TO_ELEVATOR:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['elevator_lobby']['position']
                    self.state = FSMPatientState.REACHED_ELEVATOR

            case FSMPatientState.REACHED_ELEVATOR:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.state = FSMPatientState.COMPLETED

            case FSMPatientState.COMPLETED:
                pass

            case _:
                pass

    def step(self, action):
        """
        Executes a movement step based on the action and Lidar readings.

        :param action: Action index determining the movement angle (0 to 359 degrees).
        """

        # Apply fraction to control the speed
        self.speed *= FRACTION

        relative_angle = np.clip(action, -np.pi / 2, np.pi / 2)  # Ensure within [-90°, 90°]
        # Compute global movement direction
        movement_angle = self.facing + relative_angle
        # Calculate movement vector

        while self.speed >= 1:
            if self.lidar.check_collision(self.position, movement_angle, self.speed):
                self.speed /= 2
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

    def compute_distance(self, action):
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
