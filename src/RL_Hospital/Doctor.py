# DoctorAgent.py
import numpy as np
from .constants import NEARBY_ZONE, LARGE_VALUE, FRACTION
from .lidar import Lidar
import random

class FSMDoctorState:
    START_IN_MEDICAL_CLINIC  = 'StartInMedicalClinic'
    LEAVE_MEDICAL_CLINIC = 'LeaveMedicalClinic'
    MOVING_TO_MEDICATION_ROOM = 'MovingToMedRoom'
    REACHED_MEDICATION_ROOM = 'ReachedMedRoom'
    LEAVING_MEDICATION_ROOM = 'LeavingMedRoom'
    MOVING_TO_NURSE_STATION = 'MovingToNurseStation'
    REACHED_NURSE_STATION = 'ReachedNurseStation'
    LEAVING_NURSE_STATION = 'LeavingNurseStation'
    MOVING_TO_CLINIC = 'MovingToClinic'
    REACHED_CLINIC = 'ReachedClinic'
    CONSULTING_PATIENTS = 'WaitingPatient'
    COMPLETED = 'Completed'

class DoctorAgent:
    def __init__(self, poi: dict, walls, speed=3.0, max_lidar_range=3.0):

        self.poi = poi
        self.current_target = None
        self.state = FSMDoctorState.MOVING_TO_MEDICATION_ROOM
        self.position = None # Starting position
        self.speed = speed  # Units per step
        self.facing = 0.0  # Initial facing direction in radians
        self.walls = walls

        self.epsilon = 0.8

        self.consulting = 10

        # Initialize Lidar with max_range equal to speed
        self.lidar = Lidar(walls=self.walls, max_range=50, num_rays=80)

    def reset(self):
        """
        Resets the agent to its initial state.
        """
        self.position = self.poi['clinic']['position']
        self.state = FSMDoctorState.START_IN_MEDICAL_CLINIC
        self.current_target = self.poi['clinic']['door']
        self.facing = 0.0
        self.consulting = 10

    def update(self, action):
        """
        Updates the agent's state based on the current state and action.

        :param action: Action   taken by the agent (relative angle in radians).
        """
        match self.state:

            case FSMDoctorState.START_IN_MEDICAL_CLINIC:
                self.current_target = self.poi['clinic']['door']
                self.state = FSMDoctorState.LEAVE_MEDICAL_CLINIC

            case FSMDoctorState.LEAVE_MEDICAL_CLINIC:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['med_station']['door']
                    self.state = FSMDoctorState.MOVING_TO_MEDICATION_ROOM

            case FSMDoctorState.MOVING_TO_MEDICATION_ROOM:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['med_station']['position']
                    self.state = FSMDoctorState.REACHED_MEDICATION_ROOM

            case FSMDoctorState.REACHED_MEDICATION_ROOM:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['med_station']['door']
                    self.state = FSMDoctorState.LEAVING_MEDICATION_ROOM

            case FSMDoctorState.LEAVING_MEDICATION_ROOM:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['nurse_station']['door']
                    self.state = FSMDoctorState.MOVING_TO_NURSE_STATION

            case FSMDoctorState.MOVING_TO_NURSE_STATION:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['nurse_station']['position']
                    self.state = FSMDoctorState.REACHED_NURSE_STATION

            case FSMDoctorState.REACHED_NURSE_STATION:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                 self.speed = distance_to_target
                 self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                 self.current_target = self.poi['nurse_station']['door']
                 self.state = FSMDoctorState.LEAVING_NURSE_STATION

            case FSMDoctorState.LEAVING_NURSE_STATION:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['medical_clinic']['door']
                    self.state = FSMDoctorState.MOVING_TO_CLINIC

            case FSMDoctorState.MOVING_TO_CLINIC:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.current_target = self.poi['medical_clinic']['position']
                    self.state = FSMDoctorState.REACHED_CLINIC

            case FSMDoctorState.REACHED_CLINIC:
                distance_to_target = np.linalg.norm(self.position - self.current_target)
                if distance_to_target > NEARBY_ZONE:
                    self.speed = distance_to_target
                    self.step(action)
                elif distance_to_target <= NEARBY_ZONE:
                    self.state = FSMDoctorState.CONSULTING_PATIENTS

            case FSMDoctorState.CONSULTING_PATIENTS:
                if self.consulting > 0:
                    self.consulting -= 1
                else:
                    self.state = FSMDoctorState.COMPLETED

            case FSMDoctorState.COMPLETED:
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
            if self.lidar.check_collision(self.position, movement_angle,self.speed):
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