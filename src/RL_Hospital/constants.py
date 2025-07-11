"""
File holding constants to keep the code clean
"""
import sys
#Increase the view box of the simulation to bound of drawings + buffer for a better view
BUFFER_SIZE = 80
#Prefix to the correct directory containing the hospital layouts
HOSPITAL_PATH = "hospital-layouts"
# Distance from a target we need to be, to be considered at that location
NEARBY_ZONE = 10
# Amount of different angles to check
NUM_ACTIONS = 180
# Large dummy vale
LARGE_VALUE = sys.maxsize
# Fraction to control speed
FRACTION = 1.0
# Amount of iterations our RL Algorithms should perform
TRAINING_STEPS = 100_000

# Number of rays our lidar sees the world
RAYS = 20