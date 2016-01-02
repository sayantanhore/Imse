# This file contains all the project specific custom configurations. It is meant to imported by Django settings.py



# Configurations
import os

PROJECT_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = PROJECT_PATH + '/data/Data/'
MEDIA_PATH = PROJECT_PATH + '/data/media/'
FILE_ROOT_PATH = PROJECT_PATH

FEATURES = DATA_PATH + "cl25000.npy"

GP_PORT = "/ldata/Imse/port.txt"
GP_URL_INITIAL = "http://localhost:"

# Constants

IMAGENUM = 25000
EXPLORATION_RATE = 0.008
DISTANCE_METRIC = 'cityblock'
