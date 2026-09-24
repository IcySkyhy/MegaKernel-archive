import os

_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.join(_FILE_DIRECTORY, "../../")

EXPERIMENT_DIR = os.path.join(_BASE_DIR, "experiments")
DATA_DIR = os.path.join(_BASE_DIR, "data")
FIGURE_DIR = os.path.join(_BASE_DIR, "figures")
