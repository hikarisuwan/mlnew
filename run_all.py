import subprocess
import sys
import os

# this script is the master runner for the entire project
# it executes the analysis for both dataset 1 and dataset 2 sequentially
# usage: python run_all.py

if __name__ == "__main__":

    # this runs code for dataset 1
    # we use subprocess to run the script as if it were called from the command line
    # this ensures it runs in its own isolated process, preventing variable conflicts

    subprocess.run([sys.executable, os.path.join("dataset1", "dataset_1_code.py")])
    # once dataset 1 is finished, we immediately trigger the script for dataset 2
    subprocess.run([sys.executable, os.path.join("dataset2", "dataset_2_code.py")])
