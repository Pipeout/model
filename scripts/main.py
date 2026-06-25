import os
import subprocess
PATH_DESIRED_MODEL = 'lgbm.py'
if __name__ == "__main__":
    p1 = subprocess.Popen(["python", PATH_DESIRED_MODEL])
    p1.wait()