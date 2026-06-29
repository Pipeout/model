import subprocess

PATH_DESIRED_MODEL = "scripts/randomforest.py"
if __name__ == "__main__":
    p1 = subprocess.Popen(["python", PATH_DESIRED_MODEL])
    p1.wait()
