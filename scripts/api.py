import subprocess

from fastapi import FastAPI

app = FastAPI()

TRAINING_FILE = "scripts/ensemble.py"


@app.post("/train")
def train():
    subprocess.Popen(["python", TRAINING_FILE])
    return {"status": "training started"}
