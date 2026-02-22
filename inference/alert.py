import os
import time

import playsound
from dotenv import load_dotenv
# import subprocess
from detection_header import SOMNOLENCIA_DET, ALERTA_DET

load_dotenv()
drowsy_det_file = os.getenv("DROWSY_DET_FILE")
awake_det_file = os.getenv("AWAKE_DET_FILE")

def alert_detection(detected: str):
    t = time.localtime()
    if detected == SOMNOLENCIA_DET:
        # _ =subprocess.Popen(["play", "alerts/alarm-4.wav"])
        playsound.playsound("alerts/alarm-4.wav", True)
        save_to_file(f"{t.tm_mday}:{t.tm_hour}:{t.tm_min} - {detected}", drowsy_det_file)
    if detected == ALERTA_DET:
        save_to_file(f"{t.tm_mday}:{t.tm_hour}:{t.tm_min} - {detected}", awake_det_file)

def save_to_file(line: str, filename: str):
    with open(filename, "a") as f:
        f.write(line)
