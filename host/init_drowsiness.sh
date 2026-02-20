#!/bin/bash
source /usr/local/etc/.env
export DROWSY_DET_FILE AWAKE_DET_FILE DEVICE
cd /home/pi/app/inference || exit
/home/pi/.venv/bin/python detect_drowsiness.py
