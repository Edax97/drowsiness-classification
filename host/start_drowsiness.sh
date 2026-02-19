#!/bin/bash
source .env
export DROWSY_DET_FILE, AWAKE_DET_FILE, DEVICE
cd /home/pi/drowsiness-classification/inference || exit
/home/pi/.venv/bin/python detect_drowsiness.py