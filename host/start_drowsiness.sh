#!/bin/bash
cd /home/pi/drowsiness-classification/inference || exit

/home/pi/.venv/bin/python detect_drowsiness.py