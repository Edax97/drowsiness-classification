#!/bin/bash
source /usr/local/etc/.env
export DROWSY_DET_FILE, AWAKE_DET_FILE

cd /home/pi/app/test || exit
./start_test.sh
