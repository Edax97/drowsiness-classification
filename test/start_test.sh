#!/bin/bash
START_SLEEP_MP3="./assets/close_them.mp3"
STOP_SLEEP_MP3="./assets/open_them.mp3"
INIT_TEST_MP3="./assets/test_start.mp3"
END_TEST_MP3="./assets/test_end.mp3"

declare -i TEST_LEN TEST_START_TIME MICROSLEEP_LEN
declare DROWSY_DET_FILE AWAKE_DET_FILE
source /usr/local/etc/.env
touch "$DROWSY_DET_FILE" "$AWAKE_DET_FILE"

TEST_FILE="$(date).test"
touch "$TEST_FILE"

TEST_LEN=120
MICROSLEEP_LEN=6
TEST_START_TIME=$(date +%s)

print_class(){
  EXPECTED=$1

  declare -i DROWSY_DET_TIME AWAKE_DET_TIME NOW
  DROWSY_DET_TIME=$(stat --printf="%Y" "$DROWSY_DET_FILE")
  AWAKE_DET_TIME=$(stat --printf="%Y" "$AWAKE_DET_FILE")
  NOW=$(date +%s)

  if (( DROWSY_DET_TIME + MICROSLEEP_LEN> NOW )); then
    echo "'$(date +%T)' expect: $EXPECTED, got: DROWSY" >> "$TEST_FILE"
  elif (( AWAKE_DET_TIME + MICROSLEEP_LEN  > NOW )); then
    echo "'$(date +%T)' expect: $EXPECTED, got: AWAKE" >> "$TEST_FILE"
  else
    echo "'$(date +%T)' expect: $EXPECTED, got: NO_FACE" >> "$TEST_FILE"
  fi;
}

sleep 3
play "$INIT_TEST_MP3" | aplay -q
echo "---START---" >> "$TEST_FILE"
while (( $(date +%s) < TEST_START_TIME + TEST_LEN )); do
  sleep "$MICROSLEEP_LEN"
  print_class "AWAKE"
  sleep 1
  play -q "$START_SLEEP_MP3" | aplay -q
  sleep "$MICROSLEEP_LEN"
  print_class "DROWSY"
  play -q "$STOP_SLEEP_MP3" | aplay -q
  sleep 1
done
sleep 2
echo "---END---" >> "$TEST_FILE"
play "$END_TEST_MP3" | aplay -q
