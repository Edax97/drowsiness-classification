#!/bin/bash

TEST_PIN=2
while (()); do
  sleep 1
  inference_state=$(raspi-gpio get "$TEST_PIN")
  if [[ inference_state -eq "1" ]]; then
    systemctl --user start test.service
  else
    systemctl --user stop test.service
  fi;
done