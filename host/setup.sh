#!/bin/bash

sudo cp start_drowsiness.sh ../test/start_test.sh /usr/local/bin/
sudo cp .env /usr/local/bin/

sudo cp test_drowsiness.service "$HOME/.config/systemd/user/"
cp drowsiness.desktop "$HOME/.config/autostart/"