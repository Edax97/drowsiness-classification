#!/bin/bash

sudo cp start_drowsiness.sh ../test/start_test.sh /usr/local/bin/
sudo cp .env /usr/local/etc/

sudo cp drowsiness.service test.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable drowsiness.service
systemctl --user restart drowsiness.service