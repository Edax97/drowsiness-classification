#!/bin/bash
mkdir -p "$HOME/app"
cp -r ../test "$HOME/app/test"
cp -r ../inference "$HOME/app/inference"

sudo cp init_drowsiness.sh init_test.sh /usr/local/bin/
sudo chmod +x /user/local/bin/*
sudo cp .env /usr/local/etc/

sudo chmod 644 test_drowsy.service drowsiness.desktop

sudo cp test_drowsy.service "$HOME/.config/systemd/user/"
cp drowsiness.desktop "$HOME/.config/autostart/"