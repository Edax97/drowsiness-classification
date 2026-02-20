#!/bin/bash
mkdir -p "$HOME/app"
sudo cp -r ../test "$HOME/app/test"
sudo cp -r ../inference "$HOME/app/inference"

sudo cp init_drowsiness.sh init_test.sh /usr/local/bin/
sudo cp .env /usr/local/etc/

sudo cp test_drowsy.service "$HOME/.config/systemd/user/"
cp drowsiness.desktop "$HOME/.config/autostart/"