#!/bin/bash
mkdir -p "$HOME/app"
cp -rf ../test/. "$HOME/app/test"
cp -rf ../inference/. "$HOME/app/inference"

sudo cp init_drowsiness.sh /usr/local/bin/
sudo cp .env /usr/local/etc/
sudo chmod +x /usr/local/bin/*.sh

cp drowsiness.desktop "$HOME/.config/autostart/"