#! /bin/bash

sudo docker compose down
sudo docker compose up --build -d
sudo docker ps -a
sudo docker compose logs -f --tail 500
