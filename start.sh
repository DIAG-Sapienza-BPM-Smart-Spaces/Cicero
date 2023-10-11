#!/bin/bash

volume_name=$1

mkdir $volume_name

docker volume create --driver local --opt type=none --opt device=/raid/mecella/Cicero/$volume_name --opt o=bind $volume_name
docker build . -t cicero
docker run -it --gpus all --mount source=$volume_name,target=/code --network=cicero-netw cicero
