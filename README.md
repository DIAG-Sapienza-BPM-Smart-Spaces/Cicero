# Cicero
Dockerized application for training of the Cicero generative language model for e-justice

## Setup
```sh
docker build . -t cicero
docker network create -d overlay --attachable <network_name>
docker run -it --gpus all --mount source=<volume_name>,target=/code --network=<network_name> cicero
```
