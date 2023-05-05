# Cicero
Dockerized application for training of the Cicero generative language model for e-justice

## Preprocessing
In this repository it is included the source file used to preprocess the judicial sentences dataset, that will not be shared. In order to run the training it is necessary to provide a dataset in csv format.

## Setup for Training
```sh
docker build . -t cicero
docker network create -d overlay --attachable <network_name>
docker run -it --gpus all --mount source=<volume_name>,target=/code --network=<network_name> cicero
```
