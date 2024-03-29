# Cicero
Dockerized application for training using multi-gpu parallelism of the Cicero generative language model for e-justice

## GUI Demo
It is possible to test the model sentence generation using the file in the respective folder.

## Preprocessing
In this repository it is included the source file used to preprocess the judicial sentences dataset, that will not be shared. In order to run the training it is necessary to provide a dataset in csv format.

## Setup for Training
The Dockerized envinronment supports training via NVIDIA-CUDA only (tested with CUDA 11.7). It is also natively included support for multi-gpu. In order to recreate our experiments run:

```sh
docker build . -t cicero
docker network create -d overlay --attachable <network_name>
docker run -it --gpus all --mount source=<volume_name>,target=/code --network=<network_name> cicero
```
