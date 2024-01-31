# (1)
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# (2)
WORKDIR /code

# (3)
COPY ./requirements.txt /code/requirements.txt

# (4)
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y python3-pip git

RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

# (5)
COPY ./train/ /code/

# (6)
CMD python3 main.py 