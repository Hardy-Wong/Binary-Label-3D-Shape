FROM python:3.8

WORKDIR /work
COPY requirements.txt /work/
# # pipの他に最低限使いそうなもの libgl1-mesa-devはopencvで必要
RUN apt-get update && apt-get install -y git wget libgl1-mesa-dev libstdc++6
RUN pip install -r /work/requirements.txt