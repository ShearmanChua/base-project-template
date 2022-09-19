# For more information, please refer to https://aka.ms/vscode-docker-python
# FROM nvcr.io/nvidia/pytorch:21.08-py3
FROM nvcr.io/nvidia/pytorch:21.09-py3

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

ADD build /build
WORKDIR /build
RUN make

ADD /src /src
RUN mkdir /data
RUN mkdir /models


WORKDIR /src
