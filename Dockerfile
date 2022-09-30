FROM nvcr.io/nvidia/pytorch:21.05-py3

ARG UNAME=testuser
ARG UID=1000
ARG GID=1000

RUN apt-get upgrade
RUN apt-get -y update --fix-missing
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    sudo \
    net-tools \
    ssh-client \
    apt-utils \
    wget \
    libopenmpi-dev \
    libpng-dev \
    libpng++-dev \
    unzip \
    libopenexr-dev \
    unzip \
    ffmpeg \
    psmisc \
    rsync \
    tree \
    vim \
    tmux

RUN pip install --upgrade pip

################################################################################

RUN groupadd --gid $GID $UNAME-group
RUN useradd --uid $UID --gid $GID -m -c $UNAME -s /bin/bash $UNAME
USER $UNAME
WORKDIR /home/$UNAME
