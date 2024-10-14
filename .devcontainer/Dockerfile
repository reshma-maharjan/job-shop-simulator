# Base image
FROM ubuntu:24.04

# Prevent interactive prompt
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    git \
    curl \
    zip \
    unzip \
    tar \
    autoconf \
    autoconf-archive \
    libtool \
    libxmu-dev \
    libxi-dev \
    libgl-dev \
    libglu1-mesa-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    xorg-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Create a non-root user
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME

# Run creating venv
RUN python3 -m venv /home/$USERNAME/.venv

# activate venv
ENV PATH="/home/$USERNAME/.venv/bin:$PATH"

# Install nanobind
RUN pip3 install --upgrade pip && \
    pip3 install nanobind==1.9.2

# Install latest cmake
WORKDIR /home/$USERNAME
COPY jsp /home/$USERNAME/jsp/jsp
COPY setup.py /home/$USERNAME/jsp
COPY README.md /home/$USERNAME/jsp
COPY LICENSE /home/$USERNAME/jsp
COPY pyproject.toml /home/$USERNAME/jsp
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME/jsp

# Install latest cmake

ADD https://cmake.org/files/v3.30/cmake-3.30.0-linux-x86_64.sh /cmake-3.30.0-linux-x86_64.sh
RUN mkdir /opt/cmake && \
    sh /cmake-3.30.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    rm /cmake-3.30.0-linux-x86_64.sh

USER $USERNAME

RUN mkdir -p /home/$USERNAME/jsp/jsp
RUN rm -rf /home/$USERNAME/jsp/jsp/.vcpkg && git clone https://github.com/microsoft/vcpkg.git /home/$USERNAME/jsp/jsp/.vcpkg && cd /home/$USERNAME/jsp/jsp/.vcpkg && ./bootstrap-vcpkg.sh


RUN cd /home/$USERNAME/jsp && pip3 install . -vvv

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]
