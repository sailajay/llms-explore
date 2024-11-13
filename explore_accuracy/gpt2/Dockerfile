FROM ubuntu:22.04
  RUN apt-get update -y
  RUN apt-get upgrade -y
  RUN apt-get install -y curl
  RUN apt install -y python3-pip
  RUN apt install -y vim
RUN mkdir mlperf
WORKDIR /mlperf

# Install necessary packages
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean
