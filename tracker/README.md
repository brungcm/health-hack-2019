# Tracker app

### Build docker image
```bash
docker build -t health-hack-2019/tracker .
```

### Install nvidia-docker

If you have nvidia-docker 1.0 installed, we need to remove it and all existing GPU containers

```bash
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker
```

Add the package repositories

```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
```

Install nvidia-docker2 and reload the Docker daemon configuration

```bash
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```

Test nvidia-smi with the latest official CUDA image
```bash
docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```

Run image
```bash
docker run --runtime=nvidia -it --rm tensorflow/tensorflow:latest-gpu
```

### Running Tracker App
```bash
xhost +local:docker

export XSOCK=/tmp/.X11-unix \
export XAUTH=/tmp/.docker.xauth \
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run --runtime=nvidia -it --rm --device=/dev/video0:/dev/video0 --privileged -v ${PWD}:/app -v /tmp/.X11-unix:/tmp/.X11-unix --env QT_X11_NO_MITSHM=1 -v ${PWD}:/tracker  --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH health-hack-2019/tracker

xhost -local:docker

```
