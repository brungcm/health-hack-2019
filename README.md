# health-hack-2019
Hackathon 2019

Drowning is the leading cause of death among children aged 1 to 4 years in Brazil.

Some statistics:
- 10 people die from drowning in the US every day.
- 17 people die from drowning in Brazil every day.
- More than 360.00 deaths are caused by drowning worldwide.

We strongly believe that prevention is better than cure.
That said, we developed a machine learning solution to prevent and warn of possible drowning in real time using cameras inside and outside pools.

With 3 custom models, you can identify:
- risk position (deep)
- identify and count the time a face is underwater
- identify panic

Using all of the models together, we can create an assertive risk alert.

Solution viability:

Brazil
- 1.500.000 swimming pools (2º ranking place)
- 60.000 per year

EUA
- 9.000.000 swimming pools (1º ranking place)
- 170.000 per year

Estimative solution cost:
- Equipment: $ 600
- Installation: $ 200
- Services: $ 300

The application is technical and financial viable and can contribute in the prevention of accidents and save lives.

https://drive.google.com/drive/folders/1hdRBJ6AqmI_xImORScjT0pIswnNFOCYK?usp=sharing


## Install nvidia-docker


If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers

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

Run image
```bash
xhost +local:docker

export XSOCK=/tmp/.X11-unix \
export XAUTH=/tmp/.docker.xauth \
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run --runtime=nvidia -it --rm --device=/dev/video0:/dev/video0 --privileged -v ${PWD}:/tracker -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/CIT/rodrigofp/Projects/hackathon/dataset:/dataset --env QT_X11_NO_MITSHM=1 -v ${PWD}:/tracker  --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH tf_1_13_opencv_3.4 

xhost -local:docker

```

Extract frames
```bash
python -m apps.extract_frames --config-file ./config/tracker_config.json --video-path ./Hackathon_videos/VID_20190329_164527.mp4 --output-dir ./frames/normal/VID_20190329_164527
```