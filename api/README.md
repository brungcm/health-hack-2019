# Tracker app API

### Build docker image
```bash
docker build -t health-hack-2019/tracker-api .
```

### Run docker image
```bash
docker run -d -v ${PWD}/../:/data -v ${PWD}:/app -p 5000:5000 health-hack-2019/tracker-api
```