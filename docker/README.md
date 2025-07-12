## Use the container

```Shell
cd docker/

if Jetson Xavier AGX:

# Build:

sudo docker build -f Dockerfile.xavier -t yolact_edge_image .

# Launch (with GPUs):
./start.sh /path/to/yolact_edge /path/to/datasets


Otherwise:
# Build:
docker build --build-arg USER_ID=$UID -t yolact_edge_image .

# Launch (with GPUs):
./start.sh /path/to/yolact_edge /path/to/datasets
```
