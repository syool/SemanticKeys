# run docker container
# add --rm for instant container

sudo docker run -it \
     --gpus all \
     --name paperdrip \
     --hostname paperdrip \
     -v $HOME/Codes/PAPER:/root/PAPER:rw \
     lig/paper:latest
