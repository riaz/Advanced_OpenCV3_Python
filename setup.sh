#docker build -no-cache -t aop .

docker run -it --rm -p 8888:8888  -v `pwd`:/src aop /bin/bash

#docker run -it --rm -p 8888:8888 -v /dev/video0:/dev/video0  -v /tmp/.X11-unix:/tmp/.X11-unix:ro  -v `pwd`:/src aop /bin/bash 
