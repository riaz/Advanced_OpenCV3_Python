#docker build -no-cache -t aop .

docker run -it --rm -p 8888:8888  -v `pwd`:/src aop /bin/bash 
