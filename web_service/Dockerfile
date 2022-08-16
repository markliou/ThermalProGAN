FROM tensorflow/tensorflow:2.5.0
MAINTAINER markliou <yifanliou@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive

RUN pip3 install tensorflow-addons

RUN apt update -y;\
    pip3 install tqdm;\
    pip3 install dill;\
    #pip3 install ktext;\
    pip3 install h5py;\
    pip3 install gunicorn;\
    pip3 install pandas;\
    pip3 install grpcio;\
    pip3 install Flask;\
    pip3 install biopython

RUN pip3 install Bio Flask

RUN mkdir /workspace
WORKDIR /workspace

# flask
EXPOSE 5000


CMD ["bash"]
