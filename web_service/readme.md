ThermalProGAN web service
================================================================
This section tell how to build the server for translating the normal proteins to their thermal stable form.  
The website is buit using the FLASK module of python. 
The model can be download from the [pCloud](https://u.pcloud.link/publink/show?code=XZYQg0VZknUTIn8gAY0bb9v7lsoOmVMC9IK7). The model should put under *model* folder.

# creating service using docker
```shell
# build the docker image 
sudo docker build -t markliou/thermalproganweb .
```
But due to the [h5py issue](https://github.com/tensorflow/tensorflow/issues/22480), using this dockerfile would still have some problems. In order to save the time, it's recommand to use the precompile image:
```shell
sudo docker pull markliou/thermalproganweb
```
Initiating the web service:
```shell
sudo docker run -it -v `pwd`:/workspace -p 8088:8088 markliou/thermalproganweb python3 web_service.py
# http://xxx.xxx.xxx.xxx:8088
```
You can open the web page from the browser. 
