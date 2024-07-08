### Requirements

- Docker
- Nvidia GPU

### Getting Started

1. make container (docker)
```shell
> docker run --rm -it --gpus all --shm-size=16G -p 8888:8888 -v %cd%:/workspace nvcr.io/nvidia/pytorch:20.12-py3 bash
> pip install pycuda==2024.1
```
or 
```
> docker build -t trt:latest .
> docker run --rm -it --gpus all --shm-size=16G -p 8888:8888 -v %cd%:/workspace trt bash
```

2. make onnx model
```shell
> python export.py
```

3. build tensorrt model
```shell
> trtexec --onnx=fcn-resnet101.onnx --workspace=64 --minShapes=input:1x3x256x256 --optShapes=input:4x3x256x256 --maxShapes=input:80x3x256x256 --buildOnly -saveEngine=fcn-resnet101.engine
# &&&& PASSED TensorRT.trtexec  ~
```

4. launch notebook
```shell
> jupyter notebook --ip 0.0.0.0 --allow-root --no-browser
```