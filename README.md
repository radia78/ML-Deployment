# Object Segmentation
Imported PyTorch model into C++ and perform inference.

### Prerequisites
- Install PyTorch
- Install Libtorch
- Install Torchvision C++

### Building the application
```
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ../src
cmake --build . --config Release
```
