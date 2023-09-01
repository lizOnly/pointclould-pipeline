# Point Cloud Processing Pipeline

### Dev Environment
- WSL-Ubuntu_20.04
- gcc=7.5.0

### Install Dependency
- `sudo apt install libpcl-dev`
- `sudo apt install libjsoncpp-dev `
- `sudo apt install libwebsocketpp-dev`

###  Run program
```
# configuration
cd pcd_pipeline
mkdir build
cd build
cmake ..
make -j 6

# below are examples about how to actually use this programm

# running program as a backend service, this will be the main usage of this project
./pcd_pipeline -b

# compute occlusion level, you have to adjust parameters in config.json first

./pcd_pipeline -moc # mesh visible area based occlusion level


```
