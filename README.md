# Point Cloud Processing Pipeline

### Dev Environment
- WSL-Ubuntu_18.04
- gcc=7.5.0
### Install Dependency
- `sudo apt install libpcl-dev`
###  Run program
```
# configuration
cd pcd_pipeline
mkdir build
cd build
cmake ..
make

# the second arguments should always be file name
# if no other arguments were specified, this will recenter cloud to (0, 0, 0)
./pcd_pipeline -i=cloud.pcd 

# 200 rays downsample clouds
./pcd_pipeline -i=cloud.pcd -rs=200

# compute occlusion level
./pcd_pipeline -i=cloud.pcd -o

```

