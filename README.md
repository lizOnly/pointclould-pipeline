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
make -j4 # denpending on how many cores your cpu has

# below are examples about how to actually use this programm

# if only input path specified, the cloud will be recentered to origin of coordinate system
./pcd_pipeline ( -i=cloud.pcd | --input_path==cloud.pcd )

# 200 rays downsample clouds
./pcd_pipeline -i=cloud.pcd ( -rs=200 | --raysample==200 )

# compute occlusion level
./pcd_pipeline -i=cloud.pcd ( -o | --occlusion )

# reconstruct point cloud from .txt file
./pcd_pipeline ( -rc=text.txt | --reconstruct==text.txt )

# rotate cloud along x-axis 90 degress clockwise
./pcd_pipeline ( -rt | --rotate )

```

