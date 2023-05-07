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
make    # run this whenever code has been changed

# then run at root dir
cd ..
./build/pcd_pipeline
```

