#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

#include "../headers/reconstruction.h"
#include "../headers/evaluation.h"
#include "../headers/property.h"
#include "../headers/helper.h"


int main(int argc, char *argv[])
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    std::string input_path = "../files/input/ROOM1_seg.pcd";

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_path, *cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return (-1);
    }

    std::cout << "Loaded "
              << std::endl;
    
    Property prop;
    Reconstruction recon;
    Helper helper;
    // recon.poissonReconstruction(cloud);
    // prop.calculateDensity(cloud);
    // prop.calculateLocalPointNeighborhood(cloud);
    prop.boundaryEstimation(cloud, 90, input_path);
    // helper.voxelizePointCloud(cloud);

    return 0;
}
