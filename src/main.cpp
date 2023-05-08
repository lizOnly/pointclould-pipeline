#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

#include "../headers/reconstruction.h"
#include "../headers/evaluation.h"
#include "../headers/property.h"


int main(int argc, char *argv[])
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("files/input_cloud/segmented_1.pcd", *cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return (-1);
    }

    std::cout << "Loaded "
              << std::endl;
    
    Property prop;
    Reconstruction recon;
    
    // recon.poissonReconstruction(cloud);
    // prop.calculateDensity(cloud);
    // prop.calculateLocalPointNeighborhood(cloud);
    prop.boundaryEstimation(cloud, 110);


    return 0;
}
