#include <iostream>
#include <chrono>

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
    auto start = std::chrono::high_resolution_clock::now();

    std::string input_path = "../files/input/ICH_room.pcd";

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_path, *cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return (-1);
    }

    std::cout << "Loaded "
              << std::endl;

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);          

    // if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(input_path, *colored_cloud) == -1) {
    //     PCL_ERROR("Couldn't read file\n");
    //     return (-1);
    // }

    // std::cout << "Colored cloud loaded "
    //           << std::endl;
    
    Property prop;
    Reconstruction recon;
    Helper helper;

    // recon.poissonReconstruction(cloud);
    // recon.marchingCubesReconstruction(cloud);
    // std::string s3dTxtPath = "/mnt/c/Users/yufeng/Desktop/Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1/conferenceRoom_1.txt";
    // recon.pointCloudReconstructionFromTxt(s3dTxtPath);

    // prop.calculateDensity(cloud);
    // prop.calculateLocalPointNeighborhood(cloud);
    // prop.boundaryEstimation(cloud, 90, input_path);

    // helper.estimateOcclusion(cloud);
    // helper.voxelizePointCloud(cloud);

    // helper.removeOutliers(cloud);
    // int color[3] = {188, 189, 34};
    // helper.removePointsInSpecificColor(colored_cloud, color);
    double occlusionLevel = 0.0;
    occlusionLevel = helper.rayBasedOcclusionLevel(cloud, 2000, 0.05, 0.05);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

    std::cout << "Time taken by this run: " << duration.count() << " seconds" << std::endl;
    return 0;
}
