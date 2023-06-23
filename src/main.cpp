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
#include "../headers/visualizer.h"


int main(int argc, char *argv[])
{   
    auto start = std::chrono::high_resolution_clock::now();

    std::string inputPath = "../files/input/centered_cloud.pcd";
    std::string polygonDataPath = "../files/input/centered_cloud-polygon.txt";

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(inputPath, *cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return (-1);
    }

    std::cout << "Pure cloud loaded "
              << std::endl;

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);          

    // if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(inputPath, *colored_cloud) == -1) {
    //     PCL_ERROR("Couldn't read file\n");
    //     return (-1);
    // }

    // std::cout << "Colored cloud loaded "
    //           << std::endl;
    
    Property prop;
    Reconstruction recon;
    Helper helper;

    // prop.calculateDensity(cloud);
    // prop.boundaryEstimation(cloud, 110, input_path);

    // helper.extractWalls(cloud);
    // helper.centerCloud(cloud, colored_cloud);
    // helper.voxelizePointCloud(cloud);
    // helper.regionGrowingSegmentation(cloud);
    std::vector<std::vector<pcl::PointXYZ>> polygons = helper.parsePolygonData(polygonDataPath);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds;
    std::vector<pcl::ModelCoefficients::Ptr> allCoefficients;
    for (int i = 0; i < polygons.size(); i++) {
        pcl::ModelCoefficients::Ptr coefficients = helper.computePlaneCoefficients(polygons[i]);
        allCoefficients.push_back(coefficients);
        pcl::PointCloud<pcl::PointXYZ>::Ptr polygon = helper.estimatePolygon(polygons[i], coefficients);
        polygonClouds.push_back(polygon);
    }

    // int color[3] = {188, 189, 34};
    // helper.removePointsInSpecificColor(colored_cloud, color);

    double occlusionLevel = 0.0;
    occlusionLevel = helper.rayBasedOcclusionLevel(cloud, polygonClouds, allCoefficients);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

    std::cout << " Time taken by this run: " << duration.count() << " seconds" << std::endl;


    return 0;
}
