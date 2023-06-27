#include <iostream>
#include <chrono>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>

#include "../headers/reconstruction.h"
#include "../headers/evaluation.h"
#include "../headers/property.h"
#include "../headers/helper.h"
#include "../headers/visualizer.h"
#include "../headers/validation.h"


int main(int argc, char *argv[])
{   
    std::cout << argv[0] << std::endl;
    int numRaySamples = argv[1] ? atoi(argv[1]) : 1000;
    auto start = std::chrono::high_resolution_clock::now();

    std::string inputPath = "../files/input/raySampledCloud_" + std::to_string(numRaySamples) + ".pcd";
    std::string polygonDataPath = "../files/input/centered_cloud-polygon.txt";

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(inputPath, *cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return (-1);
    }

    std::cout << "Pure cloud loaded "
              << std::endl;

    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cloud, minPt, maxPt);
    std::cout << "Max x: " << maxPt.x << ", Max y: " << maxPt.y << ", Max z: " << maxPt.z << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);          

    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(inputPath, *colored_cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return (-1);
    }

    std::cout << "Colored cloud loaded "
              << std::endl;
    
    Property prop;
    Reconstruction recon;
    Helper helper;
    Validation validation;

    // prop.calculateDensity(cloud);
    // prop.boundaryEstimation(cloud, 110, input_path);

    // helper.extractWalls(cloud);
    // helper.centerCloud(cloud, colored_cloud);
    // helper.voxelizePointCloud(cloud);
    // helper.regionGrowingSegmentation(cloud);
    // validation.raySampledCloud(0.1, 0.05, 1, numRaySamples, minPt, maxPt, cloud, colored_cloud);

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
    occlusionLevel = helper.rayBasedOcclusionLevel(minPt, maxPt, cloud, polygonClouds, allCoefficients);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

    std::cout << " Time taken by this run: " << duration.count() << " seconds" << std::endl;


    return 0;
}
