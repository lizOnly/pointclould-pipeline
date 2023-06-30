#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>

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

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Parsing arguments ... " << std::endl;

    std::map<std::string, std::string> args_map;
    args_map["-i="] = "--input==";
    args_map["-r="] = "--raysample==";
    args_map["-o"] = "--occlusion";

    int numRaySamples = 20000; // default value
    std::string file_name = ""; // without extension
    std::string input_path = "";
    std::string polygonDataPath = "";

    Property prop;
    Reconstruction recon;
    Helper helper;
    Validation validation;

    if (argc < 2) {
        std::cout << "You have to provide two arguments" << std::endl;
        return 0;
    }

    // first argument is always the input file name
    std::string arg1 = argv[1];
    if (arg1.substr(0, 3) == "-i=") {
        file_name = arg1.substr(3, arg1.length());
    } else if (arg1.substr(0, 9) == "--input==") {
        file_name = arg1.substr(9, arg1.length());
    } else {
        std::cout << "You have to provide an input file name" << std::endl;
        return 0;
    }
    input_path = "../files/input/" + file_name;
    polygonDataPath = "../files/input/" + file_name.substr(0, file_name.length() - 4) + "_centered-polygon.txt";
    std::cout << "inputPath: " << input_path << std::endl;
    std::cout << "polygonDataPath: " << polygonDataPath << std::endl;
   
    // load cloud from file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_path, *cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return (-1);
    }

    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cloud, minPt, maxPt);
    pcl::PointXYZ center;
    center.x = (maxPt.x + minPt.x) / 2; center.y = (maxPt.y + minPt.y) / 2; center.z = (maxPt.z + minPt.z) / 2;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);          
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(input_path, *colored_cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return (-1);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr centered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr centered_colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    centered_cloud = helper.centerCloud(cloud, minPt, maxPt);
    centered_colored_cloud = helper.centerColoredCloud(colored_cloud, minPt, maxPt, file_name);
    
    // calculate min and max point of the centered cloud 
    pcl::getMinMax3D(*centered_cloud, minPt, maxPt);

    // parse arguments related to functionality
    for (int i = 2; i < argc; i++) {
        std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
        std::string argi = argv[i];
        //  use ray to downsample the cloud
        if (argi.substr(0, 3) == "-rs=" || argi.substr(0, 13) == "--raysample==") {

            numRaySamples = std::stoi(argi.substr(13, argi.length()));
            std::cout << "numRaySamples: " << numRaySamples << std::endl;
            validation.raySampledCloud(0.1, 0.05, 1, numRaySamples, minPt, maxPt, centered_cloud, centered_colored_cloud);
        
        // compute occlusionlevel
        } else if (argi == "-o" || argi == "--occlusion"){
            
            std::vector<std::vector<pcl::PointXYZ>> polygons = helper.parsePolygonData(polygonDataPath);
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds;
            std::vector<pcl::ModelCoefficients::Ptr> allCoefficients;

            for (int i = 0; i < polygons.size(); i++) {
                pcl::ModelCoefficients::Ptr coefficients = helper.computePlaneCoefficients(polygons[i]);
                allCoefficients.push_back(coefficients);
                pcl::PointCloud<pcl::PointXYZ>::Ptr polygon = helper.estimatePolygon(polygons[i], coefficients);
                polygonClouds.push_back(polygon);
            }

            double occlusionLevel = helper.rayBasedOcclusionLevel(minPt, maxPt, centered_cloud, polygonClouds, allCoefficients);
        
        } 
    }

    // int color[3] = {188, 189, 34};
    // helper.removePointsInSpecificColor(colored_cloud, color);
 
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << " Time taken by this run: " << duration.count() << " seconds" << std::endl;

    return 0;
}
