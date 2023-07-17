#include <iostream>
#include <chrono>
#include <string>
#include <vector>
// #include <unordered_map>

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
#include "../headers/scanner.h"


int main(int argc, char *argv[])
{   

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Parsing arguments ... " << std::endl;

    std::map<std::string, std::string> args_map;
    args_map["-i="] = "--input==";
    args_map["-rs="] = "--raysample==";
    args_map["-o"] = "--occlusion";
    args_map["-sr="] = "--searchradius==";
    args_map["-h"] = "--help";
    args_map["-rc="] = "--reconstruct";
    args_map["-s="] = "--semantic==";
    args_map["-e"] = "--evaluate"; // calculate the evaluation metrics, IoU, F1, etc.
    args_map["-c"] = "--center";
    args_map["-f"] = "--filter";
    args_map["-rt"] = "--rotate";  // rotate the cloud around the x-axis by 90 degrees clockwise
    args_map["-d"] = "--density"; // compute density of the cloud
    args_map["-t2ply"] = "--transfer2ply"; // transfer pcd file to ply file
    args_map["-t2pcd"] = "--transfer2pcd"; // transfer ply file to pcd file



    int num_ray_sample = 1000; // default value, ray downsampling cloud
    bool filter_cloud = false;
    double epsilon = 0.1;
    size_t num_points = 0;
    size_t num_sampled_points = 0;

    std::string file_name = "";
    std::string input_path = "";
    std::string polygon_path = "";
    std::string recon_path = "";

    Property prop;
    Reconstruction recon;
    Helper helper;
    Scanner scanner;

    std::string folder_path = "/mnt/c/Users/51932/Desktop/s3d/Area_1/conferenceRoom_1/Annotations/"; // default value, batch reconstruction
    // recon.batchReconstructionFromTxt(folder_path);

    if (argc < 2) {
        std::cout << "You have to provide at least two arguments" << std::endl;
        return 0;
    }

    // parse arguments related to imput file, or --help
    std::string arg1 = argv[1];

    if (arg1.substr(0, 3) == "-i=") {

        file_name = arg1.substr(3, arg1.length());

    } else if (arg1.substr(0, 9) == "--input==") {

        file_name = arg1.substr(9, arg1.length());

    } else if (arg1.substr(0, 4) == "-rc=" || arg1.substr(0, 15) == "--reconstruct==") {

        if (arg1.substr(0, 4) == "-rc=") {
            recon_path = "../files/" + arg1.substr(4, arg1.length());
        } else {
            recon_path = "../files/" + arg1.substr(14, arg1.length());
        }

        recon.pointCloudReconstructionFromTxt(recon_path);
        std::cout << "recon_path: " << recon_path << std::endl;

        return 0;

    } else if (arg1 == "-h" || arg1 == "--help") {

        for (auto it = args_map.begin(); it != args_map.end(); it++) {
            std::cout << it->first << " " << it->second << std::endl;
        }

        return 0;

    } else {

        std::cout << "You have to provide at least two arguments" << std::endl;
        return 0;
    }

    input_path = "../files/" + file_name;
    std::cout << "input_path: " << input_path << std::endl;
   
    // load cloud from file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_path, *cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return (-1);
    }

    double density = 1.0; 

    num_points = cloud->width * cloud->height;
    std::cout << "Loaded " << num_points << " data points from " << file_name << std::endl;

    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cloud, minPt, maxPt);

    pcl::PointXYZ center;
    center.x = (minPt.x + maxPt.x) / 2;
    center.y = (minPt.y + maxPt.y) / 2;
    center.z = (minPt.z + maxPt.z) / 2;

    std::cout << "center: " << center.x << " " << center.y << " " << center.z << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);          
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(input_path, *colored_cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return (-1);
    }

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr centered_colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // centered_colored_cloud = helper.centerColoredCloud(colored_cloud, minPt, maxPt, file_name);

    // parse arguments related to functionality
    for (int i = 2; i < argc; i++) {

        std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
        std::string argi = argv[i];

        //  use ray to downsample the cloud
        if (argi.substr(0, 4) == "-rs=" || argi.substr(0, 13) == "--raysample==") {

            if (argi.substr(0, 4) == "-rs=") {

                num_ray_sample = std::stoi(argi.substr(4, argi.length()));

            } else if (argi.substr(0, 13) == "--raysample==") {

                num_ray_sample = std::stoi(argi.substr(13, argi.length()));

            }
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr sampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

            std::cout << "num_ray_sample: " << num_ray_sample << std::endl;
            sampled_cloud = scanner.multi_sphere_scanner(0.05, 0.05, 0.1, num_ray_sample, minPt, maxPt, cloud, colored_cloud, density, file_name);
            
            num_sampled_points = sampled_cloud->width * sampled_cloud->height;
            std::cout << "num_sampled_points: " << num_sampled_points << std::endl;

            double sample_rate = (double) num_sampled_points / (double) num_points;
            std::cout << "sample_rate: " << sample_rate << std::endl;

        // compute occlusionlevel
        } else if (argi == "-sc") {
                
            scanner.scan_cloud(0.05, 0.1, minPt, maxPt, cloud, colored_cloud, file_name);

        } else if (argi == "-o" || argi == "--occlusion") {
            
            std::vector<std::vector<pcl::PointXYZ>> polygons = helper.parsePolygonData(polygon_path);
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds;
            std::vector<pcl::ModelCoefficients::Ptr> allCoefficients;

            for (int i = 0; i < polygons.size(); i++) {

                pcl::ModelCoefficients::Ptr coefficients = helper.computePlaneCoefficients(polygons[i]);
                allCoefficients.push_back(coefficients);

                pcl::PointCloud<pcl::PointXYZ>::Ptr polygon = helper.estimatePolygon(polygons[i], coefficients);
                polygonClouds.push_back(polygon);
            
            }

            double search_radius = 0.1;
            // if (density <= 5.0) {
            //     density = density / 5.0 + 1.0;
            // } else if (density > 5 && density <= 10) {
            //     density = density / 10.0 + 1.0;
            // } else {
            //         density = 2.0;
            // }

            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_density(new pcl::PointCloud<pcl::PointXYZI>);
            cloud_with_density = prop.computeDensityGaussian(cloud);

            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_median_distance(new pcl::PointCloud<pcl::PointXYZI>);
            cloud_with_median_distance = helper.computeMedianDistance(search_radius, cloud, cloud_with_density);

            double rayOcclusionLevel = helper.rayBasedOcclusionLevel(minPt, maxPt, density, search_radius, 
                                                                     cloud, cloud_with_median_distance,
                                                                     polygonClouds, allCoefficients);

        } else if (argi == "-d" || argi == "--density") {

            prop.computeDensityGaussian(cloud);

        } else if (argi == "-t2ply") {

            recon.pcd2ply(colored_cloud, file_name);
            
        } else if (argi == "-f" || argi == "--filter") {

            helper.voxelizePointCloud<pcl::PointXYZRGB>(colored_cloud, file_name);

        } else if (argi.substr(0, 3) == "-p=") {

            polygon_path = "../files/" + argi.substr(3, argi.length());
            std::cout << "polygon_path: " << polygon_path << std::endl;

        }  

    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << " Time taken by this run: " << duration.count() << " seconds" << std::endl;

    return 0;
}
