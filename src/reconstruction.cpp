#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/io/obj_io.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/io/vtk_io.h>

#include "../headers/reconstruction.h"
#include "../headers/evaluation.h"

Reconstruction::Reconstruction() {
    // empty constructor
}

Reconstruction::~Reconstruction() {
    // empty destructor
}

void Reconstruction::poissonReconstruction(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    normal_estimation.setSearchMethod(tree);
    normal_estimation.setRadiusSearch(0.05); 
    normal_estimation.compute(*normals);

    // concatenate points and normals
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    // poisson reconstruction
    pcl::Poisson<pcl::PointNormal> poisson;
    poisson.setInputCloud(cloud_with_normals);
    poisson.setDepth(10); 
    pcl::PolygonMesh mesh;
    poisson.reconstruct(mesh);

    pcl::io::saveOBJFile("files/output/mesh.obj", mesh);

    std::cout << "Saved " << mesh.polygons.size() << " polygons " << std::endl;
    // saveMeshAsOBJWithMTL(mesh, "mesh.obj", "mesh.mtl");
}



void Reconstruction::marchingCubesReconstruction(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    n.setInputCloud(cloud);
    n.setSearchMethod(tree);
    n.setKSearch(20);
    n.compute(*normals);

    // Concatenate the XYZ and normal fields*
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    // Create search tree*
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud(cloud_with_normals);

    // Initialize objects
    pcl::MarchingCubesHoppe<pcl::PointNormal> mc;
    pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh);

    // Set parameters
    mc.setIsoLevel(0.0);
    mc.setGridResolution(50, 50, 50);
    mc.setPercentageExtendGrid(0.02f);

    // Reconstruct
    mc.setInputCloud(cloud_with_normals);
    mc.setSearchMethod(tree2);
    mc.reconstruct(*triangles);

    // Save the mesh to a file
    pcl::io::savePLYFile("mesh.ply", *triangles);
    std::cout << "Saved " << triangles->polygons.size() << " triangles " << std::endl;
    pcl::io::saveOBJFile("mesh.obj", *triangles);

}


void Reconstruction::pointCloudReconstructionFromTxt(std::string path)
{
    // Load the point cloud data from the text file
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    std::ifstream file(path);
    std::cout << "Loading point cloud data from " << path << std::endl;
    float x, y, z;
    int r, g, b;
    while (file >> x >> y >> z >> r >> g >> b)
    {   
        std::cout << "x: " << x << " y: " << y << " z: " << z << " r: " << r << " g: " << g << " b: " << b << std::endl;
        pcl::PointXYZRGB point;
        point.x = x;
        point.y = y;
        point.z = z;
        point.r = r;
        point.g = g;
        point.b = b;
        cloud->points.push_back(point);
    }
    cloud->width = cloud->points.size();
    std::cout << "Loaded " << cloud->width << " points" << std::endl;
    cloud->height = 1;

    pcl::io::savePCDFile("../files/recon_cloud.pcd", *cloud);
}

/*
    * Find the index of the first underscore in the string
    * @param str: the string to be searched
    * @return: the index of the first underscore in the string
    *          if no underscore is found, return -1
*/
int Reconstruction::findUnderScore(std::string& str) {
    for (int i = 0; i < str.size(); ++i) {
        if (str[i] == '_') {
            return i;
        }
    }
    return -1;
}


// build ground truth point cloud from txt files
void Reconstruction::batchReconstructionFromTxt(std::string folder_path) {

    std::map<std::string, std::vector<int>> ground_truth_map;
    ground_truth_map["beam"] = {20};
    ground_truth_map["board"] = {21};
    ground_truth_map["bookcase"] = {9};
    ground_truth_map["ceiling"] = {1}; // same as floor
    ground_truth_map["chair"] = {4};
    ground_truth_map["clutter"] = {25};
    ground_truth_map["door"] = {7};
    ground_truth_map["floor"] = {1};
    ground_truth_map["sofa"] = {5};
    ground_truth_map["table"] = {6};
    ground_truth_map["wall"] = {0};



    // Load the point cloud data from the text file
    std::cout << "Loading point cloud data from " << folder_path << std::endl;

    // intensity is used to store the label
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);

    for (const auto & entry : std::filesystem::directory_iterator(folder_path)) {
        
        std::cout << entry.path() << std::endl;
        std::ifstream file(entry.path());

        std::string file_name = entry.path().filename().string();
        std::string file_name_no_ext = file_name.substr(0, file_name.length() - 4);
        std::string file_name_no_ext_no_num = file_name_no_ext.substr(0, findUnderScore(file_name_no_ext));

        std::cout << "file_name: " << file_name_no_ext_no_num << std::endl;

        float x, y, z;
        int r, g, b;
        while (file >> x >> y >> z >> r >> g >> b)
        {   
            pcl::PointXYZI point;
            point.x = x;
            point.y = y;
            point.z = z;
            auto item = ground_truth_map.find(file_name_no_ext_no_num);
            if (item != ground_truth_map.end()) {
                point.intensity = item->second[0];
            } else {
                point.intensity = 20;
            }
            cloud->points.push_back(point);
        }
    }

    cloud->width = cloud->points.size();
    std::cout << "Loaded " << cloud->width << " points" << std::endl;
    cloud->height = 1;

}


void Reconstruction::pcd2ply(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string file_name) {

    std::string path = "../files/" + file_name.substr(0, file_name.length() - 4) + ".ply";
    pcl::io::savePLYFileASCII(path, *cloud);

}


void Reconstruction::ply2pcd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string file_name) {

    std::string path = "../files/" + file_name.substr(0, file_name.length() - 4) + ".pcd";
    pcl::io::savePCDFileASCII(path, *cloud);

}

