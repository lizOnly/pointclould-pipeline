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
        // std::cout << "x: " << x << " y: " << y << " z: " << z << " r: " << r << " g: " << g << " b: " << b << std::endl;
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
void Reconstruction::buildGroundTruthCloud(std::string folder_path) {

    // Load the point cloud data from the text file
    std::cout << "Loading point cloud data from " << folder_path << std::endl;

    // intensity is used to store the label
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr bound_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr exterior_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr interior_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

    size_t clutter_count = 0;

    for (const auto & entry : std::filesystem::directory_iterator(folder_path)) {
        
        std::cout << entry.path() << std::endl;
        std::ifstream file(entry.path());

        std::string file_name = entry.path().filename().string();
        std::string file_name_no_ext = file_name.substr(0, file_name.length() - 4);
        std::string file_name_no_ext_no_num = file_name_no_ext.substr(0, findUnderScore(file_name_no_ext));

        std::cout << "file_name: " << file_name_no_ext_no_num << std::endl;

        if (file_name_no_ext_no_num == "wall" || file_name_no_ext_no_num == "floor" || file_name_no_ext_no_num == "ceiling" || file_name_no_ext_no_num == "window" || file_name_no_ext_no_num == "column" || file_name_no_ext_no_num == "door" || file_name_no_ext_no_num == "beam" || file_name_no_ext_no_num == "board") {
            
            float x_ext, y_ext, z_ext;
            int r_ext, g_ext, b_ext;
            
            while (file >> x_ext >> y_ext >> z_ext >> r_ext >> g_ext >> b_ext)
            {   
                // pcl::PointXYZRGB point_ext;
                // point_ext.x = x_ext;
                // point_ext.y = y_ext;
                // point_ext.z = z_ext;
                // point_ext.r = r_ext;
                // point_ext.g = g_ext;
                // point_ext.b = b_ext;
                // exterior_cloud->points.push_back(point_ext);

                pcl::PointXYZI point;
                point.x = x_ext;
                point.y = y_ext;
                point.z = z_ext;

                auto item = ground_truth_map.find(file_name_no_ext_no_num);
                if (item != ground_truth_map.end()) {
                    point.intensity = item->second[0];
                } else {
                    point.intensity = -1;
                }
                cloud->points.push_back(point);

                pcl::PointXYZI point_bound;
                point_bound.x = x_ext;
                point_bound.y = y_ext;
                point_bound.z = z_ext;

                point_bound.intensity = 1.0;

                bound_cloud->points.push_back(point_bound);

            }

        } else {

            float x_int, y_int, z_int;
            int r_int, g_int, b_int;

            while (file >> x_int >> y_int >> z_int >> r_int >> g_int >> b_int) 
            {   
                clutter_count++;
                // pcl::PointXYZRGB point_int;
                // point_int.x = x_int;
                // point_int.y = y_int;
                // point_int.z = z_int;
                // point_int.r = r_int;
                // point_int.g = g_int;
                // point_int.b = b_int;
                // interior_cloud->points.push_back(point_int);

                pcl::PointXYZI point;
                point.x = x_int;
                point.y = y_int;
                point.z = z_int;
                
                auto item = ground_truth_map.find(file_name_no_ext_no_num);
                if (item != ground_truth_map.end()) {
                    point.intensity = item->second[0];
                } else {
                    point.intensity = -1;
                }
                cloud->points.push_back(point);

                pcl::PointXYZI point_bound;
                point_bound.x = x_int;
                point_bound.y = y_int;
                point_bound.z = z_int;

                point_bound.intensity = 0.0;

                bound_cloud->points.push_back(point_bound);
                
            }

        }

    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    pcl::io::savePCDFile("../files/gt.pcd", *cloud);

    std::cout << "Saved " << cloud->points.size() << " points" << std::endl;
    std::cout << "" << std::endl;

    bound_cloud->width = bound_cloud->points.size();
    bound_cloud->height = 1;

    pcl::io::savePCDFile("../files/bound_cloud.pcd", *bound_cloud);

    std::cout << "clutter count: " << clutter_count << std::endl;
    double clutter_ratio = (double)clutter_count / (double)cloud->width;
    std::cout << "clutter ratio: " << clutter_ratio << std::endl;
    std::cout << "" << std::endl;


}


void Reconstruction::pcd2ply(std::string path) {

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(path, *cloud);

    std::string output_path = path.substr(0, path.length() - 4) + ".ply";
    pcl::io::savePLYFileASCII(output_path, *cloud);

}


void Reconstruction::ply2pcd(std::string path) {

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPLYFile<pcl::PointXYZRGB>(path, *cloud);

    std::string output_path = path.substr(0, path.length() - 4) + ".pcd";
    pcl::io::savePCDFileASCII(output_path, *cloud);

}

void Reconstruction::createGT(std::string path) {

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile<pcl::PointXYZI>(path, *cloud);

    std::string output_path = path.substr(0, path.length() - 4) + "_gt.pcd";
    pcl::io::savePCDFileASCII(output_path, *cloud);

}
