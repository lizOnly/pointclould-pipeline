#include <iostream>
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

#include "../headers/reconstruction.h"

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
    normal_estimation.setRadiusSearch(0.03); 
    normal_estimation.compute(*normals);

    // concatenate points and normals
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    // poisson reconstruction
    pcl::Poisson<pcl::PointNormal> poisson;
    poisson.setInputCloud(cloud_with_normals);
    poisson.setDepth(9); 
    pcl::PolygonMesh mesh;
    poisson.reconstruct(mesh);

    pcl::io::saveOBJFile("mesh.obj", mesh);

    std::cout << "Saved " << mesh.polygons.size() << " polygons " << std::endl;
    // saveMeshAsOBJWithMTL(mesh, "mesh.obj", "mesh.mtl");
}

void Reconstruction::saveMeshAsOBJWithMTL(const pcl::PolygonMesh& mesh, const std::string& obj_filename, const std::string& mtl_filename)
{   
    
}

