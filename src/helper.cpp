#include <iostream>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/octree/octree.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>

#include "../headers/helper.h"

Helper::Helper() {
    // empty constructor
}

Helper::~Helper() {
    // empty destructor
}

void Helper::identifyNormalHoles(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

}

void Helper::identifyOcclusionHoles(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

}

void Helper::voxelizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.05f, 0.05f, 0.05f);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.filter(*cloud_filtered);

    pcl::io::savePCDFileASCII("../files/output/voxelized_pcd.pcd", *cloud_filtered);
}


/*
    Voxel grid occlusion estimation
*/
void Helper::estimateOcclusion(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> voxelFilter;
    voxelFilter.setInputCloud(cloud);
    voxelFilter.setLeafSize(0.05f, 0.05f, 0.05f);

    voxelFilter.initializeVoxelGrid();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        int state;
        pcl::PointXYZ& point = cloud->points[i];
        Eigen::Vector3i grid_coordinates = voxelFilter.getGridCoordinates(point.x, point.y, point.z);
        
        voxelFilter.occlusionEstimation(state, grid_coordinates);

        pcl::PointXYZRGB colored_point;
        colored_point.x = cloud->points[i].x;
        colored_point.y = cloud->points[i].y;
        colored_point.z = cloud->points[i].z;

        if (state == 0) {
            colored_point.r = 255;
            colored_point.g = 255;
            colored_point.b = 0;
        } else {
            colored_point.r = 0;
            colored_point.g = 0;
            colored_point.b = 255;
        }
        colored_cloud->points.push_back(colored_point);

        // state will be 0 if point is occluded, 1 if it is visible
        std::cout << "Point " << i << " is " << (state == 1 ? "visible" : "occluded") << std::endl;
    }

    colored_cloud->width = colored_cloud->points.size();
    colored_cloud->height = 1;
    colored_cloud->is_dense = true;

    pcl::io::savePCDFileASCII("../files/output/oe_colored.pcd", *colored_cloud);

}

// bad results for now
void Helper::removeOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud);

    pcl::io::savePCDFileASCII("../files/output/outlier_filtered_cloud.pcd", *cloud);
}


void Helper::removePointsInSpecificColor(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int color[3]) {

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (const auto& point : cloud->points)
    {
        
        if (point.r != color[0] || point.g != color[1] || point.b != color[2])
        {
            std::cout << "Point " << point.x << " " << point.y << " " << point.z << " is not " << color[0] << " " << color[1] << " " << color[2] << std::endl;
            filtered_cloud->points.push_back(point);
        }
    }

    filtered_cloud->width = filtered_cloud->points.size();
    filtered_cloud->height = 1;
    filtered_cloud->is_dense = true;

    pcl::io::savePCDFileASCII("../files/output/specific_color_filtered_cloud.pcd", *filtered_cloud);

}


void Helper::regionGrowingSegmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normal_estimation.setSearchMethod(tree);
    normal_estimation.setRadiusSearch(0.02);
    normal_estimation.compute(*normals);

    pcl::IndicesPtr indices(new std::vector <int>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 1.0);
    pass.filter(*indices);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(50);
    reg.setMaxClusterSize(100000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(40);
    reg.setInputCloud(cloud);
    //reg.setIndices (indices);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(1.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(0.5);

    std::vector <pcl::PointIndices> clusters;
    reg.extract(clusters);

    std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;
    std::cout << "First cluster has " << clusters[0].indices.size() << " points." << std::endl;


    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
    pcl::io::savePCDFileASCII("../files/output/region_growing_segmentation.pcd", *colored_cloud);
}

