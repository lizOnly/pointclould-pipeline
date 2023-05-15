#include <iostream>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/octree/octree.h>

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
