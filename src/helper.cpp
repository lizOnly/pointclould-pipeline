#include <iostream>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

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

