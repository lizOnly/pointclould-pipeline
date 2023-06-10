#include <iostream>
#include <thread>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "../headers/visualizer.h"


visualizer::visualizer()
{
    // empty constructor
}

visualizer::~visualizer()
{
    // empty destructor
}


void visualizer::visualizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer.addCoordinateSystem(1.0);
    viewer.initCameraParameters();
    viewer.setCameraPosition(
        0.0, 0.0, 10.0,  // Camera position (x,y,z)
        0.0, 5.0, 1.0,  // Viewpoint position
        0.0, -1.0, 0.0  // Camera "up" direction (in world coordinates)
    );
    
    while (!viewer.wasStopped()) {
        viewer.spinOnce(500);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


void visualizer::pointPickingEventOccurred(const pcl::visualization::PointPickingEvent& event, void* viewer_void)
{
    std::cout << "[INOF] Point picking event occurred." << std::endl;
    float x, y, z;
    if (event.getPointIndex() == -1) {
        return;
    }
    event.getPoint(x, y, z);
    std::cout << "[INFO] Point coordinate ( " << x << ", " << y << ", " << z << ")" << std::endl;
}


void visualizer::keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* viewer_void)
{
    std::cout << "[INFO] Keyboard event occurred." << std::endl;
    if (event.getKeySym() == "r" && event.keyDown()) {
        std::cout << "[INFO] r was pressed." << std::endl;
    }
}

// estimate plane based on a vector of points
pcl::PointCloud<pcl::PointXYZ>::Ptr visualizer::estimatePolygon(std::vector<pcl::PointXYZ> points)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto& point : points)
        cloud->points.push_back(point);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0) {
        PCL_ERROR("Could not estimate a planar model for the given dataset.");
        return nullptr;
    }

    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud);
    proj.setModelCoefficients(coefficients);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);
    proj.filter(*cloud_projected);

    pcl::PointCloud<pcl::PointXYZ>::Ptr polygon(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto& point : cloud_projected->points)
        polygon->points.push_back(point);

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->addPolygon<pcl::PointXYZ>(polygon, "polygon");

    return polygon;
}

