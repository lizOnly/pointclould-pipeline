#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

#include "../headers/property.h"
/*
    a class named Property, which can be used to calculate properties of a point cloud, and store them as a new channel in the point cloud. 
    including density, distribution, etc.
*/


Property::Property()
{
    // empty constructor
}

Property::~Property()
{
    // empty destructor
}


void Property::calculateDensity(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // Create kd-tree object
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    // Parameters for density estimation
    const double radius = 0.1; // search radius
    const int k = 10; // number of nearest neighbors
    const double sigma2 = 0.01; // bandwidth parameter for Gaussian kernel 

    // Estimate density for each point
    std::vector<double> densities(cloud->size(), 0.0);
    for (int i = 0; i < cloud->size(); ++i)
    {
        std::vector<int> indices(k);
        std::vector<float> distances(k);
        kdtree.radiusSearch(i, radius, indices, distances); // citation needed

        double density = 0.0;
        for (int j = 0; j < indices.size(); ++j)
        {
            double distance2 = distances[j] * distances[j];
            double kernel = exp(-distance2 / (2.0 * sigma2));
            density += kernel;
            
        }
        densities[i] = density;
        std::cout << " Density calculated. " << std::endl;
    }

    // Save density as a new channel in point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_density(new pcl::PointCloud<pcl::PointXYZI>);
    cloud_with_density->points.resize(cloud->size());
    for (int i = 0; i < cloud->size(); ++i)
    {
        cloud_with_density->points[i].x = cloud->points[i].x;
        cloud_with_density->points[i].y = cloud->points[i].y;
        cloud_with_density->points[i].z = cloud->points[i].z;
        cloud_with_density->points[i].intensity = densities[i];
        std::cout << " Density saved. "<< std::endl;
    }

    cloud_with_density->width = cloud->width;
    cloud_with_density->height = cloud->height;
    cloud_with_density->is_dense = cloud->is_dense;
    
    pcl::io::savePCDFile<pcl::PointXYZI>("output_cloud/ICH_room_output.pcd", *cloud_with_density);
    std::cout << " Point cloud with density saved. " << std::endl;
}


void Property::calculateDistribution(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{

}


void Property::calculateLocalPointNeighborhood(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    const int k = 10; 

    
    std::vector<std::vector<int>> local_neighborhoods(cloud->size());

    for (int i = 0; i < cloud->size(); ++i)
    {
        std::vector<int> indices(k);
        std::vector<float> distances(k);

        kdtree.nearestKSearch(i, k, indices, distances);

        local_neighborhoods[i] = indices;
    }

    for (int i = 0; i < local_neighborhoods.size(); ++i)
    {
        std::cout << "Local neighborhood of point " << i << ": ";
        for (int j = 0; j < local_neighborhoods[i].size(); ++j)
        {
            std::cout << local_neighborhoods[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
