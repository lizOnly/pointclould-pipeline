#define _USE_MATH_DEFINES

#include <cmath>
#include <vector>
#include <iostream>
#include <string>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/boundary.h>

#include "../headers/property.h"
#include "../headers/reconstruction.h"

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

double Property::computeDensity(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

    std::cout << "Computing density... " << std::endl;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    double radius = 0.01; // search radius
    double volume = 4.0 / 3.0 * M_PI * radius * radius * radius;

    std::vector<double> densities(cloud->size(), 0.0);

    for (int i = 0; i < cloud->size(); ++i)
    {
        std::vector<int> indices;
        std::vector<float> distances;
        kdtree.radiusSearch(i, radius, indices, distances);

        double density = indices.size() / volume;
        densities[i] = density;
    }

    double average_density = 0.0;

    for (int i = 0; i < densities.size(); ++i)
    {
        average_density += densities[i];
    }

    average_density /= densities.size();

    std::cout << " Average density is " << average_density << std::endl;

    return average_density;

}

/*
    calculate average density of a point cloud using Gaussian kernel density estimation
    input: point cloud
    output: point cloud with density as a new channel
*/
double Property::computeDensityGaussian(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{   
    std::cout << "Computing gaussian density... " << std::endl;
    // Create kd-tree object
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    // Parameters for density estimation
    const double radius = 0.01; // search radius
    const double sigma2 = 0.01; // bandwidth parameter for Gaussian kernel 

    // Estimate density for each point
    std::vector<double> densities(cloud->size(), 0.0);
    for (int i = 0; i < cloud->size(); ++i)
    {
        std::vector<int> indices;
        std::vector<float> distances;
        kdtree.radiusSearch(i, radius, indices, distances); // citation needed

        double density = 0.0;
        for (int j = 0; j < indices.size(); ++j)
        {
            double distance2 = distances[j] * distances[j];
            double kernel = exp(-distance2 / (2.0 * sigma2));
            density += kernel;
            
        }
        densities[i] = density;
    }

    double average_density = 0.0;
    
    for (int i = 0; i < densities.size(); ++i)
    {
        average_density += densities[i];
    }

    average_density /= densities.size();

    std::cout << " Average gaussian density is " << average_density << std::endl;
    
    return average_density;

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


void Property::boundaryEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double angle_threshold, std::string input_path)
{   
    std::cout << "Estimating boundary... "
              << std::endl;
    // compute normals
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normal_estimation.setSearchMethod(tree);
    normal_estimation.setRadiusSearch(0.01);
    normal_estimation.compute(*normals);

    std::cout << "Normal estimation finished. "
              << std::endl;

    pcl::PointCloud<pcl::Boundary>::Ptr boundaries(new pcl::PointCloud<pcl::Boundary>);
    // boundary estimation
    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimation;
    boundary_estimation.setInputCloud(cloud);
    boundary_estimation.setInputNormals(normals);
    boundary_estimation.setRadiusSearch(0.035);
    boundary_estimation.setSearchMethod(tree);
    double angle_threshold_rad = angle_threshold * (M_PI / 180.0);
    boundary_estimation.setAngleThreshold(angle_threshold_rad);
    boundary_estimation.compute(*boundaries);

    std::cout << "Boundary estimation finished."
              << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (int i = 0; i < cloud->size(); ++i)
    {
        pcl::PointXYZRGB colored_point;
        colored_point.x = cloud->points[i].x;
        colored_point.y = cloud->points[i].y;
        colored_point.z = cloud->points[i].z;

        if (boundaries->points[i].boundary_point)
        {
            // yellow boundary points
            colored_point.r = 255;
            colored_point.g = 255;
            colored_point.b = 0;
        }
        else
        {
            // blue points
            colored_point.r = 0;
            colored_point.g = 0;
            colored_point.b = 255;
        }

        colored_cloud->points.push_back(colored_point);
    }
    std::cout << "Boundary colored. "
              << std::endl;

    colored_cloud->width = colored_cloud->points.size();
    colored_cloud->height = 1;
    colored_cloud->is_dense = true;

    std::string output_path = input_path;
    std::size_t pos = output_path.find("input");
    if(pos != std::string::npos) {
        output_path.replace(pos, 5, "output");
    }

    output_path.replace(output_path.end()-4, output_path.end(), "_boundary.pcd");

    pcl::io::savePCDFile<pcl::PointXYZRGB>(output_path, *colored_cloud);
    std::cout << "Colored point cloud saved as "
              << output_path 
              << std::endl;
}

