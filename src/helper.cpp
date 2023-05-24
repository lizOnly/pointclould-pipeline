#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <chrono>

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
#include "../headers/BaseStruct.h"


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


pcl::PointCloud<pcl::Normal>::Ptr normalEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normal_estimation.setSearchMethod(tree);
    normal_estimation.setRadiusSearch(0.02);
    normal_estimation.compute(*normals);

    return normals;
}

std::vector<pcl::PointXYZ> Helper::getSphereLightSourceCenters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    std::vector<pcl::PointXYZ> centers;
    // calculate bounding box
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cloud, minPt, maxPt);
    std::cout << "Max x: " << maxPt.x << ", Max y: " << maxPt.y << ", Max z: " << maxPt.z << std::endl;
    std::cout << "Min x: " << minPt.x << ", Min y: " << minPt.y << ", Min z: " << minPt.z << std::endl;

    pcl::PointXYZ center;
    center.x = (maxPt.x + minPt.x) / 2;
    center.y = (maxPt.y + minPt.y) / 2;
    center.z = (maxPt.z + minPt.z) / 2;

    centers.push_back(center);

    // Points at the midpoints of the body diagonals (diagonal was divided by center point )
    pcl::PointXYZ midpoint1, midpoint2, midpoint3, midpoint4,
                  midpoint5, midpoint6, midpoint7, midpoint8;
    midpoint1.x = (center.x + minPt.x) / 2; midpoint1.y = (center.y + minPt.y) / 2; midpoint1.z = (center.z + minPt.z) / 2; 
    midpoint2.x = (center.x + minPt.x) / 2; midpoint2.y = (center.y + minPt.y) / 2; midpoint2.z = (center.z + maxPt.z) / 2;
    midpoint3.x = (center.x + minPt.x) / 2; midpoint3.y = (center.y + maxPt.y) / 2; midpoint3.z = (center.z + minPt.z) / 2;
    midpoint4.x = (center.x + minPt.x) / 2; midpoint4.y = (center.y + maxPt.y) / 2; midpoint4.z = (center.z + maxPt.z) / 2;
    midpoint5.x = (center.x + maxPt.x) / 2; midpoint5.y = (center.y + minPt.y) / 2; midpoint5.z = (center.z + minPt.z) / 2;
    midpoint6.x = (center.x + maxPt.x) / 2; midpoint6.y = (center.y + minPt.y) / 2; midpoint6.z = (center.z + maxPt.z) / 2;
    midpoint7.x = (center.x + maxPt.x) / 2; midpoint7.y = (center.y + maxPt.y) / 2; midpoint7.z = (center.z + minPt.z) / 2;
    midpoint8.x = (center.x + maxPt.x) / 2; midpoint8.y = (center.y + maxPt.y) / 2; midpoint8.z = (center.z + maxPt.z) / 2;

    centers.push_back(midpoint1); centers.push_back(midpoint2); centers.push_back(midpoint3); centers.push_back(midpoint4);
    centers.push_back(midpoint5); centers.push_back(midpoint6); centers.push_back(midpoint7); centers.push_back(midpoint8);

    return centers;
}

std::vector<pcl::PointXYZ> Helper::UniformSamplingSphere(pcl::PointXYZ center, double radius, size_t num_samples) {
    
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);

    std::vector<pcl::PointXYZ> samples;
    samples.reserve(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
        double theta = 2 * M_PI * distribution(generator);  // Azimuthal angle
        double phi = acos(2 * distribution(generator) - 1); // Polar angle
        pcl::PointXYZ sample;
        sample.x = center.x + radius * sin(phi) * cos(theta);
        sample.y = center.y + radius * sin(phi) * sin(theta);
        sample.z = center.z + radius * cos(phi);
        samples.push_back(sample);
    }

    return samples;
} 

Ray3D Helper::generateRay(const pcl::PointXYZ& center, const pcl::PointXYZ& surfacePoint) {
    Ray3D ray;
    ray.origin = center;

    // Compute the direction of the ray
    ray.direction.x = surfacePoint.x - center.x;
    ray.direction.y = surfacePoint.y - center.y;
    ray.direction.z = surfacePoint.z - center.z;

    // Normalize the direction to make it a unit vector
    double magnitude = sqrt(ray.direction.x * ray.direction.x +
                            ray.direction.y * ray.direction.y +
                            ray.direction.z * ray.direction.z);
    ray.direction.x /= magnitude;
    ray.direction.y /= magnitude;
    ray.direction.z /= magnitude;

    return ray;
}


bool Helper::rayIntersectDisk(const Ray3D& ray, const Disk3D& disk) {
    // Compute the vector from the ray origin to the disk center
    pcl::PointXYZ oc;
    oc.x = disk.center.x - ray.origin.x;
    oc.y = disk.center.y - ray.origin.y;
    oc.z = disk.center.z - ray.origin.z;

    // Compute the projection of oc onto the ray direction
    double projection = oc.x * ray.direction.x + oc.y * ray.direction.y + oc.z * ray.direction.z;

    // Compute the squared distance from the disk center to the projection point
    double oc_squared = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z;
    double distance_squared = oc_squared - projection * projection;

    // If the squared distance is less than or equal to the squared radius, the ray intersects the disk
    if (distance_squared <= disk.radius * disk.radius) {
        return true;
    }
    
    // If we get here, the ray does not intersect any disk
    return false;
}

Disk3D Helper::convertPointToDisk(const pcl::PointXYZ& point, const pcl::Normal& normal, const double& radius) {
    Disk3D disk;
    disk.center = point;
    disk.normal = normal;
    disk.radius = radius;

    return disk;
}


pcl::PointXYZ Helper::rayBoxIntersection(const Ray3D& ray, const pcl::PointXYZ& minPt, const pcl::PointXYZ& maxPt) {
    double tx, ty, tz, t;

    if (ray.direction.x >= 0) {
        tx = (maxPt.x - ray.origin.x) / ray.direction.x;
    } else {
        tx = (minPt.x - ray.origin.x) / ray.direction.x;
    }

    if (ray.direction.y >= 0) {
        ty = (maxPt.y - ray.origin.y) / ray.direction.y;
    } else {
        ty = (minPt.y - ray.origin.y) / ray.direction.y;
    }

    if (ray.direction.z >= 0) {
        tz = (maxPt.z - ray.origin.z) / ray.direction.z;
    } else {
        tz = (minPt.z - ray.origin.z) / ray.direction.z;
    }

    t = std::min(std::min(tx, ty), tz);

    pcl::PointXYZ intersection;
    intersection.x = ray.origin.x + t * ray.direction.x;
    intersection.y = ray.origin.y + t * ray.direction.y;
    intersection.z = ray.origin.z + t * ray.direction.z;

    std::cout << "*--> Intersection: " << intersection.x << ", " << intersection.y << ", " << intersection.z << std::endl;

    return intersection;
}

/*
    * This function checks if a ray intersects a point cloud.
    * It returns true if the ray intersects the point cloud, and false otherwise.
*/
bool Helper::rayIntersectPointCloud(const Ray3D& ray, 
                                    double step, double radius, 
                                    pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt,
                                    pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree) {
    // get the first point along the ray, which is the intersection of the ray with the bounding box
    pcl::PointXYZ currentPoint = Helper::rayBoxIntersection(ray, minPt, maxPt);
    
    while((currentPoint.x - ray.origin.x) > step || (currentPoint.y - ray.origin.y) > step || (currentPoint.z - ray.origin.z) > step) {
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        
        if (kdtree.radiusSearch(currentPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
            return true;
        }
        // update the current point along the ray
        currentPoint.x -= step * ray.direction.x;
        currentPoint.y -= step * ray.direction.y;
        currentPoint.z -= step * ray.direction.z;
    }
    return false;
}

/*
    * Compute the occlusion level of a point cloud using ray-based occlusion
    * @param cloud: the point cloud
    * @param num_samples: the number of samples to use for each light source
    * @return: the occlusion level of the point cloud
*/

double Helper::rayBasedOcclusionLevel(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, size_t num_samples, double step, double radius) {
    std::vector<pcl::PointXYZ> centers = Helper::getSphereLightSourceCenters(cloud);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    double occlusionLevel = 0.0;
    int numRays = centers.size() * num_samples;
    int occlusionRays = 0;

    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cloud, minPt, maxPt);

    for (size_t i = 0; i < centers.size(); ++i) {
        std::cout << "*********Center " << i << ": " << centers[i].x << ", " << centers[i].y << ", " << centers[i].z << "*********" << std::endl;
        std::vector<pcl::PointXYZ> samples = Helper::UniformSamplingSphere(centers[i], 0.1, num_samples);
        for (size_t j = 0; j < samples.size(); ++j) {
            auto start = std::chrono::high_resolution_clock::now();
            std::cout << "-----------Sample " << j << ": " << samples[j].x << ", " << samples[j].y << ", " << samples[j].z << "-----------" <<std::endl;
            Ray3D ray = Helper::generateRay(centers[i], samples[j]);
            if (!Helper::rayIntersectPointCloud(ray, step, radius, minPt, maxPt, kdtree)) {
                std::cout << "*--> Ray hit occlusion!!!" << std::endl;
                occlusionRays++;
            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "*--> Time taken for handling sample: " << duration.count() << " milliseconds" << std::endl;
            std::cout << std::endl;
        }
    }
    occlusionLevel = (double) occlusionRays / (double) numRays;
    std::cout << "Number of rays: " << numRays << std::endl;
    std::cout << "Number of occlusion rays: " << occlusionRays << std::endl;
    std::cout << "Occlusion level: " << occlusionLevel << std::endl;
    return occlusionLevel;
}
