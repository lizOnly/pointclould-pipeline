#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <thread>
#include <future>
#include <mutex>

#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/octree/octree.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/surface/convex_hull.h>

#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "../headers/helper.h"
#include "../headers/BaseStruct.h"


Helper::Helper() {
    
}

Helper::~Helper() {

}


// transform the point to the origin
pcl::PointXYZ Helper::transformPoint(pcl::PointXYZ& point,
                                     pcl::PointXYZ& center) {
    point.x -= center.x;
    point.y -= center.y;
    point.z -= center.z;

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(-M_PI/2, Eigen::Vector3f::UnitX()));

    Eigen::Vector4f pointAsVector(point.x, point.y, point.z, 1.0);
    Eigen::Vector4f transformedPointAsVector = transform * pointAsVector;
    
    return pcl::PointXYZ(transformedPointAsVector[0], transformedPointAsVector[1], transformedPointAsVector[2]);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Helper::centerCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                        pcl::PointXYZ& minPt, 
                                                        pcl::PointXYZ& maxPt) {
    pcl::PointXYZ center;
    center.x = (maxPt.x + minPt.x) / 2;
    center.y = (maxPt.y + minPt.y) / 2;
    center.z = (maxPt.z + minPt.z) / 2;

    for (size_t i = 0; i < cloud->points.size(); ++i) {

        cloud->points[i].x -= center.x;
        cloud->points[i].y -= center.y;
        cloud->points[i].z -= center.z;

    }

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(-M_PI/2, Eigen::Vector3f::UnitX()));
    pcl::transformPointCloud(*cloud, *cloud, transform);
    std::cout << "Transformed cloud" << std::endl;

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    
    return cloud;
}


// void Helper::reversePos(pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt) {
//     pcl::PointXYZ temp = minPt;
//     minPt = maxPt;
//     maxPt = temp;
// }


pcl::PointCloud<pcl::PointXYZRGB>::Ptr Helper::centerColoredCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
                                                                  pcl::PointXYZ& minPt, 
                                                                  pcl::PointXYZ& maxPt,
                                                                  std::string file_name) {
    pcl::PointXYZ center;
    center.x = (maxPt.x + minPt.x) / 2;
    center.y = (maxPt.y + minPt.y) / 2;
    center.z = (maxPt.z + minPt.z) / 2;

    for (size_t i = 0; i < coloredCloud->points.size(); ++i) {

        coloredCloud->points[i].x -= center.x;
        coloredCloud->points[i].y -= center.y;
        coloredCloud->points[i].z -= center.z;
    
    }

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(-M_PI / 2, Eigen::Vector3f::UnitX()));
    pcl::transformPointCloud(*coloredCloud, *coloredCloud, transform);
    std::cout << "Transformed colored cloud" << std::endl;

    coloredCloud->width = coloredCloud->points.size();
    coloredCloud->height = 1;
    coloredCloud->is_dense = true;
    
    pcl::io::savePCDFileASCII("../files/c_" + file_name, *coloredCloud);

    return coloredCloud;
}

/*
    This function is used exstract the walls from the point cloud
    and save them in a separate file
    @param cloud: the point cloud
    @return void
*/
void Helper::extractWalls(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch(0.05); 

    ne.compute(*cloud_normals);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_walls(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_walls_north(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_walls_south(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_walls_east(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_walls_west(new pcl::PointCloud<pcl::PointXYZ>);

    for (size_t i = 0; i < cloud->points.size(); ++i) {

        if (fabs(cloud_normals->points[i].normal_z) < 0.05) {

            if (fabs(cloud_normals->points[i].normal_x) > fabs(cloud_normals->points[i].normal_y)) {

                if (cloud_normals->points[i].normal_x > 0) {

                    cloud_walls_north->points.push_back(cloud->points[i]);

                } else {

                    cloud_walls_south->points.push_back(cloud->points[i]);

                }
            } else {

                if (cloud_normals->points[i].normal_y > 0) {

                    cloud_walls_east->points.push_back(cloud->points[i]);

                } else {

                    cloud_walls_west->points.push_back(cloud->points[i]);

                }
            }

            cloud_walls->points.push_back(cloud->points[i]);
        }

    }

    cloud_walls->width = cloud_walls->points.size();
    cloud_walls->height = 1;
    cloud_walls->is_dense = true;

    cloud_walls_north->width = cloud_walls_north->points.size();
    cloud_walls_north->height = 1;

    cloud_walls_south->width = cloud_walls_south->points.size();
    cloud_walls_south->height = 1;

    cloud_walls_east->width = cloud_walls_east->points.size();
    cloud_walls_east->height = 1;

    cloud_walls_west->width = cloud_walls_west->points.size();
    cloud_walls_west->height = 1;

    pcl::io::savePCDFileASCII("../files/walls_north.pcd", *cloud_walls_north);
    pcl::io::savePCDFileASCII("../files/walls_south.pcd", *cloud_walls_south);
    pcl::io::savePCDFileASCII("../files/walls_east.pcd", *cloud_walls_east);
    pcl::io::savePCDFileASCII("../files/walls_west.pcd", *cloud_walls_west);
    pcl::io::savePCDFileASCII("../files/walls.pcd", *cloud_walls);
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

    pcl::io::savePCDFileASCII("../files/specific_color_filtered_cloud.pcd", *filtered_cloud);

}


void Helper::regionGrowingSegmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normal_estimation.setSearchMethod(tree);
    normal_estimation.setRadiusSearch(0.06);
    normal_estimation.compute(*normals);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(1000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(30);
    reg.setInputCloud(cloud);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.0);

    std::vector <pcl::PointIndices> clusters;
    reg.extract(clusters);

    std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;
    std::cout << "First cluster has " << clusters[0].indices.size() << " points." << std::endl;


    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
    pcl::io::savePCDFileASCII("../files/output/region_growing_segmentation.pcd", *colored_cloud);
}


std::vector<pcl::PointXYZ> Helper::getSphereLightSourceCenters(pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt) {

    std::vector<pcl::PointXYZ> centers;
    pcl::PointXYZ center;
    center.x = (maxPt.x + minPt.x) / 2;
    center.y = (maxPt.y + minPt.y) / 2;
    center.z = (maxPt.z + minPt.z) / 2;

    centers.push_back(center);

    // Points at the midpoints of the body diagonals (diagonal was divided by center point )
    pcl::PointXYZ midpoint1, midpoint2, midpoint3, midpoint4,
                  midpoint5, midpoint6, midpoint7, midpoint8;

    midpoint1.x = (center.x + minPt.x) / 2; // v2
    midpoint1.y = (center.y + minPt.y) / 2; 
    midpoint1.z = (center.z + minPt.z) / 2;

    midpoint2.x = (center.x + minPt.x) / 2; 
    midpoint2.y = (center.y + minPt.y) / 2; 
    midpoint2.z = (center.z + maxPt.z) / 2;
    
    midpoint3.x = (center.x + minPt.x) / 2; 
    midpoint3.y = (center.y + maxPt.y) / 2; 
    midpoint3.z = (center.z + minPt.z) / 2;
    
    midpoint4.x = (center.x + minPt.x) / 2; 
    midpoint4.y = (center.y + maxPt.y) / 2; 
    midpoint4.z = (center.z + maxPt.z) / 2;
    
    midpoint5.x = (center.x + maxPt.x) / 2; 
    midpoint5.y = (center.y + minPt.y) / 2; 
    midpoint5.z = (center.z + minPt.z) / 2;
    
    midpoint6.x = (center.x + maxPt.x) / 2; 
    midpoint6.y = (center.y + minPt.y) / 2; 
    midpoint6.z = (center.z + maxPt.z) / 2;
    
    midpoint7.x = (center.x + maxPt.x) / 2; 
    midpoint7.y = (center.y + maxPt.y) / 2; 
    midpoint7.z = (center.z + minPt.z) / 2;
    
    midpoint8.x = (center.x + maxPt.x) / 2; // v1
    midpoint8.y = (center.y + maxPt.y) / 2; 
    midpoint8.z = (center.z + maxPt.z) / 2;

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


pcl::ModelCoefficients::Ptr Helper::computePlaneCoefficients(std::vector<pcl::PointXYZ> points) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto& point : points){
        cloud->points.push_back(point);
    }
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(false);
    seg.setModelType(pcl::SACMODEL_PLANE); // SACMODEL_PLANE
    seg.setMethodType(pcl::SAC_RANSAC); // SAC_RANSAC
    seg.setDistanceThreshold(0.01);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0) {
        PCL_ERROR("Could not estimate a planar model for the given dataset.");
        return nullptr;
    }
    return coefficients;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr Helper::estimatePolygon(std::vector<pcl::PointXYZ> points, 
                                                            pcl::ModelCoefficients::Ptr coefficients) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    for (auto& point : points){
        cloud->points.push_back(point);
    }
    
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud);
    proj.setModelCoefficients(coefficients);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);
    proj.filter(*cloud_projected);

    pcl::ConvexHull<pcl::PointXYZ> chull;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
    chull.setDimension(2);
    chull.setInputCloud(cloud_projected);
    chull.reconstruct(*cloud_hull);

    std::cout << "Convex hull has: " << cloud_hull->points.size() << " data points." << std::endl;
    std::cout << "Points are: " << std::endl;
    
    for (size_t i = 0; i < cloud_hull->points.size(); ++i) {

        std::cout << "    " << cloud_hull->points[i].x << " "
                  << cloud_hull->points[i].y << " "
                  << cloud_hull->points[i].z << std::endl;

    }

    return cloud_hull;
}

std::vector<std::vector<pcl::PointXYZ>> Helper::parsePointString(const std::string& input) {
    std::vector<std::vector<pcl::PointXYZ>> result;
    
    std::istringstream ss(input);
    std::string line;
    while (std::getline(ss, line, ';')) { 
        std::vector<pcl::PointXYZ> group;
        std::istringstream group_ss(line);
        std::string point_str;

        while (std::getline(group_ss, point_str, ',')) { 
            std::istringstream point_ss(point_str);
            double x, y, z;
            if (point_ss >> x >> y >> z) {
                // std::cout << "Point: " << x << " " << y << " " << z << std::endl;
                group.push_back(pcl::PointXYZ(x, y, z));
            }
        }
        result.push_back(group);
    }

    return result;
}


std::vector<std::vector<pcl::PointXYZ>> Helper::parsePolygonData(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    std::vector<std::vector<pcl::PointXYZ>> polygons;
    std::vector<pcl::PointXYZ> polygon;

    while (std::getline(infile, line)) {
        if (line.empty()) {
            if (!polygon.empty()) {
                polygons.push_back(polygon);
                polygon.clear();
            }
        } else {
            std::istringstream iss(line);
            double x, y, z;
            if (iss >> x >> y >> z) {
                polygon.push_back(pcl::PointXYZ(x, y, z));
            }
        }
    }

    if (!polygon.empty()) {
        polygons.push_back(polygon);
    }

    std::cout << polygons.size() << std::endl;
    return polygons;
}


bool Helper::rayIntersectPolygon(const Ray3D& ray, 
                                 const pcl::PointCloud<pcl::PointXYZ>::Ptr& polygonCloud, 
                                 const pcl::ModelCoefficients::Ptr coefficients) {
                                    
    float a = coefficients->values[0];
    float b = coefficients->values[1];
    float c = coefficients->values[2];
    float d = coefficients->values[3];

    pcl::PointXYZ origin = ray.origin;
    pcl::PointXYZ direction = ray.direction;
    pcl::PointXYZ intersection;
    // Direction of the ray
    float dx = direction.x;
    float dy = direction.y;
    float dz = direction.z;

    // Origin of the ray
    float ox = origin.x;
    float oy = origin.y;
    float oz = origin.z;

    // Calculate the denominator of the t parameter
    float denom = a * dx + b * dy + c * dz;
    if (fabs(denom) > std::numeric_limits<float>::epsilon()) {

        float t = -(a * ox + b * oy + c * oz + d) / denom;
        // If t is negative, the intersection point is "behind" the origin of the ray (we usually discard this case)
        
        if (t >= 0) {
            intersection.x = ox + t * dx;
            intersection.y = oy + t * dy;
            intersection.z = oz + t * dz;
        }
        else {
            return false;
        }
    }

    int n = polygonCloud->points.size();

    Eigen::Vector3f intersectionPoint(intersection.x, intersection.y, intersection.z);
    Eigen::Vector3f polygonVertex(polygonCloud->points[0].x, polygonCloud->points[0].y, polygonCloud->points[0].z);
    Eigen::Vector3f polygonVertex2(polygonCloud->points[1].x, polygonCloud->points[1].y, polygonCloud->points[1].z);

    Eigen::Vector3f polygonVector = polygonVertex - intersectionPoint;
    Eigen::Vector3f polygonVector2 = polygonVertex2 - intersectionPoint;

    Eigen::Vector3f cross = polygonVector.cross(polygonVector2);
    
    for (int i = 1; i < n; ++i) {
        polygonVertex = polygonVertex2;
        polygonVertex2 = Eigen::Vector3f(polygonCloud->points[(i + 1) % n].x, polygonCloud->points[(i + 1) % n].y, polygonCloud->points[(i + 1) % n].z);

        polygonVector = polygonVector2;
        polygonVector2 = polygonVertex2 - intersectionPoint;

        Eigen::Vector3f cross2 = polygonVector.cross(polygonVector2);
        if (cross.dot(cross2) < 0) {
            return false;
        }
    }
    return true;
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


pcl::PointXYZ Helper::rayBoxIntersection(const Ray3D& ray, 
                                         const pcl::PointXYZ& minPt, 
                                         const pcl::PointXYZ& maxPt) {
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
    
    return intersection;
}


/*
    * This function checks if a ray intersects a point cloud.
    * It returns true if the ray intersects the point cloud, and false otherwise.
    * @param ray: the ray
    * @param step: the step size for ray traversal
    * @param radius: the radius for radius search
    * @param minPt: the minimum point of the bounding box of the point cloud
    * @param maxPt: the maximum point of the bounding box of the point cloud
    * @param kdtree: the kd-tree for the point cloud
    * @return: true if the ray intersects the point cloud, and false otherwise
*/
bool Helper::rayIntersectPointCloud(const Ray3D& ray, 
                                    double step, 
                                    double radius, 
                                    pcl::PointXYZ& minPt, 
                                    pcl::PointXYZ& maxPt,
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


pcl::PointCloud<pcl::PointXYZI>::Ptr Helper::computeMedianDistance( double radius, 
                                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_density) {

    std::cout << "Computing median distance..." << std::endl;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_median_distance(new pcl::PointCloud<pcl::PointXYZI>);
    
    std::vector<std::future<void>> futures;
    std::mutex mtx;

    double density_threshold = 10.0;

    for (size_t i = 0; i < cloud->points.size(); ++i) {

        // futures.push_back(std::async(std::launch::async, [=, &mtx, &cloud_with_median_distance, &kdtree]() {
  
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
        
            pcl::PointXYZI point;
            point.x = cloud->points[i].x;
            point.y = cloud->points[i].y;
            point.z = cloud->points[i].z;
            point.intensity = 0.025;

            pcl::PointXYZI point_with_density = cloud_with_density->points[i];

            double density = point_with_density.intensity;

            if (density < density_threshold) {
                cloud_with_median_distance->points.push_back(point);
                continue;
            }

            double search_radius = radius;
            
            if (kdtree.radiusSearch(cloud->points[i], search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
                if(pointIdxRadiusSearch.size() < 2) {
                    continue;
                }
                // calculate pairwise distances
                std::vector<double> distances;
                // if (pointIdxRadiusSearch.size() > 50) {
                //     pointIdxRadiusSearch.resize(50);
                // }

                for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {

                    for (size_t k = j + 1; k < pointIdxRadiusSearch.size(); ++k) {

                        double distance = sqrt(pow(cloud->points[pointIdxRadiusSearch[j]].x - cloud->points[pointIdxRadiusSearch[k]].x, 2) +
                                            pow(cloud->points[pointIdxRadiusSearch[j]].y - cloud->points[pointIdxRadiusSearch[k]].y, 2) +
                                            pow(cloud->points[pointIdxRadiusSearch[j]].z - cloud->points[pointIdxRadiusSearch[k]].z, 2));
                        
                        distances.push_back(distance);
                    }
                }

                std::sort(distances.begin(), distances.end());
                point.intensity = distances[distances.size() / 2];
            }
            
            // mtx.lock();
            cloud_with_median_distance->points.push_back(point);
            // mtx.unlock();

        // }));
    }

    // for (auto& f : futures) {
    //     f.get();
    // }

    return cloud_with_median_distance;
}


bool Helper::rayIntersectPcdMedianDistance( const Ray3D& ray, 
                                            double step, 
                                            double radius, 
                                            pcl::PointXYZ& minPt, 
                                            pcl::PointXYZ& maxPt,
                                            pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree,
                                            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_median_distance) {
    
    pcl::PointXYZ currentPoint = Helper::rayBoxIntersection(ray, minPt, maxPt);

    while((currentPoint.x - ray.origin.x) > step || (currentPoint.y - ray.origin.y) > step || (currentPoint.z - ray.origin.z) > step) {
        
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        
        if (kdtree.radiusSearch(currentPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
            
            for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {

                pcl::PointXYZI point = cloud_with_median_distance->points[pointIdxRadiusSearch[i]];
                double per_point_distance = sqrt(pow(point.x - currentPoint.x, 2) +
                                                 pow(point.y - currentPoint.y, 2) +
                                                 pow(point.z - currentPoint.z, 2));
                
                if (point.intensity > per_point_distance ) {
                    return true;
                }
            }
            
        }

        currentPoint.x -= step * ray.direction.x;
        currentPoint.y -= step * ray.direction.y;
        currentPoint.z -= step * ray.direction.z;

    }

    return false;
}


pcl::PointCloud<pcl::PointXYZI>::Ptr Helper::computeDistanceVariance(double radius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    
        std::cout << "Computing distance variance..." << std::endl;
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud);
    
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_distance_variance(new pcl::PointCloud<pcl::PointXYZI>);
        
        std::vector<std::future<void>> futures;
        std::mutex mtx;
    
        for (size_t i = 0; i < cloud->points.size(); ++i) {
        
                std::vector<int> pointIdxRadiusSearch;
                std::vector<float> pointRadiusSquaredDistance;
                
                pcl::PointXYZI point;
                point.x = cloud->points[i].x;
                point.y = cloud->points[i].y;
                point.z = cloud->points[i].z;
                point.intensity = 0.0;
                

                if (kdtree.radiusSearch(cloud->points[i], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
                    
                    if(pointIdxRadiusSearch.size() < 2) {
                        continue;
                    }
                    
                    std::vector<double> distances;

                    for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
    
                        for (size_t k = j + 1; k < pointIdxRadiusSearch.size(); ++k) {
    
                            double distance = sqrt(pow(cloud->points[pointIdxRadiusSearch[j]].x - cloud->points[pointIdxRadiusSearch[k]].x, 2) +
                                                pow(cloud->points[pointIdxRadiusSearch[j]].y - cloud->points[pointIdxRadiusSearch[k]].y, 2) +
                                                pow(cloud->points[pointIdxRadiusSearch[j]].z - cloud->points[pointIdxRadiusSearch[k]].z, 2));
                            
                            distances.push_back(distance);
                        }
                    }
    
                    double mean = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
                    double variance = 0.0;
                    for (size_t j = 0; j < distances.size(); ++j) {
                        variance += pow(distances[j] - mean, 2);
                    }

                    variance /= distances.size();
                    point.intensity = variance;
                }
        }

    return cloud_with_distance_variance;
}


/*
    * Compute the occlusion level of a point cloud using ray-based occlusion
    * @param cloud: the point cloud
    * @param num_samples: the number of samples to use for each light source
    * @return: the occlusion level of the point cloud
*/
double Helper::rayBasedOcclusionLevelMedian(pcl::PointXYZ& minPt, 
                                      pcl::PointXYZ& maxPt,
                                      double density,
                                      double radius,
                                      int pattern,
                                      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                                      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_median_distance,
                                      std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds,
                                      std::vector<pcl::ModelCoefficients::Ptr> allCoefficients) {
    

    std::vector<pcl::PointXYZ> centers = Helper::getSphereLightSourceCenters(minPt, maxPt);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    size_t num_samples = 4000;
    double step = 0.05; // step size for ray traversal
    double occlusionLevel = 0.0;

    // radius = radius / density;

    std::cout << "Radius: " << radius << std::endl;

    int numRays = centers.size() * num_samples;
    int occlusionRays = 0;
    int polygonIntersecRays = 0;
    int cloudIntersecRays = 0;
    double sphereRadius = 0.1;


    for (size_t i = 0; i < centers.size(); ++i) {
        if (pattern == 2) {
            if (i == 0) { // ignore the center
                continue;
            }
        } else if (pattern == 4) {
            if (i == 8) { // ignore the max point
                continue;
            }
        } else if (pattern == 5) { 
            if (i == 1) { // ignore the min point
                continue;
            }
        }

        // std::cout << "*********Center " << i << ": " << centers[i].x << ", " << centers[i].y << ", " << centers[i].z << "*********" << std::endl;
        std::vector<pcl::PointXYZ> samples = Helper::UniformSamplingSphere(centers[i], sphereRadius, num_samples);

        // iterate over the samples
        for (size_t j = 0; j < samples.size(); ++j) {
            Ray3D ray = Helper::generateRay(centers[i], samples[j]);            
            // check if the ray intersects any polygon or point cloud
            // if (Helper::rayIntersectPointCloud(ray, step, radius, minPt, maxPt, kdtree)) {
            if (Helper::rayIntersectPcdMedianDistance(ray, step, radius, minPt, maxPt, kdtree, cloud_with_median_distance)) {
                // std::cout << "*--> Ray hit cloud!!!" << std::endl;
                cloudIntersecRays++;

            } else {

                for (size_t k = 0; k < polygonClouds.size(); ++k) {

                    if (Helper::rayIntersectPolygon(ray, polygonClouds[k], allCoefficients[k])) {

                        // std::cout << "*--> Ray didn't hit cloud but hit polygon of index " << k << std::endl;
                        polygonIntersecRays++;
                        break;

                    } else if (k == (polygonClouds.size() - 1)) {

                        // std::cout << "*--> Ray did not hit anything, it's an occlusion" << std::endl;
                        occlusionRays++;

                    }
                }
            }
        }
    }

    occlusionLevel = (double) occlusionRays / (double) numRays;
    
    std::cout << "Number of rays: " << numRays << std::endl;
    std::cout << "Number of cloud intersection rays: " << cloudIntersecRays << std::endl;
    std::cout << "Number of polygon intersection rays: " << polygonIntersecRays << std::endl;
    std::cout << "Number of occlusion rays: " << occlusionRays << std::endl;
    std::cout << "Occlusion level: " << occlusionLevel << std::endl;
    
    return occlusionLevel;
}


double Helper::rayBasedOcclusionLevel(pcl::PointXYZ& minPt, 
                                      pcl::PointXYZ& maxPt,
                                      double radius,
                                      int pattern,
                                      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                                      std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds,
                                      std::vector<pcl::ModelCoefficients::Ptr> allCoefficients) {
    

    std::vector<pcl::PointXYZ> centers = Helper::getSphereLightSourceCenters(minPt, maxPt);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    size_t num_samples = 10000;
    double step = 0.05; // step size for ray traversal
    double occlusionLevel = 0.0;

    // radius = radius / density;

    std::cout << "Radius: " << radius << std::endl;

    int numRays = centers.size() * num_samples;
    int occlusionRays = 0;
    int polygonIntersecRays = 0;
    int cloudIntersecRays = 0;
    double sphereRadius = 0.1;

    for (size_t i = 0; i < centers.size(); ++i) {
        if (pattern == 2) {
            if (i == 0) { // ignore the center
                continue;
            }
        } else if (pattern == 4) {
            if (i == 8) { // ignore the max point
                continue;
            }
        } else if (pattern == 5) { 
            if (i == 1) { // ignore the min point
                continue;
            }
        }

        // std::cout << "*********Center " << i << ": " << centers[i].x << ", " << centers[i].y << ", " << centers[i].z << "*********" << std::endl;
        std::vector<pcl::PointXYZ> samples = Helper::UniformSamplingSphere(centers[i], sphereRadius, num_samples);

        // iterate over the samples
        for (size_t j = 0; j < samples.size(); ++j) {
            Ray3D ray = Helper::generateRay(centers[i], samples[j]);            
            // check if the ray intersects any polygon or point cloud
            if (Helper::rayIntersectPointCloud(ray, step, radius, minPt, maxPt, kdtree)) {
                // std::cout << "*--> Ray hit cloud!!!" << std::endl;
                cloudIntersecRays++;

            } else {

                for (size_t k = 0; k < polygonClouds.size(); ++k) {

                    if (Helper::rayIntersectPolygon(ray, polygonClouds[k], allCoefficients[k])) {

                        // std::cout << "*--> Ray didn't hit cloud but hit polygon of index " << k << std::endl;
                        polygonIntersecRays++;
                        break;

                    } else if (k == (polygonClouds.size() - 1)) {

                        // std::cout << "*--> Ray did not hit anything, it's an occlusion" << std::endl;
                        occlusionRays++;

                    }
                }
            }
        }
    }

    occlusionLevel = (double) occlusionRays / (double) numRays;
    
    std::cout << "Number of rays: " << numRays << std::endl;
    std::cout << "Number of cloud intersection rays: " << cloudIntersecRays << std::endl;
    std::cout << "Number of polygon intersection rays: " << polygonIntersecRays << std::endl;
    std::cout << "Number of occlusion rays: " << occlusionRays << std::endl;
    std::cout << "Occlusion level: " << occlusionLevel << std::endl;
    
    return occlusionLevel;
}
