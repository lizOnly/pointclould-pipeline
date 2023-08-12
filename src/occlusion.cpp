#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <thread>
#include <future>
#include <mutex>
#include <unordered_map>
#include <tuple>

#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/passthrough.h>

#include <pcl/octree/octree_base.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_pointcloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/transforms.h>
#include <pcl/surface/convex_hull.h>

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "../headers/occlusion.h"
#include "../headers/BaseStruct.h"


Occlusion::Occlusion() {
    
}

Occlusion::~Occlusion() {

}



pcl::PointCloud<pcl::PointXYZ>::Ptr Occlusion::centerCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt) {
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


pcl::PointCloud<pcl::PointXYZRGB>::Ptr Occlusion::centerColoredCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
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
void Occlusion::extractWalls(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

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


void Occlusion::removePointsInSpecificColor(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int color[3]) {

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


void Occlusion::regionGrowingSegmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

    input_cloud = cloud;

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

    
    reg.extract(rg_clusters);

    std::cout << "Number of clusters is equal to " << rg_clusters.size() << std::endl;

    // pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
    // pcl::io::savePCDFileASCII("../files/rg.pcd", *colored_cloud);
}

Eigen::Vector3d Occlusion::computeCentroid(pcl::PointCloud<pcl::PointXYZ>::Ptr polygon_cloud) {
    double A = 0;  
    Eigen::Vector3d centroid(0, 0, 0);

    for (size_t i = 0; i < polygon_cloud->points.size(); ++i) {
        size_t next = (i + 1) % polygon_cloud->points.size();

        double xi = polygon_cloud->points[i].x;
        double yi = polygon_cloud->points[i].y;
        double xi1 = polygon_cloud->points[next].x;
        double yi1 = polygon_cloud->points[next].y;

        double Ai = xi * yi1 - xi1 * yi;
        A += Ai;

        centroid[0] += (xi + xi1) * Ai;
        centroid[1] += (yi + yi1) * Ai;
    }


    A *= 0.5;  
    centroid /= (6.0 * A); 

    return centroid;
}

void Occlusion::generateTriangleFromCluster() {
    size_t tri_idx = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr polygon_clouds(new pcl::PointCloud<pcl::PointXYZ>);

    for(auto& cluster : rg_clusters) {
        std::vector<pcl::PointXYZ> points; // points for plane estimation
        for(auto& index : cluster.indices) {
            points.push_back(input_cloud->points[index]);
        }
        pcl::ModelCoefficients::Ptr coefficients = computePlaneCoefficients(points);
        pcl::PointCloud<pcl::PointXYZ>::Ptr polygon_cloud = estimatePolygon(points, coefficients);

        Eigen::Vector3d polygon_center;
        polygon_center = computeCentroid(polygon_cloud);

        size_t polygon_size = polygon_cloud->points.size();
        for(size_t i = 0; i < polygon_size; ++i) {
            polygon_clouds->points.push_back(polygon_cloud->points[i]);
            Triangle triangle;
            triangle.index = tri_idx;
            tri_idx++;

            triangle.v1 = Eigen::Vector3d(polygon_center[0], polygon_center[1], polygon_center[2]);
            triangle.v2 = Eigen::Vector3d(polygon_cloud->points[i].x, polygon_cloud->points[i].y, polygon_cloud->points[i].z);
            triangle.v3 = Eigen::Vector3d(polygon_cloud->points[(i + 1) % polygon_size].x, polygon_cloud->points[(i + 1) % polygon_size].y, polygon_cloud->points[(i + 1) % polygon_size].z);


            triangle.center = (triangle.v1 + triangle.v2 + triangle.v3) / 3.0;
            t_triangles[triangle.index] = triangle;
        
        }
    }
    polygon_clouds->width = polygon_clouds->points.size();
    polygon_clouds->height = 1;
    polygon_clouds->is_dense = true;

    pcl::io::savePCDFileASCII("../files/polygon_clouds.pcd", *polygon_clouds);

    std::cout << "Number of triangles is: " << t_triangles.size() << std::endl;
}


std::vector<pcl::PointXYZ> Occlusion::getSphereLightSourceCenters(pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt) {

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


std::vector<pcl::PointXYZ> Occlusion::UniformSamplingSphere(pcl::PointXYZ center, double radius, size_t num_samples) {
    
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


pcl::ModelCoefficients::Ptr Occlusion::computePlaneCoefficients(std::vector<pcl::PointXYZ> points) {
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


pcl::PointCloud<pcl::PointXYZ>::Ptr Occlusion::estimatePolygon(std::vector<pcl::PointXYZ> points, pcl::ModelCoefficients::Ptr coefficients) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    for (auto& point : points) {
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

    // std::cout << "Convex hull has: " << cloud_hull->points.size() << " data points." << std::endl;
    // std::cout << "Points are: " << std::endl;
    
    // for (size_t i = 0; i < cloud_hull->points.size(); ++i) {

    //     std::cout << "    " << cloud_hull->points[i].x << " "
    //               << cloud_hull->points[i].y << " "
    //               << cloud_hull->points[i].z << std::endl;

    // }

    return cloud_hull;
}

std::vector<std::vector<pcl::PointXYZ>> Occlusion::parsePointString(const std::string& input) {
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


std::vector<std::vector<pcl::PointXYZ>> Occlusion::parsePolygonData(const std::string& filename) {
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


bool Occlusion::rayIntersectPolygon(const Ray3D& ray, const pcl::PointCloud<pcl::PointXYZ>::Ptr& polygonCloud, const pcl::ModelCoefficients::Ptr coefficients) {
                                    
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


Ray3D Occlusion::generateRay(const pcl::PointXYZ& center, const pcl::PointXYZ& surfacePoint) {

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


bool Occlusion::rayIntersectDisk(const Ray3D& ray, const Disk3D& disk) {

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


/*
    This function calculates the intersection point of a ray with the bounding box of point cloud.
    @param ray: the ray
    @param minPt: the minimum point of the bounding box
    @param maxPt: the maximum point of the bounding box
    @return: the intersection point
*/
bool Occlusion::rayBoxIntersection(const Ray3D& ray, const pcl::PointXYZ& minPt, const pcl::PointXYZ& maxPt) {

    if(ray.origin.x >= minPt.x && ray.origin.x <= maxPt.x &&
       ray.origin.y >= minPt.y && ray.origin.y <= maxPt.y &&
       ray.origin.z >= minPt.z && ray.origin.z <= maxPt.z) {
        return true;
    }
    
    double tmin, tmax, tymin, tymax, tzmin, tzmax;

    if (ray.direction.x != 0) {
        if (ray.direction.x >= 0) {
            tmin = (minPt.x - ray.origin.x) / ray.direction.x;
            tmax = (maxPt.x - ray.origin.x) / ray.direction.x;
        } else {
            tmin = (maxPt.x - ray.origin.x) / ray.direction.x;
            tmax = (minPt.x - ray.origin.x) / ray.direction.x;
        }
    } else {
        if (ray.origin.x < minPt.x || ray.origin.x > maxPt.x) {
            return false;
        }
        tmin = std::numeric_limits<double>::lowest();
        tmax = std::numeric_limits<double>::max();
    }

    if (ray.direction.y != 0) {
        if (ray.direction.y >= 0) {
            tymin = (minPt.y - ray.origin.y) / ray.direction.y;
            tymax = (maxPt.y - ray.origin.y) / ray.direction.y;
        } else {
            tymin = (maxPt.y - ray.origin.y) / ray.direction.y;
            tymax = (minPt.y - ray.origin.y) / ray.direction.y;
        }
    } else {
        if (ray.origin.y < minPt.y || ray.origin.y > maxPt.y) {
            return false;
        }
        tymin = std::numeric_limits<double>::lowest();
        tymax = std::numeric_limits<double>::max();
    }

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    if (ray.direction.z != 0) {
        if (ray.direction.z >= 0) {
            tzmin = (minPt.z - ray.origin.z) / ray.direction.z;
            tzmax = (maxPt.z - ray.origin.z) / ray.direction.z;
        } else {
            tzmin = (maxPt.z - ray.origin.z) / ray.direction.z;
            tzmax = (minPt.z - ray.origin.z) / ray.direction.z;
        }
    } else {
        if (ray.origin.z < minPt.z || ray.origin.z > maxPt.z) {
            return false;
        }
        tzmin = std::numeric_limits<double>::lowest();
        tzmax = std::numeric_limits<double>::max();
    }

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    return true;
}


bool Occlusion::rayIntersectSpehre(pcl::PointXYZ& origin, pcl::PointXYZ& direction, pcl::PointXYZ& point) {
    
    double radius = 0.025;
    double dirMagnitude = sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    direction.x /= dirMagnitude;
    direction.y /= dirMagnitude;
    direction.z /= dirMagnitude;

    pcl::PointXYZ L(point.x - origin.x, point.y - origin.y, point.z - origin.z);

    double originDistance2 = L.x * L.x + L.y * L.y + L.z * L.z;
    if (originDistance2 < radius * radius) return true;  // origin is inside the sphere

    double t_ca = L.x * direction.x + L.y * direction.y + L.z * direction.z;

    if (t_ca < 0) return false;

    double d2 = originDistance2 - t_ca * t_ca;

    if (d2 > radius * radius) return false;

    return true;

}

/*
    * This function checks if a ray intersects a point cloud.
    * It returns true if the ray intersects the point cloud, and false otherwise.
    * @param ray: the ray
    * @return: true if the ray intersects the point cloud, and false otherwise
*/
bool Occlusion::rayIntersectPointCloud(const Ray3D& ray) {

    pcl::PointXYZ origin = ray.origin;
    pcl::PointXYZ direction = ray.direction;

    for(auto& bbox : octree_leaf_bbox) {

        pcl::PointXYZ min_pt(bbox.min_pt.x(), bbox.min_pt.y(), bbox.min_pt.z());
        pcl::PointXYZ max_pt(bbox.max_pt.x(), bbox.max_pt.y(), bbox.max_pt.z());

        if(rayBoxIntersection(ray, min_pt, max_pt)) {
            for(auto& point_idx : bbox.point_idx) {
                pcl::PointXYZ point = input_cloud->points[point_idx];
                if(rayIntersectSpehre(origin, direction, point)) {
                    return true;
                }
            }
        }

    }

    return false;
}


pcl::PointCloud<pcl::PointXYZI>::Ptr Occlusion::computeMedianDistance(double radius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_density) {

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
                std::vector<double> distances;

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



void Occlusion::traverseOctree() {

    float resolution = 0.5;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
    octree.setInputCloud(input_cloud);
    octree.addPointsFromInputCloud();

    int max_depth = octree.getTreeDepth();
    std::cout << "Max depth: " << max_depth << std::endl;
    int num_leaf_nodes = octree.getLeafCount();
    std::cout << "Total number of leaf nodes: " << num_leaf_nodes << std::endl;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::LeafNodeIterator it;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::LeafNodeIterator it_end = octree.leaf_depth_end();

    

    for (it = octree.leaf_depth_begin(max_depth); it != it_end; ++it) {
        Eigen::Vector3f min_pt, max_pt;
        
        pcl::octree::OctreeKey key = it.getCurrentOctreeKey();

        octree.getVoxelBounds(it, min_pt, max_pt);

        // std::cout << "Min point: " << min_pt.x() << ", " << min_pt.y() << ", " << min_pt.z() << std::endl;
        LeafBBox bbox;
        bbox.min_pt.x() = static_cast<double>(min_pt.x());
        bbox.min_pt.y() = static_cast<double>(min_pt.y());
        bbox.min_pt.z() = static_cast<double>(min_pt.z());
        
        bbox.max_pt.x() = static_cast<double>(max_pt.x());
        bbox.max_pt.y() = static_cast<double>(max_pt.y());
        bbox.max_pt.z() = static_cast<double>(max_pt.z());

        std::vector<int> point_idx = it.getLeafContainer().getPointIndicesVector();
        // std::cout << "Number of points in leaf node: " << point_idx.size() << std::endl;
        for (auto& idx : point_idx) {
            bbox.point_idx.push_back(idx);
        }
        octree_leaf_bbox.push_back(bbox);
    }

    std::cout << "Number of leaf bbox: " << octree_leaf_bbox.size() << std::endl;
}

double Occlusion::rayBasedOcclusionLevel(pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt, int pattern, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                                         std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds, std::vector<pcl::ModelCoefficients::Ptr> allCoefficients) {
                                            
    input_cloud = cloud;

    traverseOctree();

    std::vector<pcl::PointXYZ> centers = Occlusion::getSphereLightSourceCenters(minPt, maxPt);

    size_t num_samples = 10000;
    double step = 0.05; // step size for ray traversal
    double occlusionLevel = 0.0;

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
        std::vector<pcl::PointXYZ> samples = UniformSamplingSphere(centers[i], sphereRadius, num_samples);

        // iterate over the samples
        for (size_t j = 0; j < samples.size(); ++j) {
            Ray3D ray = generateRay(centers[i], samples[j]);            
            // check if the ray intersects any polygon or point cloud
            if (rayIntersectPointCloud(ray)) {
                // std::cout << "*--> Ray hit cloud!!!" << std::endl;
                cloudIntersecRays++;

            } else {

                for (size_t k = 0; k < polygonClouds.size(); ++k) {

                    if (Occlusion::rayIntersectPolygon(ray, polygonClouds[k], allCoefficients[k])) {

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

// read the .obj file and return a vector of triangles
void Occlusion::parseTrianglesFromOBJ(const std::string& mesh_path) {
    std::cout << "Parsing mesh from " << mesh_path << std::endl;
    size_t t_idx = 0;

    std::ifstream file(mesh_path);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << mesh_path << std::endl;
        return;
    }

    std::string line;
    std::cout << "Reading file..." << std::endl;

    Eigen::Vector3d min_vertex(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    Eigen::Vector3d max_vertex(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string type;
        ss >> type;

        if (type == "v") { 
            Eigen::Vector3d vertex;
            ss >> vertex.x() >> vertex.y() >> vertex.z();

            min_vertex = min_vertex.cwiseMin(vertex);
            max_vertex = max_vertex.cwiseMax(vertex);

            vertices.push_back(vertex);
        } else if (type == "f") { 
            std::string i_str, j_str, k_str;
            ss >> i_str >> j_str >> k_str;

            size_t i = std::stoi(i_str.substr(0, i_str.find('/'))) - 1;
            size_t j = std::stoi(j_str.substr(0, j_str.find('/'))) - 1;
            size_t k = std::stoi(k_str.substr(0, k_str.find('/'))) - 1;

            Triangle triangle;
            triangle.v1 = vertices[i];
            triangle.v2 = vertices[j];
            triangle.v3 = vertices[k];
            triangle.index = t_idx;
            triangle.center = (triangle.v1 + triangle.v2 + triangle.v3) / 3.0;
            t_idx++;
            t_triangles[triangle.index] = triangle;
        }
    }
    std::cout << "Number of triangles: " << t_triangles.size() << std::endl;
    std::cout << "Minimum vertex: " << min_vertex.transpose() << std::endl;
    std::cout << "Maximum vertex: " << max_vertex.transpose() << std::endl;
    
    file.close();
}


void Occlusion::computeMeshBoundingBox() {
    std::cout << "Computing mesh bounding box..." << std::endl;
    for (const Eigen::Vector3d &vertex : vertices) {
        bbox.extend(vertex); 
    }
}

double Occlusion::calculateTriangleArea(Triangle& tr) {
    Eigen::Vector3d v1 = tr.v1;
    Eigen::Vector3d v2 = tr.v2;
    Eigen::Vector3d v3 = tr.v3;

    Eigen::Vector3d v12 = v2 - v1;
    Eigen::Vector3d v13 = v3 - v1;

    double area = 0.5 * v12.cross(v13).norm();

    if (std::isnan(area)) {
        return 0.0;
    }

    return area;
}

// generate a ray from the light source to the point on the sphere
void Occlusion::generateRaysWithIdx(std::vector<Eigen::Vector3d>& origins, size_t num_samples) {
    std::cout << "Generating rays..." << std::endl;
    size_t idx = 0;
    double radius = 1.0; // radius of the sphere
    // uniform sampling on a sphere which center is the origin
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (auto& origin : origins) {
        for (size_t i = 0; i < num_samples; ++i) {
            double theta = 2 * M_PI * distribution(generator);  // Azimuthal angle
            double phi = acos(2 * distribution(generator) - 1); // Polar angle

            Eigen::Vector3d surface_point;
            surface_point(0) = origin(0) + radius * sin(phi) * cos(theta); // x component
            surface_point(1) = origin(1) + radius * sin(phi) * sin(theta); // y component
            surface_point(2) = origin(2) + radius * cos(phi);

            Eigen::Vector3d direction;
            direction = surface_point - origin;
            direction.normalize();

            Ray ray;
            ray.origin = origin;
            ray.direction = direction;
            ray.index = idx;
            t_rays[ray.index] = ray;
            idx++;
        }
    }
    std::cout << "Number of rays: " << t_rays.size() << std::endl;
}


// Möller–Trumbore intersection algorithm
bool Occlusion::rayTriangleIntersect(Triangle& tr, Ray& ray, Eigen::Vector3d& intersectionPoint) {
    double t, u, v;

    Eigen::Vector3d h, s, q;
    double a, f;

    Eigen::Vector3d e1 = tr.v2 - tr.v1;
    Eigen::Vector3d e2 = tr.v3 - tr.v1;

    h = ray.direction.cross(e2);
    a = e1.dot(h);

    if (a > -1e-8 && a < 1e-8) {
        return false;
    }

    f = 1.0 / a;
    s = ray.origin - tr.v1;
    u = f * (s.dot(h));

    if (u < 0.0 || u > 1.0) {
        return false;
    }

    q = s.cross(e1);
    v = f * ray.direction.dot(q);

    if (v < 0.0 || u + v > 1.0) {
        return false;
    }

    t = f * e2.dot(q);

    if (t > 1e-8) {
        intersectionPoint = ray.origin + ray.direction * t;
        return true;
    }

    return false;
}


bool Occlusion::getRayTriangleIntersectionPt(Triangle& tr, Ray& ray, Eigen::Vector3d& origin, size_t idx, Intersection& intersection) {

    Eigen::Vector3d intersectionPt = Eigen::Vector3d::Zero();
    bool isIntersect = rayTriangleIntersect(tr, ray, intersectionPt);

    if (isIntersect) {
        intersection.point = intersectionPt;
        intersection.index = idx;
        intersection.triangle_index = tr.index;
        intersection.ray_index = ray.index;

        double distance_to_origin = (intersectionPt - origin).norm();
        intersection.distance_to_origin = distance_to_origin;

        tr.intersectionIdx.push_back(idx);
        ray.intersectionIdx.push_back(idx);

        return true;
    } 

    return false;
}


void Occlusion::isFirstHitIntersection(Ray& ray) {
    std::vector<size_t> intersectionIdx = ray.intersectionIdx;
    if (intersectionIdx.size() == 0) {
        return;
    }
    double min_distance = std::numeric_limits<double>::max();
    size_t min_idx = 0;
    for (auto idx : intersectionIdx) {
        t_intersections[idx].is_first_hit = false;
        if (t_intersections[idx].distance_to_origin < min_distance) {
            min_distance = t_intersections[idx].distance_to_origin;
            min_idx = idx;
        }
    }
    t_intersections[min_idx].is_first_hit = true;
}


double Occlusion::triangleBasedOcclusionLevel(Eigen::Vector3d& origin) {    

    size_t intersection_idx = 0;

    for(auto& ray : t_rays) {
        // std::cout << "Ray " << ray.second.index <<" is hitting now" << std::endl;
        for (auto& bbox : t_octree_leaf_bbox) {
            if(rayIntersectLeafBbox(ray.second, bbox)) {
                // std::cout << "Ray " << ray.second.index <<" is hitting " << bbox.triangle_idx.size() << " triangles" << std::endl;
                for(auto& idx : bbox.triangle_idx) {
                    Intersection intersection;
                    if(getRayTriangleIntersectionPt(t_triangles[idx], ray.second, origin, intersection_idx, intersection)) {
                        t_intersections[intersection_idx] = intersection;
                        intersection_idx++;
                    }
                }
            }
        }
        isFirstHitIntersection(ray.second);
    }

    std::cout << "Number of intersections: " << t_intersections.size() << std::endl;

    double occlusion_level = 0.0;
    double total_area = 0.0;
    double total_visible_area = 0.0;

    for (auto& triangle : t_triangles) {
        double area = calculateTriangleArea(triangle.second);
        total_area += area;
        t_triangles[triangle.second.index].area = area;
        size_t num_first_hit_intersections = 0;
        for (auto& idx : triangle.second.intersectionIdx) {
            if (t_intersections[idx].is_first_hit) {
                num_first_hit_intersections++;
            }
        }
        double visible_weight = 0.0;
        if(triangle.second.intersectionIdx.size() > 0){
            size_t num_intersections = triangle.second.intersectionIdx.size();
            visible_weight = (double)num_first_hit_intersections / (double)num_intersections;
        }
        double visible_area = visible_weight * area;
        total_visible_area += visible_area;
        // if (visible_weight > 0.0) {
        //     std::cout << "Visible weight is: " << visible_weight << std::endl;
        // }
    }
    std::cout << "Total area: " << total_area << std::endl;
    std::cout << "Total visible area: " << total_visible_area << std::endl;
    occlusion_level = 1.0 - total_visible_area / total_area;

    return occlusion_level;
}


void Occlusion::traverseOctreeTriangle() {

    float resolution = 64.0;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZI> octree(resolution);
    octree.setInputCloud(t_octree_cloud);
    octree.addPointsFromInputCloud();

    int max_depth = octree.getTreeDepth();
    std::cout << "Max depth: " << max_depth << std::endl;
    int num_leaf_nodes = octree.getLeafCount();
    std::cout << "Total number of leaf nodes: " << num_leaf_nodes << std::endl;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>::LeafNodeIterator it;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>::LeafNodeIterator it_end = octree.leaf_depth_end();

    for (it = octree.leaf_depth_begin(max_depth); it != it_end; ++it) {
        Eigen::Vector3f min_pt, max_pt;

        float size = resolution;
        
        pcl::octree::OctreeKey key = it.getCurrentOctreeKey();

        octree.getVoxelBounds(it, min_pt, max_pt);

        // std::cout << "Min point: " << min_pt.x() << ", " << min_pt.y() << ", " << min_pt.z() << std::endl;

        LeafBBox bbox;
        bbox.min_pt.x() = static_cast<double>(min_pt.x());
        bbox.min_pt.y() = static_cast<double>(min_pt.y());
        bbox.min_pt.z() = static_cast<double>(min_pt.z());
        
        bbox.max_pt.x() = static_cast<double>(max_pt.x());
        bbox.max_pt.y() = static_cast<double>(max_pt.y());
        bbox.max_pt.z() = static_cast<double>(max_pt.z());

        std::vector<int> point_idx = it.getLeafContainer().getPointIndicesVector();
        // std::cout << "Number of points in leaf node: " << point_idx.size() << std::endl;
        for (auto& idx : point_idx) {
            size_t triangle_idx = (size_t) t_octree_cloud->points[idx].intensity;
            bbox.triangle_idx.push_back(triangle_idx);
        }
        t_octree_leaf_bbox.push_back(bbox);
    }
    std::cout << "Number of leaf bbox: " << t_octree_leaf_bbox.size() << std::endl;
}


// build octree from center point of triangles
void Occlusion::buildOctreeCloud() {
    t_octree_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    t_pure_octree_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto& tr : t_triangles) {
        pcl::PointXYZI point;
        
        point.x = tr.second.center(0);
        point.y = tr.second.center(1);
        point.z = tr.second.center(2);
        point.intensity = (float) tr.second.index;

        t_octree_cloud->points.push_back(point); 

        pcl::PointXYZ pure_point;
        pure_point.x = tr.second.center(0);
        pure_point.y = tr.second.center(1);
        pure_point.z = tr.second.center(2);

        t_pure_octree_cloud->points.push_back(pure_point);
    }

    // pcl::PointXYZ min_pt, max_pt;
    // pcl::getMinMax3D(*t_pure_octree_cloud, min_pt, max_pt);

    // oc_cloud_min_pt = Eigen::Vector3d(min_pt.x, min_pt.y, min_pt.z);
    // oc_cloud_max_pt = Eigen::Vector3d(max_pt.x, max_pt.y, max_pt.z);

    std::cout << "Number of points in octree cloud: " << t_octree_cloud->points.size() << std::endl;

    t_pure_octree_cloud->width = t_pure_octree_cloud->points.size();
    t_pure_octree_cloud->height = 1;
    t_pure_octree_cloud->is_dense = true;

    pcl::io::savePCDFileASCII("../files/octree_cloud.pcd", *t_pure_octree_cloud);
}


bool Occlusion::rayIntersectLeafBbox(Ray& ray, LeafBBox& bbox) {
    Eigen::Vector3d origin = ray.origin;
    Eigen::Vector3d direction = ray.direction;

    Eigen::Vector3d min_pt = bbox.min_pt;
    Eigen::Vector3d max_pt = bbox.max_pt;

    if ((origin[0] >= min_pt[0] && origin[0] <= max_pt[0]) &&
        (origin[1] >= min_pt[1] && origin[1] <= max_pt[1]) &&
        (origin[2] >= min_pt[2] && origin[2] <= max_pt[2])) 
    {   
        return true;
    }

    double tmin, tmax, tymin, tymax, tzmin, tzmax;

    if (std::abs(direction[0]) > 1e-8) {
        tmin = (min_pt[0] - origin[0]) / direction[0];
        tmax = (max_pt[0] - origin[0]) / direction[0];
    } else {
        tmin = (min_pt[0] - origin[0]) > 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
        tmax = (max_pt[0] - origin[0]) > 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    }

    if (tmin > tmax) std::swap(tmin, tmax);

    if (std::abs(direction[1]) > 1e-8) {
        tymin = (min_pt[1] - origin[1]) / direction[1];
        tymax = (max_pt[1] - origin[1]) / direction[1];
    } else {
        tymin = (min_pt[1] - origin[1]) > 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
        tymax = (max_pt[1] - origin[1]) > 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    }

    if (tymin > tymax) std::swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax)) {
        return false;
    }

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax) 
        tmax = tymax;

    if (std::abs(direction[2]) > 1e-8) {
        tzmin = (min_pt[2] - origin[2]) / direction[2];
        tzmax = (max_pt[2] - origin[2]) / direction[2];
    } else {
        tzmin = (min_pt[2] - origin[2]) > 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
        tzmax = (max_pt[2] - origin[2]) > 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    }

    if (tzmin > tzmax) std::swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax)) {
        return false;
    }
    return true;
}


// generate cloud from first hit intersection points
void Occlusion::generateCloudFromIntersection() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (auto& intersection : t_intersections) {
        // std::cout << "Intersection index: " << intersection.second.index << std::endl;
        if (intersection.second.is_first_hit) {
            // std::cout << "First hit intersection point: " << intersection.second.index << std::endl;
            pcl::PointXYZ point;
            point.x = intersection.second.point(0);
            point.y = intersection.second.point(1);
            point.z = intersection.second.point(2);
            cloud->points.push_back(point);
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    pcl::io::savePCDFileASCII("../files/mesh_cloud.pcd", *cloud);
    std::cout << "Saved " << cloud->points.size() << " data points to cloud generated from mesh." << std::endl;

}

void Occlusion::generateCloudFromTriangle() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    for(auto& tr : t_triangles) {
        pcl::PointXYZ point;
        point.x = tr.second.v1(0);
        point.y = tr.second.v1(1);
        point.z = tr.second.v1(2);
        cloud->points.push_back(point);

        point.x = tr.second.v2(0);
        point.y = tr.second.v2(1);
        point.z = tr.second.v2(2);
        cloud->points.push_back(point);

        point.x = tr.second.v3(0);
        point.y = tr.second.v3(1);
        point.z = tr.second.v3(2);
        cloud->points.push_back(point);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    pcl::io::savePCDFileASCII("../files/triangle_cloud.pcd", *cloud);
    std::cout << "Saved " << cloud->points.size() << " data points to cloud generated from triangle." << std::endl;
}


