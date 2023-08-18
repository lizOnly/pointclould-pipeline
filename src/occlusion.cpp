#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <thread>
#include <future>
#include <mutex>
#include <unordered_map>

#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

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

#include <omp.h>

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


pcl::PointCloud<pcl::PointXYZRGB>::Ptr Occlusion::centerColoredCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt, std::string file_name) {
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


void Occlusion::regionGrowingSegmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, size_t min_cluster_size, size_t max_cluster_size, int num_neighbours, int k_search_neighbours, double smoothness_threshold, double curvature_threshold) {

    input_cloud = cloud;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud(cloud);
    normal_estimation.setSearchMethod(tree);
    normal_estimation.setKSearch(50);
    normal_estimation.compute(*normals);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(min_cluster_size);
    reg.setMaxClusterSize(max_cluster_size);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(num_neighbours);
    reg.setInputCloud(cloud);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(smoothness_threshold / 180.0 * M_PI);
    reg.setCurvatureThreshold(curvature_threshold);

    reg.extract(rg_clusters);

    std::cout << "Number of clusters is equal to " << rg_clusters.size() << std::endl;

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
    pcl::io::savePCDFileASCII("../files/rg.pcd", *colored_cloud);
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
            triangle.v1 = Eigen::Vector3d(polygon_center[0], polygon_center[1], polygon_center[2]);
            triangle.v2 = Eigen::Vector3d(polygon_cloud->points[i].x, polygon_cloud->points[i].y, polygon_cloud->points[i].z);
            triangle.v3 = Eigen::Vector3d(polygon_cloud->points[(i + 1) % polygon_size].x, polygon_cloud->points[(i + 1) % polygon_size].y, polygon_cloud->points[(i + 1) % polygon_size].z);
            triangle.center = (triangle.v1 + triangle.v2 + triangle.v3) / 3.0;

            double area = calculateTriangleArea(triangle);
            triangle.area = area;
            t_triangles[triangle.index] = triangle;
            tri_idx++;

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


std::vector<pcl::PointXYZ> Occlusion::UniformSamplingSphere(pcl::PointXYZ center, size_t num_samples) {
    
    double radius = 0.1;
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


std::vector<pcl::PointXYZ> Occlusion::generateDefaultPolygon() {
    std::vector<pcl::PointXYZ> default_polygon;

    pcl::PointXYZ default_point;
    default_point.x = 0;
    default_point.y = 0;
    default_point.z = 0;
    pcl::PointXYZ default_point2;
    default_point2.x = 1;
    default_point2.y = 1;
    default_point2.z = 1;
    pcl::PointXYZ default_point3;
    default_point3.x = 1;
    default_point3.y = 0;
    default_point3.z = 1;

    default_polygon.push_back(default_point);
    default_polygon.push_back(default_point2);
    default_polygon.push_back(default_point3);

    return default_polygon;
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

    std::cout << polygons.size() << " polygons" << std::endl;
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


bool Occlusion::rayIntersectSpehre(pcl::PointXYZ& origin, pcl::PointXYZ& direction, pcl::PointXYZ& point, double radius) {
    
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
                if(rayIntersectSpehre(origin, direction, point, point_radius)) {
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

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(octree_resolution);
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


double Occlusion::rayBasedOcclusionLevel(pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt, size_t num_rays_per_vp, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds, std::vector<pcl::ModelCoefficients::Ptr> allCoefficients) {
                                            
    input_cloud = cloud;

    traverseOctree();

    std::vector<pcl::PointXYZ> centers = Occlusion::getSphereLightSourceCenters(min_pt, max_pt);

    double occlusion_level = 0.0;

    size_t num_rays = centers.size() * num_rays_per_vp;
    size_t occlusion_rays = 0;
    size_t polygon_intersec_rays = 0;
    size_t cloud_intersec_rays = 0;


    for (size_t i = 0; i < centers.size(); ++i) {

        // std::cout << "*********Center " << i << ": " << centers[i].x << ", " << centers[i].y << ", " << centers[i].z << "*********" << std::endl;
        std::vector<pcl::PointXYZ> samples = UniformSamplingSphere(centers[i], num_rays_per_vp);

        // iterate over the samples
        for (size_t j = 0; j < samples.size(); ++j) {
            Ray3D ray = generateRay(centers[i], samples[j]);            
            // check if the ray intersects any polygon or point cloud
            if (rayIntersectPointCloud(ray)) {
                // std::cout << "*--> Ray hit cloud!!!" << std::endl;
                cloud_intersec_rays++;

            } else {

                for (size_t k = 0; k < polygonClouds.size(); ++k) {

                    if (Occlusion::rayIntersectPolygon(ray, polygonClouds[k], allCoefficients[k])) {
                        // std::cout << "*--> Ray didn't hit cloud but hit polygon of index " << k << std::endl;
                        polygon_intersec_rays++;
                        break;
                    } else if (k == (polygonClouds.size() - 1)) {
                        // std::cout << "*--> Ray did not hit anything, it's an occlusion" << std::endl;
                        occlusion_rays++;
                    }
                }
            }
        }
    }

    occlusion_level = (double) occlusion_rays / (double) num_rays;
    
    std::cout << "Number of rays: " << num_rays << std::endl;
    std::cout << "Number of cloud intersection rays: " << cloud_intersec_rays << std::endl;
    std::cout << "Number of polygon intersection rays: " << polygon_intersec_rays << std::endl;
    std::cout << "Number of occlusion rays: " << occlusion_rays << std::endl;
    std::cout << "Occlusion level: " << occlusion_level << std::endl;
    
    return occlusion_level;
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
            triangle.center = (triangle.v1 + triangle.v2 + triangle.v3) / 3.0;
            triangle.index = t_idx;
            double area = calculateTriangleArea(triangle);
            triangle.area = area;
            t_triangles[triangle.index] = triangle;
            t_idx++;
        }
    }
    std::cout << "Number of triangles: " << t_triangles.size() << std::endl;
    std::cout << "Minimum vertex: " << min_vertex.transpose() << std::endl;
    std::cout << "Maximum vertex: " << max_vertex.transpose() << std::endl;
    
    file.close();
}


void Occlusion::uniformSampleTriangle(double samples_per_unit_area) {

    std::cout << "Uniformly sampling triangles..." << std::endl;

    size_t idx = 0;
    for (auto& tri : t_triangles) {

        double area = calculateTriangleArea(tri.second);
        size_t num_samples = static_cast<size_t>(area * samples_per_unit_area);

        // if (num_samples == 0) {
        //     num_samples = 1;
        // }

        if(num_samples == 1) {
            
            Sample sample;
            sample.index = idx;
            sample.point = tri.second.center;
            sample.triangle_index = tri.second.index;
            t_samples[sample.index] = sample;
            tri.second.sample_idx.push_back(sample.index);
            idx++;
            continue;

        }

        static std::default_random_engine generator;
        static std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for (size_t i = 0; i < num_samples; ++i) {

            double r1 = distribution(generator);
            double r2 = distribution(generator);

            double sqrtR1 = std::sqrt(r1);
            double alpha = 1 - sqrtR1;
            double beta = r2 * sqrtR1;
            double gamma = 1 - alpha - beta;

            Eigen::Vector3d sample_point = alpha * tri.second.v1 + beta * tri.second.v2 + gamma * tri.second.v3;
            Sample sample;
            sample.index = idx;
            sample.point = sample_point;
            sample.triangle_index = tri.second.index;
            t_samples[sample.index] = sample;
            tri.second.sample_idx.push_back(sample.index);
            idx++;
        }

        
    }
    std::cout << "Number of samples: " << t_samples.size() << std::endl;
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


std::vector<Eigen::Vector3d> Occlusion::viewPointPattern(const int& pattern, Eigen::Vector3d& min, Eigen::Vector3d& max, Eigen::Vector3d& center) {
    std::vector<Eigen::Vector3d> origins;

    Eigen::Vector3d min_mid((min.x() + center.x()) / 2.0, (min.y() + center.y()) / 2.0, (min.z() + center.z()) / 2.0);
    Eigen::Vector3d max_mid((max.x() + center.x()) / 2.0, (max.y() + center.y()) / 2.0, (max.z() + center.z()) / 2.0);
    
    std::cout << "Viewpoint pattern: " << pattern << std::endl;

    if (pattern == 0) {

        origins.push_back(center);

    } else if (pattern == 1) {

        origins.push_back(min_mid);

    } else if (pattern == 2) {

        origins.push_back(max_mid);

    } else if (pattern == 3) {

        origins.push_back(min_mid);
        origins.push_back(max_mid);

    } else if (pattern == 4) {

        origins.push_back(min_mid);
        origins.push_back(center);

    } else if (pattern == 5) {

        origins.push_back(max_mid);
        origins.push_back(center);

    } else if (pattern == 6) {

        origins.push_back(min_mid);
        origins.push_back(max_mid);
        origins.push_back(center);

    }

    return origins;
}


void Occlusion::generateRayFromTriangle(std::vector<Eigen::Vector3d>& origins) {

    std::cout << "Generating rays from triangle to origins..." << std::endl;
    size_t idx = 0;

    for (auto& vp_origin : origins) {
        
        for (auto& tri : t_triangles) {

            if (tri.second.sample_idx.empty()) {
                continue;
            }
            for (auto& sample_idx : tri.second.sample_idx) {
                Ray ray;
                ray.origin = t_samples[sample_idx].point;
                ray.look_at_point = vp_origin;
                ray.direction = vp_origin - t_samples[sample_idx].point;
                ray.direction.normalize();
                ray.index = idx;
                
                ray.source_triangle_index = tri.second.index;
                ray.source_sample_index = sample_idx;
                t_rays[ray.index] = ray;

                tri.second.ray_idx.push_back(ray.index);
                t_samples[sample_idx].ray_idx.push_back(ray.index);
                idx++;

            }
        }
    }
    std::cout << "Number of rays: " << t_rays.size() << std::endl;
}


// generate a ray from the light source to the point on the sphere
void Occlusion::generateRaysWithIdx(std::vector<Eigen::Vector3d>& origins, size_t num_rays_per_vp) {

    std::cout << "Generating rays..." << std::endl;
    size_t idx = 0;
    double radius = 1.0; // radius of the sphere
    // uniform sampling on a sphere which center is the origin
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sphere_light_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (auto& origin : origins) {
        pcl::PointXYZRGB origin_point;
        origin_point.x = origin(0);
        origin_point.y = origin(1);
        origin_point.z = origin(2);
        origin_point.r = 255;
        origin_point.g = 0;
        origin_point.b = 255;

        sphere_light_cloud->points.push_back(origin_point);

        for (size_t i = 0; i < num_rays_per_vp; ++i) {
            pcl::PointXYZRGB point;
            double theta = 2 * M_PI * distribution(generator);  // Azimuthal angle
            double phi = acos(2 * distribution(generator) - 1); // Polar angle

            Eigen::Vector3d surface_point;
            surface_point(0) = origin(0) + radius * sin(phi) * cos(theta); // x component
            surface_point(1) = origin(1) + radius * sin(phi) * sin(theta); // y component
            surface_point(2) = origin(2) + radius * cos(phi);
            
            Eigen::Vector3d direction;
            direction = surface_point - origin;
            direction.normalize();
            
            point.x = surface_point(0);
            point.y = surface_point(1);
            point.z = surface_point(2);
            point.r = 255;
            point.g = 0;
            point.b = 255;

            sphere_light_cloud->points.push_back(point);

            Ray ray;
            ray.origin = origin;
            ray.direction = direction;
            ray.index = idx;
            t_rays[ray.index] = ray;
            idx++;
        }
    }
    sphere_light_cloud->width = sphere_light_cloud->points.size();
    sphere_light_cloud->height = 1;
    sphere_light_cloud->is_dense = true;

    pcl::io::savePCDFileASCII("../files/sphere_light_cloud.pcd", *sphere_light_cloud);

    std::cout << "Number of rays: " << t_rays.size() << std::endl;
}


// Möller–Trumbore intersection algorithm
bool Occlusion::rayTriangleIntersect(Triangle& tr, Ray& ray, Eigen::Vector3d& intersection_point) {
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
        intersection_point = ray.origin + ray.direction * t;
        return true;
    }

    return false;
}


bool Occlusion::getRayTriangleIntersectionPt(Triangle& tr, Ray& ray, size_t idx, Intersection& intersection) {

    Eigen::Vector3d intersection_point = Eigen::Vector3d::Zero();
    bool isIntersect = rayTriangleIntersect(tr, ray, intersection_point);

    if (isIntersect) {
        intersection.point = intersection_point;
        intersection.index = idx;
        intersection.triangle_index = tr.index;
        intersection.ray_index = ray.index;

        double distance_to_look_at_point = (intersection_point - ray.look_at_point).norm();
        intersection.distance_to_look_at_point = distance_to_look_at_point;

        tr.intersection_idx.push_back(idx);
        ray.intersection_idx.push_back(idx);
    } 

    return isIntersect;
}


// void Occlusion::isFirstHitIntersection(Ray& ray) {
//     std::vector<size_t> intersection_idx = ray.intersection_idx;
//     if (intersection_idx.size() == 0) {
//         return;
//     }

//     double min_distance = std::numeric_limits<double>::max();
//     size_t min_idx = 0;

//     for (auto idx : intersection_idx) {
//         t_intersections[idx].is_first_hit = false;
//         if (t_intersections[idx].distance_to_look_at_point < min_distance) {
//             min_distance = t_intersections[idx].distance_to_look_at_point;
//             min_idx = idx;
//         }
//     }

//     t_intersections[min_idx].is_first_hit = true;
// }


double Occlusion::triangleBasedOcclusionLevel(bool enable_acceleration) {    

    size_t intersection_table_size = t_intersections.size();
    size_t intersection_idx = intersection_table_size;

    if (enable_acceleration) {

        std::cout << "Using octree acceleration structure..." << std::endl;

        #pragma omp parallel for
        for(auto& ray : t_rays) {

            // std::cout << "Ray " << ray.second.index <<" is hitting now" << std::endl;
            bool intersectBbox = false;
            int current_idx = intersection_idx;

            for (auto& bbox : t_octree_leaf_bbox_triangle) {

                if(rayIntersectLeafBbox(ray.second, bbox)) {

                    intersectBbox = true;

                    for(auto& idx : bbox.triangle_idx) {

                        if (idx == ray.second.source_triangle_index) {
                            // std::cout << "Ray " << ray.second.index << " is hitting the source triangle" << std::endl;
                            continue;
                        }

                        Intersection intersection;

                        if(getRayTriangleIntersectionPt(t_triangles[idx], ray.second, intersection_idx, intersection)) {
                            
                            t_intersections[intersection_idx] = intersection;
                            intersection_idx++;

                        }
                    }
                }
            }

        }

    } else {

        for (auto& ray : t_rays) {

            for (auto& triangle : t_triangles) {

                if (triangle.second.index == ray.second.source_triangle_index) {
                    continue;
                }

                Intersection intersection;

                if (getRayTriangleIntersectionPt(triangle.second, ray.second, intersection_idx, intersection)) {

                    t_intersections[intersection_idx] = intersection;
                    intersection_idx++;

                }

            }
            // isFirstHitIntersection(ray.second);
        }

    }

    std::cout << "Number of intersections: " << t_intersections.size() << std::endl;

    double occlusion_level = 0.0;
    double total_area = 0.0;
    double total_visible_area = 0.0;

    for (auto& tri : t_triangles) {

        int total_samples = tri.second.sample_idx.size();
        if (total_samples == 0) {
            continue;
        }
        total_area += tri.second.area;
        
        int visible_samples = 0;
        double visible_weight = 0.0;

        // std::cout << "Triangle " << tri.second.index << " has " << total_samples << " samples" << std::endl;

        for (auto& sample_idx : tri.second.sample_idx) {
            
            std::vector<size_t> ray_idx = t_samples[sample_idx].ray_idx;

            for (auto& r_idx : ray_idx) {

                // std::cout << "Ray " << idx << " has " << t_rays[idx].intersection_idx.size() << " intersections" << std::endl;    
                if (t_rays[r_idx].intersection_idx.size() == 0) {
                    // std::cout << "Ray " << r_idx << " has no intersections except samplings on triangle" << std::endl;
                    visible_samples++;
                    t_samples[sample_idx].is_visible = true;
                    break;
                }

                double source_sample_distance_to_look_at_point = (t_rays[r_idx].look_at_point - t_rays[r_idx].origin).norm();
                std::vector<size_t> intersection_idx = t_rays[r_idx].intersection_idx;

                int further_intersection = 0;

                for (auto& i_idx : intersection_idx) {
                    
                    double distance_to_sample =  (t_intersections[i_idx].point - t_rays[r_idx].origin).norm();

                    if (distance_to_sample > source_sample_distance_to_look_at_point) {
                        further_intersection++;
                    } 

                }

                if (further_intersection == intersection_idx.size()) {

                    visible_samples++;
                    t_samples[sample_idx].is_visible = true;
                    break;

                }
            
            }
            
        }        

        visible_weight = (double) visible_samples / (double) total_samples;

        if (std::isnan(visible_weight)) {

            // std::cout << "Visible weight is NaN" << std::endl;
            // std::cout << "Visible samples: " << visible_samples << std::endl;
            // std::cout << "Total samples: " << total_samples << std::endl;
            visible_weight = 0.0;

        }
                    
        double visible_area = visible_weight * tri.second.area;
        total_visible_area += visible_area;

    }

    std::cout << "Total area: " << total_area << std::endl;
    std::cout << "Total visible area: " << total_visible_area << std::endl;
    occlusion_level = 1.0 - total_visible_area / total_area;

    return occlusion_level;
}


void Occlusion::traverseOctreeTriangle() {

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZI> octree(octree_resolution_triangle);
    octree.setInputCloud(t_octree_cloud);
    octree.addPointsFromInputCloud();

    int max_depth = octree.getTreeDepth();
    std::cout << "Max depth: " << max_depth << std::endl;

    int num_leaf_nodes = octree.getLeafCount();
    std::cout << "Total number of leaf nodes: " << num_leaf_nodes << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr t_octree_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>::LeafNodeIterator it;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>::LeafNodeIterator it_end = octree.leaf_depth_end();

    int r = 255;
    int g = 0;
    int b = 255;

    for (it = octree.leaf_depth_begin(max_depth); it != it_end; ++it) {

        r -= 30;
        g += 30;
        b -= 15;

        if (r < 0) {
            r = 250;
            g = 0;
            b = 250;
        }
        Eigen::Vector3f min_pt, max_pt;        
        octree.getVoxelBounds(it, min_pt, max_pt);

        LeafBBox bbox;
        bbox.min_pt.x() = static_cast<double>(min_pt.x());
        bbox.min_pt.y() = static_cast<double>(min_pt.y());
        bbox.min_pt.z() = static_cast<double>(min_pt.z());
        
        bbox.max_pt.x() = static_cast<double>(max_pt.x());
        bbox.max_pt.y() = static_cast<double>(max_pt.y());
        bbox.max_pt.z() = static_cast<double>(max_pt.z());
        
        // colored bounding box of each leaf node
        pcl::PointXYZRGB min_pt_rgb, min_x_rgb, min_y_rgb, min_diag_rgb, max_pt_rgb, max_x_rgb, max_y_rgb, max_diag_rgb;
        min_pt_rgb.x = min_pt.x(); 
        min_pt_rgb.y = min_pt.y(); 
        min_pt_rgb.z = min_pt.z();

        min_pt_rgb.r = r; min_pt_rgb.g = g; min_pt_rgb.b = b;

        min_x_rgb.x = max_pt.x(); 
        min_x_rgb.y = min_pt.y(); 
        min_x_rgb.z = min_pt.z();

        min_x_rgb.r = r; min_x_rgb.g = g; min_x_rgb.b = b;

        min_y_rgb.x = min_pt.x(); 
        min_y_rgb.y = max_pt.y(); 
        min_y_rgb.z = min_pt.z();

        min_y_rgb.r = r; min_y_rgb.g = g; min_y_rgb.b = b;

        min_diag_rgb.x = max_pt.x(); 
        min_diag_rgb.y = max_pt.y(); 
        min_diag_rgb.z = min_pt.z();

        min_diag_rgb.r = r; min_diag_rgb.g = g; min_diag_rgb.b = b;

        max_pt_rgb.x = max_pt.x();
        max_pt_rgb.y = max_pt.y();
        max_pt_rgb.z = max_pt.z();

        max_pt_rgb.r = r; max_pt_rgb.g = g; max_pt_rgb.b = b;

        max_x_rgb.x = min_pt.x();
        max_x_rgb.y = max_pt.y();
        max_x_rgb.z = max_pt.z();

        max_x_rgb.r = r; max_x_rgb.g = g; max_x_rgb.b = b;

        max_y_rgb.x = max_pt.x();
        max_y_rgb.y = min_pt.y();
        max_y_rgb.z = max_pt.z();

        max_y_rgb.r = r; max_y_rgb.g = g; max_y_rgb.b = b;

        max_diag_rgb.x = min_pt.x();
        max_diag_rgb.y = min_pt.y();
        max_diag_rgb.z = max_pt.z();

        max_diag_rgb.r = r; max_diag_rgb.g = g; max_diag_rgb.b = b;

        t_octree_cloud_rgb->points.push_back(min_pt_rgb);
        t_octree_cloud_rgb->points.push_back(max_pt_rgb);
        t_octree_cloud_rgb->points.push_back(min_x_rgb);
        t_octree_cloud_rgb->points.push_back(max_x_rgb);
        t_octree_cloud_rgb->points.push_back(min_y_rgb);
        t_octree_cloud_rgb->points.push_back(max_y_rgb);
        t_octree_cloud_rgb->points.push_back(min_diag_rgb);
        t_octree_cloud_rgb->points.push_back(max_diag_rgb);

        std::vector<int> point_idx = it.getLeafContainer().getPointIndicesVector();

        LeafBBox bbox_triangle;

        Eigen::Vector3d min_pt_triangle = std::numeric_limits<double>::max() * Eigen::Vector3d::Ones();
        Eigen::Vector3d max_pt_triangle = std::numeric_limits<double>::lowest() * Eigen::Vector3d::Ones();

        for (auto& idx : point_idx) {

            size_t triangle_idx = (size_t) t_octree_cloud->points[idx].intensity;

            if (triangle_idx == -1) {
                continue;
            }

            Triangle triangle = t_triangles[triangle_idx];

            min_pt_triangle = min_pt_triangle.cwiseMin(triangle.v1);
            min_pt_triangle = min_pt_triangle.cwiseMin(triangle.v2);
            min_pt_triangle = min_pt_triangle.cwiseMin(triangle.v3);

            max_pt_triangle = max_pt_triangle.cwiseMax(triangle.v1);
            max_pt_triangle = max_pt_triangle.cwiseMax(triangle.v2);
            max_pt_triangle = max_pt_triangle.cwiseMax(triangle.v3);

            bbox_triangle.min_pt = min_pt_triangle;
            bbox_triangle.max_pt = max_pt_triangle;

            // std::cout << "Triangle index: " << triangle_idx << std::endl;
            bbox.triangle_idx.push_back(triangle_idx);
            bbox_triangle.triangle_idx.push_back(triangle_idx);

        }

        t_octree_leaf_bbox.push_back(bbox);
        t_octree_leaf_bbox_triangle.push_back(bbox_triangle);

    }

    t_octree_cloud_rgb->width = t_octree_cloud_rgb->points.size();
    t_octree_cloud_rgb->height = 1;
    t_octree_cloud_rgb->is_dense = true;

    std::cout << "Number of points in octree cloud with bbox: " << t_octree_cloud_rgb->points.size() << std::endl;
    pcl::io::savePCDFileASCII("../files/octree_cloud_rgb.pcd", *t_octree_cloud_rgb);

    std::cout << "Number of leaf bbox: " << t_octree_leaf_bbox.size() << std::endl;
}


// build octree from center point of triangles
void Occlusion::buildOctreeCloud() {
    Eigen::Vector3d min = bbox.min();
    Eigen::Vector3d max = bbox.max();
    
    min.x() -= 1.0;
    min.y() -= 1.0;
    min.z() -= 1.0;

    max.x() += 1.0;
    max.y() += 1.0;
    max.z() += 1.0;

    // make sure the bounding box is slightly larger than the mesh 
    pcl::PointXYZI min_pt, min_x, min_y, min_diag, max_pt, max_x, max_y, max_diag;
    min_pt.x = min.x();
    min_pt.y = min.y();
    min_pt.z = min.z();
    min_pt.intensity = -1.0;

    min_x.x = max.x();
    min_x.y = min.y();
    min_x.z = min.z();
    min_x.intensity = -1.0;

    min_y.x = min.x();
    min_y.y = max.y();
    min_y.z = min.z();
    min_y.intensity = -1.0;

    min_diag.x = max.x();
    min_diag.y = max.y();
    min_diag.z = min.z();
    min_diag.intensity = -1.0;

    max_pt.x = max.x();
    max_pt.y = max.y();
    max_pt.z = max.z();
    max_pt.intensity = -1.0;

    max_x.x = min.x();
    max_x.y = max.y();
    max_x.z = max.z();
    max_x.intensity = -1.0;
    
    max_y.x = max.x();
    max_y.y = min.y();
    max_y.z = max.z();
    max_y.intensity = -1.0;

    max_diag.x = min.x();
    max_diag.y = min.y();
    max_diag.z = max.z();
    max_diag.intensity = -1.0;


    t_octree_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);

    t_octree_cloud->points.push_back(min_pt);
    t_octree_cloud->points.push_back(max_pt);
    t_octree_cloud->points.push_back(min_x);
    t_octree_cloud->points.push_back(max_x);
    t_octree_cloud->points.push_back(min_y);
    t_octree_cloud->points.push_back(max_y);
    t_octree_cloud->points.push_back(min_diag);
    t_octree_cloud->points.push_back(max_diag);

    for (auto& tr : t_triangles) {

        pcl::PointXYZI point;
        
        point.x = tr.second.center(0);
        point.y = tr.second.center(1);
        point.z = tr.second.center(2);
        point.intensity = (float) tr.second.index;

        t_octree_cloud->points.push_back(point); 
    }

    std::cout << "Number of points in octree cloud: " << t_octree_cloud->points.size() << std::endl;

    t_octree_cloud->width = t_octree_cloud->points.size();
    t_octree_cloud->height = 1;
    t_octree_cloud->is_dense = true;

    pcl::io::savePCDFileASCII("../files/octree_cloud.pcd", *t_octree_cloud);
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
        // std::cout << "Ray origin is inside the bbox" << std::endl;
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_first_hit(new pcl::PointCloud<pcl::PointXYZ>);

    for (auto& intersection : t_intersections) {
        // std::cout << "Intersection index: " << intersection.second.index << std::endl;
        pcl::PointXYZ point;
        point.x = intersection.second.point(0);
        point.y = intersection.second.point(1);
        point.z = intersection.second.point(2);
        cloud->points.push_back(point);
        if (intersection.second.is_first_hit) {
            // std::cout << "First hit intersection point: " << intersection.second.index << std::endl;
            pcl::PointXYZ point_first_hit;
            point_first_hit.x = intersection.second.point(0);
            point_first_hit.y = intersection.second.point(1);
            point_first_hit.z = intersection.second.point(2);
            cloud_first_hit->points.push_back(point_first_hit);
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    pcl::io::savePCDFileASCII("../files/intersection_cloud.pcd", *cloud);
    std::cout << "Saved " << cloud->points.size() << " data points to cloud generated from mesh." << std::endl;

    cloud_first_hit->width = cloud_first_hit->points.size();
    cloud_first_hit->height = 1;
    cloud_first_hit->is_dense = true;

    pcl::io::savePCDFileASCII("../files/first_hit_intersection_cloud.pcd", *cloud_first_hit);
    std::cout << "Saved " << cloud_first_hit->points.size() << " data points to first cloud generated from mesh." << std::endl;

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


