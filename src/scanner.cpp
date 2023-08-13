#include <vector>
#include <iostream>
#include <unordered_set>
#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>

#include <pcl/octree/octree_base.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_pointcloud.h>

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "../headers/BaseStruct.h"
#include "../headers/occlusion.h"
#include "../headers/scanner.h"
#include "../headers/property.h"


#define DEG_TO_RAD 0.0174533


Scanner::Scanner()
{
    // empty constructor
}

Scanner::~Scanner()
{
    // empty destructor
}

void Scanner::traverseOctree() {

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


bool Scanner::rayBoxIntersection(const Ray3D& ray, const pcl::PointXYZ& minPt, const pcl::PointXYZ& maxPt) {

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


bool Scanner::rayIntersectSpehre(pcl::PointXYZ& origin, pcl::PointXYZ& direction, pcl::PointXYZ& point) {
    
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


bool Scanner::rayIntersectPointCloud(Ray3D& ray, pcl::PointXYZ& intersection, size_t& index) {

    pcl::PointXYZ origin = ray.origin;
    pcl::PointXYZ direction = ray.direction;

    for(auto& bbox : octree_leaf_bbox) {

        pcl::PointXYZ min_pt(bbox.min_pt.x(), bbox.min_pt.y(), bbox.min_pt.z());
        pcl::PointXYZ max_pt(bbox.max_pt.x(), bbox.max_pt.y(), bbox.max_pt.z());

        if(rayBoxIntersection(ray, min_pt, max_pt)) {
            for(auto& point_idx : bbox.point_idx) {
                pcl::PointXYZ point = input_cloud->points[point_idx];
                if(rayIntersectSpehre(origin, direction, point)) {
                    intersection = point;
                    index = point_idx;
                    return true;
                }
            }
        }
    }

    return false;
}


/*
    This method is used to generate a sampled cloud using ray sampling method. We cast a ray from a light source to a point on the sphere. 
    Then we sample points along the ray with given step, for each point we search for its nearest neighbor within a given search radius.
    The neighbor points are added to the sampled cloud.
*/
pcl::PointCloud<pcl::PointXYZRGB>::Ptr Scanner::sphere_scanner(size_t num_rays_per_vp, int pattern, std::vector<pcl::PointXYZ> scanning_positions, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr gt_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, std::string path) {

    Occlusion occlusion;
    input_cloud = cloud;

    traverseOctree();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sampledCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr sampledCloudGT(new pcl::PointCloud<pcl::PointXYZI>);
    std::cout << "total rays: " << num_rays_per_vp * scanning_positions.size() << std::endl;

    std::unordered_set<int> addedPoints;

    for (int k = 0; k < scanning_positions.size(); k++) {

        std::vector<pcl::PointXYZ> sampledPoints = occlusion.UniformSamplingSphere(scanning_positions[k], num_rays_per_vp);
        std::cout << "scanning position " << k << std::endl;

        // store all index of points that should be added to the sampled cloud
        for (int i = 0; i < sampledPoints.size(); i++) {

            pcl::PointXYZ sampledPoint = sampledPoints[i];
            Ray3D ray = occlusion.generateRay(scanning_positions[k], sampledPoint);

            pcl::PointXYZ intersection;
            std::vector<pcl::PointXYZ> intersections;
            size_t index;
            std::vector<size_t> indices;
            if (rayIntersectPointCloud(ray, intersection, index)) {
                intersections.push_back(intersection);
                indices.push_back(index);
            }

            if (indices.size() == 0) {
                continue;
            }

            double min_distance = std::numeric_limits<double>::max();
            size_t min_idx = 0;
            for (size_t j = 0; j < intersections.size(); j++) {
                double distance = sqrt(pow(intersections[j].x - ray.origin.x, 2) + pow(intersections[j].y - ray.origin.y, 2) + pow(intersections[j].z - ray.origin.z, 2));
                if (distance < min_distance) {
                    min_distance = distance;
                    min_idx = indices[j];
                }
            }
            addedPoints.insert(min_idx);
        }
        
    }

    std::cout << "total points after scanning: " << addedPoints.size() << std::endl;
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree_gt;
    kdtree_gt.setInputCloud(gt_cloud);

    for (const auto& ptIdx : addedPoints) {

        pcl::PointXYZI search_point;
        search_point.x = cloud->points[ptIdx].x;
        search_point.y = cloud->points[ptIdx].y;
        search_point.z = cloud->points[ptIdx].z;
        search_point.intensity = 0.0;

        std::vector<int> indices;
        std::vector<float> distances;

        kdtree_gt.nearestKSearch(search_point, 1, indices, distances);
        pcl::PointXYZI gt_point = gt_cloud->points[indices[0]];
        sampledCloudGT->push_back(gt_point);

        pcl::PointXYZRGB point;

        point.x = cloud->points[ptIdx].x;
        point.y = cloud->points[ptIdx].y;
        point.z = cloud->points[ptIdx].z;

        point.r = coloredCloud->points[ptIdx].r;
        point.g = coloredCloud->points[ptIdx].g;
        point.b = coloredCloud->points[ptIdx].b;

        sampledCloud->push_back(point);

    }

    sampledCloud->width = sampledCloud->size();
    sampledCloud->height = 1;
    sampledCloud->is_dense = true;
    std::cout << "scanned cloud size: " << sampledCloud->size() << std::endl;

    sampledCloudGT->width = sampledCloudGT->size();
    sampledCloudGT->height = 1;
    sampledCloudGT->is_dense = true;
    std::cout << "scanned cloud ground truth size: " << sampledCloudGT->size() << std::endl;

    std::string output_path = path.substr(0, path.length() - 4) + "_" + std::to_string(num_rays_per_vp) + ".pcd";
    std::string gt_output_path = path.substr(0, path.length() - 4) + "_" + std::to_string(num_rays_per_vp) + "_gt.pcd";
    if(pattern == 1) {
        output_path = path.substr(0, path.length() - 4) + "_" + std::to_string(num_rays_per_vp) + "_v1.pcd";
        gt_output_path = path.substr(0, path.length() - 4) + "_" + std::to_string(num_rays_per_vp) + "_v1_gt.pcd";
    } else if (pattern == 2) {
        output_path = path.substr(0, path.length() - 4) + "_" + std::to_string(num_rays_per_vp) + "_v2.pcd";
        gt_output_path = path.substr(0, path.length() - 4) + "_" + std::to_string(num_rays_per_vp) + "_v2_gt.pcd";
    }
    pcl::io::savePCDFileASCII (output_path, *sampledCloud);
    pcl::io::savePCDFileASCII (gt_output_path, *sampledCloudGT);

    return sampledCloud;

}



std::vector<pcl::PointXYZ> Scanner::scanning_positions(pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt, int pattern) {

    Occlusion occlusion;

    std::vector<pcl::PointXYZ> positions;
    
    pcl::PointXYZ center;
    center.x = (min_pt.x + max_pt.x) / 2;
    center.y = (min_pt.y + max_pt.y) / 2;
    center.z = (min_pt.z + max_pt.z) / 2;

    if (pattern == 0) { // one center position

        positions.push_back(center);
        
    } else if (pattern == 1) {
        pcl::PointXYZ max_position;
        max_position.x = (center.x + max_pt.x) / 2;
        max_position.y = (center.y + max_pt.y) / 2;
        max_position.z = (center.z + max_pt.z) / 2;

        positions.push_back(max_position); 

    } else if (pattern == 2) {
        pcl::PointXYZ min_position;
        min_position.x = (center.x + min_pt.x) / 2;
        min_position.y = (center.y + min_pt.y) / 2;
        min_position.z = (center.z + min_pt.z) / 2;

        positions.push_back(min_position);
    }  else if (pattern == 3) {
        positions = occlusion.getSphereLightSourceCenters(min_pt, max_pt);
    }

    return positions;

}


std::vector<pcl::PointXYZ> Scanner::sample_square_points(const pcl::PointXYZ& scanner_position, int sample_step, double distance, double angle) {

    double half_distance = distance / 2;
    std::vector<pcl::PointXYZ> points;

    pcl::PointXYZ square_center; // looking at the center of the square
    square_center.x = scanner_position.x + distance * cos(angle * DEG_TO_RAD);
    square_center.y = scanner_position.y + distance * sin(angle * DEG_TO_RAD);
    square_center.z = scanner_position.z;

    double step = half_distance / (double)sample_step; // 5 points

    for (int i = 0; i < sample_step + 1; i++) {
        
        if (i == 0) {
            points.push_back(square_center);

            for (int j = 1; j < sample_step + 1; j++) {

                pcl::PointXYZ center_up;
                center_up.x = square_center.x;
                center_up.y = square_center.y;
                center_up.z = square_center.z + step * j;

                points.push_back(center_up);

                pcl::PointXYZ center_down;

                center_down.x = square_center.x;
                center_down.y = square_center.y;
                center_down.z = square_center.z - step * j;

                points.push_back(center_down);

            }
            continue;
        }

        pcl::PointXYZ point_hor_left;
        point_hor_left.x = square_center.x + step * i * sin(angle * DEG_TO_RAD);
        point_hor_left.y = square_center.y - step * i * cos(angle * DEG_TO_RAD);
        point_hor_left.z = square_center.z;

        points.push_back(point_hor_left);

        pcl::PointXYZ point_hor_right;
        point_hor_right.x = square_center.x - step * i * sin(angle * DEG_TO_RAD);
        point_hor_right.y = square_center.y + step * i * cos(angle * DEG_TO_RAD);
        point_hor_right.z = square_center.z;

        points.push_back(point_hor_right);

        for (int j = 1; j < sample_step + 1; j++) {

            pcl::PointXYZ point_left_vert_up;
            point_left_vert_up.x = point_hor_left.x;
            point_left_vert_up.y = point_hor_left.y;
            point_left_vert_up.z = point_hor_left.z + step * j;
            points.push_back(point_left_vert_up);

            pcl::PointXYZ point_left_vert_down;
            point_left_vert_down.x = point_hor_left.x;
            point_left_vert_down.y = point_hor_left.y;
            point_left_vert_down.z = point_hor_left.z - step * j;
            points.push_back(point_left_vert_down);


            pcl::PointXYZ point_right_vert_up;
            point_right_vert_up.x = point_hor_right.x;
            point_right_vert_up.y = point_hor_right.y;
            point_right_vert_up.z = point_hor_right.z + step * j;
            points.push_back(point_right_vert_up);

            pcl::PointXYZ point_right_vert_down;
            point_right_vert_down.x = point_hor_right.x;
            point_right_vert_down.y = point_hor_right.y;
            point_right_vert_down.z = point_hor_right.z - step * j;
            points.push_back(point_right_vert_down);

        }
    }
    
    std::cout << points.size() << " points generated for a square"<< std::endl;

    return points;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr Scanner::multi_square_scanner(double step, double searchRadius, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt,
                                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, std::string file_name)
{
    Occlusion occlusion;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scanned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    std::vector<pcl::PointXYZ> scanner_positions = scanning_positions(minPt, maxPt, 0); // generate fixed scanners
    std::cout << "total scanners: " << scanner_positions.size() << std::endl;

    std::unordered_set<int> addedPoints;

    for (int i = 0; i < scanner_positions.size(); i++) {

        pcl::PointXYZ scanner_position = scanner_positions[i];
        for (double angle = 0.0; angle <= 360.0; angle += 10.0) {
            std::vector<pcl::PointXYZ> points = sample_square_points(scanner_position, 10, 0.1, angle);

            for (int j = 0; j < points.size(); j++) {

                pcl::PointXYZ point = points[j];
                Ray3D ray = occlusion.generateRay(scanner_position, point);
                
                while ( point.x < maxPt.x && point.y < maxPt.y && point.z < maxPt.z && point.x > minPt.x && point.y > minPt.y && point.z > minPt.z) {
                    
                    std::vector<int> pointIdxRadiusSearch;
                    std::vector<float> pointRadiusSquaredDistance;
                    
                    if ( kdtree.radiusSearch(point, searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
                
                        addedPoints.insert(pointIdxRadiusSearch[0]);
                        break;
                        
                    }
                    
                    point.x += step * ray.direction.x;
                    point.y += step * ray.direction.y;
                    point.z += step * ray.direction.z;
                    
                }
            }
        }
    }    

    std::cout << "total points after scanning: " << addedPoints.size() << std::endl;

    for (const auto& ptIdx : addedPoints) {

        pcl::PointXYZRGB point;

        point.x = coloredCloud->points[ptIdx].x;
        point.y = coloredCloud->points[ptIdx].y;
        point.z = coloredCloud->points[ptIdx].z;

        point.r = coloredCloud->points[ptIdx].r;
        point.g = coloredCloud->points[ptIdx].g;
        point.b = coloredCloud->points[ptIdx].b;

        scanned_cloud->push_back(point);

    }                            

    scanned_cloud->width = scanned_cloud->size();
    scanned_cloud->height = 1;
    scanned_cloud->is_dense = true;

    std::string outputPath = "../files/scanned_" + file_name.substr(0, file_name.length() - 4) + ".pcd";
    pcl::io::savePCDFileASCII (outputPath, *scanned_cloud); 

    return scanned_cloud;   
}

std::vector<pcl::PointXYZ> random_look_at_direction(int num_directions, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt) {

    std::vector<pcl::PointXYZ> points;

    for (int i = 0; i < num_directions; i++) {

        pcl::PointXYZ point;
        point.x = minPt.x + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxPt.x-minPt.x)));
        point.y = minPt.y + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxPt.y-minPt.y)));
        point.z = minPt.z + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxPt.z-minPt.z)));

        points.push_back(point);

    }

    return points;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr Scanner::random_scanner(double step,
                                                                double searchRadius, // search radius
                                                                size_t num_random_positions,
                                                                pcl::PointXYZ& minPt, 
                                                                pcl::PointXYZ& maxPt,
                                                                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
                                                                std::string file_name)
{

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scanned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    Occlusion occlusion;

    std::vector<pcl::PointXYZ> scanner_positions = scanning_positions(minPt, maxPt, 1); // generate random scanners

    std::unordered_set<int> addedPoints;

    for (size_t i = 0; i < scanner_positions.size(); i++) {
        
        pcl::PointXYZ scanner_position = scanner_positions[i];
        std::vector<pcl::PointXYZ> look_at_directions = random_look_at_direction(10, minPt, maxPt);

        for (size_t j = 0; j < look_at_directions.size(); j++) {

            pcl::PointXYZ look_at_direction = look_at_directions[j];
            Ray3D ray = occlusion.generateRay(scanner_position, look_at_direction);

            pcl::PointXYZ point = scanner_position;
            while ( point.x < maxPt.x && point.y < maxPt.y && point.z < maxPt.z && point.x > minPt.x && point.y > minPt.y && point.z > minPt.z) {
                
                std::vector<int> pointIdxRadiusSearch;
                std::vector<float> pointRadiusSquaredDistance;
                
                if ( kdtree.radiusSearch(point, searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
            
                    addedPoints.insert(pointIdxRadiusSearch[0]);
                    break;
                }
                
                point.x += step * ray.direction.x;
                point.y += step * ray.direction.y;
                point.z += step * ray.direction.z;
                
            }
        }
    }

    std::cout << "total points after scanning: " << addedPoints.size() << std::endl;

    for (const auto& ptIdx : addedPoints) {

        pcl::PointXYZRGB point;

        point.x = coloredCloud->points[ptIdx].x;
        point.y = coloredCloud->points[ptIdx].y;
        point.z = coloredCloud->points[ptIdx].z;

        point.r = coloredCloud->points[ptIdx].r;
        point.g = coloredCloud->points[ptIdx].g;
        point.b = coloredCloud->points[ptIdx].b;

        scanned_cloud->push_back(point);

    }

    scanned_cloud->width = scanned_cloud->size();
    scanned_cloud->height = 1;
    scanned_cloud->is_dense = true;

    std::string outputPath = "../files/random_scanned_" + file_name.substr(0, file_name.length() - 4) + "-" + std::to_string(num_random_positions) + ".pcd";
    pcl::io::savePCDFileASCII (outputPath, *scanned_cloud);

    return scanned_cloud;
}

