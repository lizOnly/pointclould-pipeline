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


void Scanner::buildCompleteOctreeNodes() {

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(octree_resolution);
    octree.setInputCloud(input_cloud);
    octree.addPointsFromInputCloud();

    int max_depth = octree.getTreeDepth();
    std::cout << "Max depth: " << max_depth << std::endl;

    std::unordered_map<int, int> depth_map; // <diagonal distance of bounding box, depth of curren level>
    std::unordered_map<int, std::vector<size_t>> depth_index_map; // <depth, index of nodes at current depth>
    // depth first traversal

    size_t idx = 0;

    pcl::PointCloud<pcl::PointXYZ>::Ptr octree_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (auto it = octree.begin(); it != octree.end(); ++it) {

        OctreeNode node;
        node.index = idx;

        Eigen::Vector3f min_pt, max_pt;
        octree.getVoxelBounds(it, min_pt, max_pt);

        node.min_pt.x() = static_cast<double>(min_pt.x());
        node.min_pt.y() = static_cast<double>(min_pt.y());
        node.min_pt.z() = static_cast<double>(min_pt.z());

        node.max_pt.x() = static_cast<double>(max_pt.x());
        node.max_pt.y() = static_cast<double>(max_pt.y());
        node.max_pt.z() = static_cast<double>(max_pt.z());

        float d = (max_pt-min_pt).norm();
        int d_int = static_cast<int>(d);
        node.diagonal_distance = d_int;

        depth_map[d_int] = 0;
        
        if (it.isBranchNode()) {
            node.is_branch = true;
            // std::cout << "Branch Distance: " << d << std::endl;
        }

        if (it.isLeafNode()) {
            node.is_leaf = true;
            // std::cout << "Leaf Distance: " << d << std::endl;

            std::vector<int> point_idx = it.getLeafContainer().getPointIndicesVector();

            for (auto& idx : point_idx) {
                node.point_idx.push_back(idx);
            }

        }

        t_octree_nodes[node.index] = node;
        idx++;
    }

    std::cout << "Number of octree nodes: " << t_octree_nodes.size() << std::endl;
    std::cout << "" << std::endl;
    t_octree_nodes[0].is_root = true;

    std::set<int> size_set;
    for (auto& d : depth_map) {
        size_set.insert(d.first);
    }
    
    if (size_set.size() != depth_map.size()) {
        std::cout << "Wrong depth map created !!!" << std::endl;
        return;
    } else {
        for (auto& s : size_set) {
            std::cout << "Diagonal distance: " << s << std::endl;
        }
    }

    std::cout << "" << std::endl;

    for (auto& d : depth_map) {
        int i = max_depth;
        for (auto& s : size_set) {
            if (d.first == s) {
                depth_map[d.first] = i;
            }
            i--;
        }
        std::cout << "Diagonal distance: " << d.first << " in depth: " << depth_map[d.first] << std::endl;
    }

    std::cout << "" << std::endl;

    if (depth_map.size() != max_depth + 1) {
        std::cout << "Wrong depth map created !!!" << std::endl;
        return;
    }

    for (size_t i = 0; i < t_octree_nodes.size(); ++i) {

        t_octree_nodes[i].depth = depth_map[t_octree_nodes[i].diagonal_distance];
        depth_index_map[t_octree_nodes[i].depth].push_back(t_octree_nodes[i].index);

    }

    // build connections between nodes 
    for (int d = 0; d < max_depth; d++) {
        std::cout << "Depth: " << d << " has " << depth_index_map[d].size() << " nodes" <<std::endl;

        for (auto& idx : depth_index_map[d]) {
            
            if (t_octree_nodes[idx].prev == -1) {
                
                if (t_octree_nodes[idx].next == -1) {

                    for (int i = 0; i < depth_index_map[d+1].size(); ++i) {

                        if (t_octree_nodes[depth_index_map[d+1][i]].parent_index != -1) {
                            continue;
                        }

                        t_octree_nodes[idx].children.push_back(depth_index_map[d+1][i]);
                        t_octree_nodes[depth_index_map[d+1][i]].parent_index = idx;

                        if ((i + 1) < depth_index_map[d+1].size()) {

                            t_octree_nodes[depth_index_map[d+1][i]].next = depth_index_map[d+1][i+1];
                            t_octree_nodes[depth_index_map[d+1][i+1]].prev = depth_index_map[d+1][i];

                        } 
                    }

                } else {

                    for (int i = 0; i < depth_index_map[d+1].size(); ++i) {
                        
                        if (depth_index_map[d+1][i] < t_octree_nodes[idx].next) {
                            
                            if (t_octree_nodes[depth_index_map[d+1][i]].parent_index != -1) {
                                continue;
                            }

                            t_octree_nodes[idx].children.push_back(depth_index_map[d+1][i]);
                            t_octree_nodes[depth_index_map[d+1][i]].parent_index = idx;

                            if ((i + 1) < depth_index_map[d+1].size()) {

                                t_octree_nodes[depth_index_map[d+1][i]].next = depth_index_map[d+1][i+1];
                                t_octree_nodes[depth_index_map[d+1][i+1]].prev = depth_index_map[d+1][i];

                            } 
                        }
                    }

                }
            } else if (t_octree_nodes[idx].next == -1) {

                for (int i = 0; i < depth_index_map[d+1].size(); ++i) {
                        
                    if (depth_index_map[d+1][i] > t_octree_nodes[idx].prev) {
                        
                        if (t_octree_nodes[depth_index_map[d+1][i]].parent_index != -1) {
                            continue;
                        }

                        t_octree_nodes[idx].children.push_back(depth_index_map[d+1][i]);
                        t_octree_nodes[depth_index_map[d+1][i]].parent_index = idx;

                        if ((i + 1) < depth_index_map[d+1].size()) {

                            t_octree_nodes[depth_index_map[d+1][i]].next = depth_index_map[d+1][i+1];
                            t_octree_nodes[depth_index_map[d+1][i+1]].prev = depth_index_map[d+1][i];

                        } 
                    }
                }

            } else {

                for (int i = 0; i < depth_index_map[d+1].size(); ++i) {
                        
                    if (depth_index_map[d+1][i] > t_octree_nodes[idx].prev && depth_index_map[d+1][i] < t_octree_nodes[idx].next) {
                        
                        if (t_octree_nodes[depth_index_map[d+1][i]].parent_index != -1) {
                            continue;
                        }

                        t_octree_nodes[idx].children.push_back(depth_index_map[d+1][i]);
                        t_octree_nodes[depth_index_map[d+1][i]].parent_index = idx;

                        if ((i + 1) < depth_index_map[d+1].size()) {

                            t_octree_nodes[depth_index_map[d+1][i]].next = depth_index_map[d+1][i+1];
                            t_octree_nodes[depth_index_map[d+1][i+1]].prev = depth_index_map[d+1][i];

                        } 
                    }
                }

            }
        }

    }

    std::cout << "" << std::endl;
    std::cout << "Octree built! Number of octree nodes: " << t_octree_nodes.size() << std::endl;
    std::cout << "" << std::endl;

    for (int d = 0; d < max_depth; d++) {
        
        size_t num_children = 0;
        
        for (auto& idx : depth_index_map[d]) {
            
            num_children += t_octree_nodes[idx].children.size();
        
        }

        std::cout << "Depth: " << d << " has " << depth_index_map[d].size() << " nodes and " << num_children << " children" <<std::endl;
    }

    std::cout << "" << std::endl;

}


std::vector<pcl::PointXYZ> Scanner::fixed_scanning_positions(pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt, int pattern) {

    Occlusion occlusion;

    std::vector<pcl::PointXYZ> positions;
    
    pcl::PointXYZ center;
    center.x = (min_pt.x + max_pt.x) / 2;
    center.y = (min_pt.y + max_pt.y) / 2;
    center.z = (min_pt.z + max_pt.z) / 2;

    pcl::PointXYZ max_position;
    max_position.x = (center.x + max_pt.x) / 2;
    max_position.y = (center.y + max_pt.y) / 2;
    max_position.z = (center.z + max_pt.z) / 2;

    pcl::PointXYZ min_position;
    min_position.x = (center.x + min_pt.x) / 2;
    min_position.y = (center.y + min_pt.y) / 2;
    min_position.z = (center.z + min_pt.z) / 2;

    if (pattern == 0) { // one center position

        positions.push_back(center);
        
    } else if (pattern == 1) {

        positions.push_back(min_position); 

    } else if (pattern == 2) {

        positions.push_back(max_position);

    }  else if (pattern == 3) {

        positions.push_back(min_position);
        positions.push_back(max_position);

    } else if (pattern == 4) {

        positions.push_back(center);
        positions.push_back(min_position);

    } else if (pattern == 5) {

        positions.push_back(center);
        positions.push_back(max_position);

    } else if (pattern == 6) {

        positions.push_back(center);
        positions.push_back(min_position);
        positions.push_back(max_position);

    } else if (pattern == 7) { // extreme case

        positions.push_back(min_pt);

    } else if (pattern == 8) { // extreme case

        positions.push_back(max_pt);

    } else if (pattern == 9) {


    }


    return positions;

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


bool Scanner::rayBoxIntersection(const Ray3D& ray, const pcl::PointXYZ& min_pt, const pcl::PointXYZ& max_pt) {

    if(ray.origin.x >= min_pt.x && ray.origin.x <= max_pt.x &&
       ray.origin.y >= min_pt.y && ray.origin.y <= max_pt.y &&
       ray.origin.z >= min_pt.z && ray.origin.z <= max_pt.z) {
        return true;
    }
    
    double tmin, tmax, tymin, tymax, tzmin, tzmax;

    if (ray.direction.x != 0) {
        if (ray.direction.x >= 0) {
            tmin = (min_pt.x - ray.origin.x) / ray.direction.x;
            tmax = (max_pt.x - ray.origin.x) / ray.direction.x;
        } else {
            tmin = (max_pt.x - ray.origin.x) / ray.direction.x;
            tmax = (min_pt.x - ray.origin.x) / ray.direction.x;
        }
    } else {
        if (ray.origin.x < min_pt.x || ray.origin.x > max_pt.x) {
            return false;
        }
        tmin = std::numeric_limits<double>::lowest();
        tmax = std::numeric_limits<double>::max();
    }

    if (ray.direction.y != 0) {
        if (ray.direction.y >= 0) {
            tymin = (min_pt.y - ray.origin.y) / ray.direction.y;
            tymax = (max_pt.y - ray.origin.y) / ray.direction.y;
        } else {
            tymin = (max_pt.y - ray.origin.y) / ray.direction.y;
            tymax = (min_pt.y - ray.origin.y) / ray.direction.y;
        }
    } else {
        if (ray.origin.y < min_pt.y || ray.origin.y > max_pt.y) {
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
            tzmin = (min_pt.z - ray.origin.z) / ray.direction.z;
            tzmax = (max_pt.z - ray.origin.z) / ray.direction.z;
        } else {
            tzmin = (max_pt.z - ray.origin.z) / ray.direction.z;
            tzmax = (min_pt.z - ray.origin.z) / ray.direction.z;
        }
    } else {
        if (ray.origin.z < min_pt.z || ray.origin.z > max_pt.z) {
            return false;
        }
        tzmin = std::numeric_limits<double>::lowest();
        tzmax = std::numeric_limits<double>::max();
    }

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    return true;
}


void Scanner::checkFirstHitPoint(Ray3D& ray) {

    double min_distance = std::numeric_limits<double>::max();
    size_t min_idx = -1;

    for (auto& idx : ray.intersection_idx) {

        pcl::PointXYZ point = input_cloud->points[idx];

        double distance = sqrt(pow(point.x - ray.origin.x, 2) + pow(point.y - ray.origin.y, 2) + pow(point.z - ray.origin.z, 2));
        
        if (distance < min_distance) {
            min_distance = distance;
            min_idx = idx;
        }

    }

    ray.first_hit_point = min_idx;

}

void Scanner::checkRayOctreeIntersection(Ray3D& ray, OctreeNode& node) {

    if (node.is_leaf) {

        // std::cout << "Ray: " << ray.index << " intersects with leaf node: " << node.index << std::endl;
        // std::cout << "Number of points in leaf node: " << node.point_idx.size() << std::endl;

        for (auto& point_idx : node.point_idx) {
            // std::cout << "Check hitting point: " << point_idx << std::endl;
            pcl::PointXYZ point = input_cloud->points[point_idx];
            
            if (rayIntersectSpehre(ray.origin, ray.direction, point)) {
                
                // std::cout << "Ray: " << ray.index << " intersects with point: " << point_idx << std::endl;
                ray.intersection_idx.push_back(point_idx);

            }

        }

        
    } else {

        for (auto& child_idx : node.children) {
            
            pcl::PointXYZ min_pt(node.min_pt.x(), node.min_pt.y(), node.min_pt.z());
            pcl::PointXYZ max_pt(node.max_pt.x(), node.max_pt.y(), node.max_pt.z());

            if (rayBoxIntersection(ray, min_pt, max_pt)) {
                checkRayOctreeIntersection(ray, t_octree_nodes[child_idx]);
            }

        }

    }
}


bool Scanner::rayIntersectSpehre(pcl::PointXYZ& origin, pcl::PointXYZ& direction, pcl::PointXYZ& point) {
    
   double dirMagnitude = sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    direction.x /= dirMagnitude;
    direction.y /= dirMagnitude;
    direction.z /= dirMagnitude;

    pcl::PointXYZ L(point.x - origin.x, point.y - origin.y, point.z - origin.z);

    double originDistance2 = L.x * L.x + L.y * L.y + L.z * L.z;
    if (originDistance2 < point_radius * point_radius) return true;  // origin is inside the sphere

    double t_ca = L.x * direction.x + L.y * direction.y + L.z * direction.z;

    if (t_ca < 0) return false;

    double d2 = originDistance2 - t_ca * t_ca;

    if (d2 > point_radius * point_radius) return false;


    return true;

}


void Scanner::generateRays(size_t num_rays_per_vp, std::vector<pcl::PointXYZ> origins) {

    std::cout << "Generating rays ..." << std::endl;

    double radius = 0.2;
    size_t ray_idx = 0;

    Occlusion occlusion;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (size_t i = 0; i < origins.size(); i++) {

        pcl::PointXYZ origin = origins[i];

        for (size_t j = 0; j < num_rays_per_vp; j++) {

            double theta = 2 * M_PI * occlusion.halton(j, 2); 
            double phi = acos(2 * occlusion.halton(j, 3) - 1);

            double x = origin.x + radius * sin(phi) * cos(theta);
            double y = origin.y + radius * sin(phi) * sin(theta);
            double z = origin.z + radius * cos(phi);

            pcl::PointXYZ look_at(x, y, z);

            cloud->push_back(look_at);

            Ray3D ray;
            ray.origin = origin;
            ray.direction.x = look_at.x - origin.x;
            ray.direction.y = look_at.y - origin.y;
            ray.direction.z = look_at.z - origin.z;
            ray.index = ray_idx;
            t_rays[ray.index] = ray;
            ray_idx++;
        }

    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    pcl::io::savePCDFileASCII ("../files/sphere_scanners.pcd", *cloud);

    std::cout << "" << std::endl;
    std::cout << "Number of rays: " << t_rays.size() << std::endl;
    std::cout << "" << std::endl;
}

/*
    This method is used to generate a sampled cloud using ray sampling method. We cast a ray from a light source to a point on the sphere. 
    Then we sample points along the ray with given step, for each point we search for its nearest neighbor within a given search radius.
    The neighbor points are added to the sampled cloud.
*/
void Scanner::sphere_scanner(int pattern, std::string scene_name) {

    Occlusion occlusion;

    pcl::PointCloud<pcl::PointXYZI>::Ptr scanned_cloud_gt(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr scanned_cloud_bound(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scanned_cloud_bound_color(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scanned_cloud_color(new pcl::PointCloud<pcl::PointXYZRGB>);

    size_t clutter_count = 0;

    for (auto& ray : t_rays) {

        OctreeNode root = t_octree_nodes[0];
        checkRayOctreeIntersection(ray.second, root);
        checkFirstHitPoint(ray.second);

        pcl::PointXYZI point_gt;
        pcl::PointXYZI point_bound;
        pcl::PointXYZRGB point_bound_color;
        pcl::PointXYZRGB point_color;

        if (ray.second.first_hit_point != -1) {
            
            // std::cout << "Ray: " << ray.second.index << " first hit on point " << ray.second.first_hit_point << std::endl;
            size_t idx = ray.second.first_hit_point;
            
            point_gt.x = input_cloud->points[idx].x;
            point_gt.y = input_cloud->points[idx].y;
            point_gt.z = input_cloud->points[idx].z;

            // std::cout << "Point: " << point_gt.x << " " << point_gt.y << " " << point_gt.z << std::endl;
            point_gt.intensity = input_cloud_gt->points[idx].intensity;

            point_bound.x = input_cloud->points[idx].x;
            point_bound.y = input_cloud->points[idx].y;
            point_bound.z = input_cloud->points[idx].z;

            point_bound_color.x = input_cloud->points[idx].x;
            point_bound_color.y = input_cloud->points[idx].y;
            point_bound_color.z = input_cloud->points[idx].z;

            if (point_gt.intensity == 0 || point_gt.intensity == 1 || point_gt.intensity == 7 ) {
                
                point_bound.intensity = 1;
                point_bound_color.r = 227;
                point_bound_color.g = 221;
                point_bound_color.b = 220;

            } else {

                point_bound.intensity = 0;
                point_bound_color.r = 40;
                point_bound_color.g = 126;
                point_bound_color.b = 166;

                clutter_count++;
            
            }

            point_color.x = input_cloud->points[idx].x;
            point_color.y = input_cloud->points[idx].y;
            point_color.z = input_cloud->points[idx].z;

            point_color.r = input_cloud_color->points[idx].r;
            point_color.g = input_cloud_color->points[idx].g;
            point_color.b = input_cloud_color->points[idx].b;

            scanned_cloud_gt->push_back(point_gt);
            scanned_cloud_color->push_back(point_color);

            scanned_cloud_bound->push_back(point_bound);
            scanned_cloud_bound_color->push_back(point_bound_color);

        }
    }

    scanned_cloud_gt->width = scanned_cloud_gt->size();
    scanned_cloud_gt->height = 1;
    scanned_cloud_gt->is_dense = true;

    scanned_cloud_color->width = scanned_cloud_color->size();
    scanned_cloud_color->height = 1;
    scanned_cloud_color->is_dense = true;

    scanned_cloud_bound->width = scanned_cloud_bound->size();
    scanned_cloud_bound->height = 1;
    scanned_cloud_bound->is_dense = true;

    scanned_cloud_bound_color->width = scanned_cloud_bound_color->size();
    scanned_cloud_bound_color->height = 1;
    scanned_cloud_bound_color->is_dense = true;

    pcl::io::savePCDFileASCII ("../files/" + scene_name + "_gt_" + std::to_string(pattern) + ".pcd", *scanned_cloud_gt);
    std::cout << "Saved " << scanned_cloud_gt->size() << " data points to gt cloud" << std::endl;
    std::cout << "" << std::endl;

    pcl::io::savePCDFileASCII ("../files/" + scene_name + "_color_" + std::to_string(pattern) + ".pcd", *scanned_cloud_color);
    std::cout << "Saved " << scanned_cloud_color->size() << " data points to color cloud" << std::endl;
    std::cout << "" << std::endl;

    pcl::io::savePCDFileASCII ("../files/" + scene_name + "_bound_" + std::to_string(pattern) + ".pcd", *scanned_cloud_bound);
    std::cout << "Saved " << scanned_cloud_bound->size() << " data points to bound cloud" << std::endl;
    std::cout << "Number of clutter points: " << clutter_count << std::endl;
    std::cout << "Clutter ratio is: " << (double)clutter_count / (double)scanned_cloud_bound->size() << std::endl;
    std::cout << "" << std::endl;

    pcl::io::savePCDFileASCII ("../files/" + scene_name + "_bound_color_" + std::to_string(pattern) + ".pcd", *scanned_cloud_bound_color);
    std::cout << "Saved " << scanned_cloud_bound_color->size() << " data points to bound color cloud" << std::endl;
    std::cout << "" << std::endl;

}

// random scanning positions

std::vector<pcl::PointXYZ> Scanner::random_scanning_positions(pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt, int num_scanners) {

    std::vector<pcl::PointXYZ> positions;

    for (int i = 0; i < num_scanners; i++) {

        pcl::PointXYZ position;
        position.x = min_pt.x + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_pt.x-min_pt.x)));
        position.y = min_pt.y + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_pt.y-min_pt.y)));
        position.z = min_pt.z + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_pt.z-min_pt.z)));

        positions.push_back(position);

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


pcl::PointCloud<pcl::PointXYZRGB>::Ptr Scanner::multi_square_scanner(double step, double searchRadius, pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt,
                                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, std::string file_name)
{
    Occlusion occlusion;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scanned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    std::vector<pcl::PointXYZ> scanner_positions = fixed_scanning_positions(min_pt, max_pt, 0); // generate fixed scanners
    std::cout << "total scanners: " << scanner_positions.size() << std::endl;

    std::unordered_set<int> addedPoints;

    for (int i = 0; i < scanner_positions.size(); i++) {

        pcl::PointXYZ scanner_position = scanner_positions[i];
        for (double angle = 0.0; angle <= 360.0; angle += 10.0) {
            std::vector<pcl::PointXYZ> points = sample_square_points(scanner_position, 10, 0.1, angle);

            for (int j = 0; j < points.size(); j++) {

                pcl::PointXYZ point = points[j];
                Ray3D ray = occlusion.generateRay(scanner_position, point);
                
                while ( point.x < max_pt.x && point.y < max_pt.y && point.z < max_pt.z && point.x > min_pt.x && point.y > min_pt.y && point.z > min_pt.z) {
                    
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


std::vector<pcl::PointXYZ> random_look_at_direction(int num_directions, pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt) {

    std::vector<pcl::PointXYZ> points;

    for (int i = 0; i < num_directions; i++) {

        pcl::PointXYZ point;
        point.x = min_pt.x + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_pt.x-min_pt.x)));
        point.y = min_pt.y + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_pt.y-min_pt.y)));
        point.z = min_pt.z + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max_pt.z-min_pt.z)));

        points.push_back(point);

    }

    return points;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr Scanner::random_scanner(double step, double searchRadius, size_t num_random_positions, pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, std::string file_name)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scanned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    Occlusion occlusion;

    std::vector<pcl::PointXYZ> scanner_positions = fixed_scanning_positions(min_pt, max_pt, 1); // generate random scanners

    std::unordered_set<int> addedPoints;

    for (size_t i = 0; i < scanner_positions.size(); i++) {
        
        pcl::PointXYZ scanner_position = scanner_positions[i];
        std::vector<pcl::PointXYZ> look_at_directions = random_look_at_direction(10, min_pt, max_pt);

        for (size_t j = 0; j < look_at_directions.size(); j++) {

            pcl::PointXYZ look_at_direction = look_at_directions[j];
            Ray3D ray = occlusion.generateRay(scanner_position, look_at_direction);

            pcl::PointXYZ point = scanner_position;
            while ( point.x < max_pt.x && point.y < max_pt.y && point.z < max_pt.z && point.x > min_pt.x && point.y > min_pt.y && point.z > min_pt.z) {
                
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

