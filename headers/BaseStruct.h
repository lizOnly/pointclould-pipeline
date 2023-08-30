#ifndef BASESTRUCT_H
#define BASESTRUCT_H

#include <pcl/point_types.h>
#include <Eigen/Core>



struct Ray3D { // ray structure used for point cloud ray tracing

    size_t index;
    pcl::PointXYZ origin;
    pcl::PointXYZ direction;
    std::vector<size_t> bound_intersection_idx; // index in point cloud
    std::vector<size_t> clutter_intersection_idx; // index in triangle vector
    int type; // how many boundary points are hit by this ray
    bool intersect_clutter = false;

};

struct Intersection {

    size_t index;
    size_t triangle_index;
    size_t ray_index;
    Eigen::Vector3d point;
    double distance_to_look_at_point;
    double distance_to_origin;
    bool is_first_hit;

};

struct Ray { 

    size_t index;
    size_t source_triangle_index;
    size_t source_sample_index;
    size_t first_hit_intersection_idx;
    Eigen::Vector3d origin;
    Eigen::Vector3d look_at_point;
    Eigen::Vector3d direction;
    std::vector<size_t> intersection_idx;
    std::vector<size_t> triangle_idx;

};


struct Sample {

    size_t index;
    size_t triangle_index;
    Eigen::Vector3d point;
    bool is_visible = false;
    std::vector<size_t> ray_idx;

};

struct Triangle {

    size_t index;
    Eigen::Vector3d v1;
    Eigen::Vector3d v2;
    Eigen::Vector3d v3;
    Eigen::Vector3d center;
    double area;
    std::vector<size_t> sample_idx;
    std::vector<size_t> intersection_idx;
    std::vector<size_t> ray_idx;
    
};


struct OctreeNode {

    size_t index;
    size_t parent_index = -1;
    size_t prev = -1;
    size_t next = -1;
    int depth;
    std::vector<size_t> children;
    std::vector<size_t> triangle_idx;
    std::vector<size_t> point_idx;

    int diagonal_distance;
    
    Eigen::Vector3d min_pt;
    Eigen::Vector3d max_pt;

    Eigen::Vector3d min_pt_triangle;
    Eigen::Vector3d max_pt_triangle;
    
    bool is_leaf = false;
    bool is_branch = false;
    bool is_root = false;

};

struct LeafBBox {
    
    size_t index;
    Eigen::Vector3d min_pt;
    Eigen::Vector3d max_pt;
    std::vector<size_t> triangle_idx;
    std::vector<size_t> point_idx;

};

#endif
