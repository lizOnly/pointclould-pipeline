#ifndef BASESTRUCT_H
#define BASESTRUCT_H

#include <pcl/point_types.h>
#include <Eigen/Core>

struct Point3D {
    double x, y, z;
};

struct Ray3D {
    pcl::PointXYZ origin;
    pcl::PointXYZ direction;
};

struct Disk3D {
    pcl::PointXYZ center;
    double radius;
    pcl::Normal normal;
};

struct Intersection {
    size_t index;
    size_t triangle_index;
    size_t ray_index;
    Eigen::Vector3d point;
    double distance_to_origin;
    bool is_first_hit;
};

struct Ray {
    size_t index;
    Eigen::Vector3d origin;
    Eigen::Vector3d direction;
    std::vector<size_t> intersectionIdx;
};

struct Triangle {
    size_t index;
    Eigen::Vector3d v1;
    Eigen::Vector3d v2;
    Eigen::Vector3d v3;
    Eigen::Vector3d normal;
    Eigen::Vector3d center; // center of gravity
    double area;
    double weighted_area;
    double occlusion_ratio;
    std::vector<size_t> intersectionIdx;
};

struct LeafBBox {
    Eigen::Vector3d min_pt;
    Eigen::Vector3d max_pt;
    std::vector<size_t> triangle_idx;
};

#endif
