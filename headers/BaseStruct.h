#ifndef BASESTRUCT_H
#define BASESTRUCT_H

#include <pcl/point_types.h>

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

#endif
