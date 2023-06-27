#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "BaseStruct.h"


class Validation {

    private:
        /* data */
    public:
        Validation();
        ~Validation();
        void raySampledCloud(double step, 
                             double searchRadius, // search radius
                             double sphereRadius, // radius of sphere
                             size_t num_samples, // number of samples
                             pcl::PointXYZ& minPt, 
                             pcl::PointXYZ& maxPt,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud);

};

