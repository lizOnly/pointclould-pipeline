#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "BaseStruct.h"


class Validation {

    private:
        /* data */
    public:
        Validation();
        ~Validation();
        void raySampleCloud(double step, 
                            double searchRadius,
                            double sphereRadius, 
                            size_t num_samples,
                            pcl::PointXYZ& minPt, 
                            pcl::PointXYZ& maxPt,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
                            bool hit_first_pt);

};

