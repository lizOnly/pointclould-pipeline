#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "BaseStruct.h"


class Validation {

    private:
        /* data */
    public:
        Validation();

        ~Validation();

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr raySampleCloud(double step, 
                            double searchRadius,
                            double sphereRadius, 
                            size_t num_samples,
                            pcl::PointXYZ& minPt, 
                            pcl::PointXYZ& maxPt,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
                            double density,
                            std::string file_name);

};

