#include <chrono>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "BaseStruct.h"


class Helper
{
    public:
    
        Helper();

        ~Helper();

        void displayRunningTime(std::chrono::high_resolution_clock::time_point start);

        pcl::PointCloud<pcl::PointXYZ>::Ptr centerCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr centerColoredCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt, std::string file_name);

        void extractWalls(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

        void removePointsInSpecificColor(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int color[3]);

    private:
        /* data */
};
