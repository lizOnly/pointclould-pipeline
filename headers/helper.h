#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


class Helper {
    public:
        Helper();
        ~Helper();
        
        void identifyNormalHoles(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        void identifyOcclusionHoles(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        void voxelizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        void estimateOcclusion(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        void removeOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        void removePointsInSpecificColor(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int color[3]);
};

