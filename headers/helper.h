#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "BaseStruct.h"



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
        void regionGrowingSegmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

        pcl::PointCloud<pcl::Normal>::Ptr normalEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        std::vector<pcl::PointXYZ> getSphereLightSourceCenters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        std::vector<pcl::PointXYZ> UniformSamplingSphere(pcl::PointXYZ center, double radius, size_t num_samples);
        
        Ray3D generateRay(const pcl::PointXYZ& center, const pcl::PointXYZ& surfacePoint);
        bool rayIntersectDisk(const Ray3D& ray, const Disk3D& disk);
        bool rayIntersectPointCloud(const Ray3D& ray, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double step);
        Disk3D convertPointToDisk(const pcl::PointXYZ& point, const pcl::Normal& normal, const double& radius);
        double rayBasedOcclusionLevel(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, size_t num_samples);

};

