#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "BaseStruct.h"


class Helper {
    public:
        Helper();
        ~Helper();
        
        template <typename PointT>

        typename pcl::PointCloud<PointT>::Ptr voxelizePointCloud(typename pcl::PointCloud<PointT>::Ptr cloud);

        pcl::PointXYZ transformPoint(pcl::PointXYZ& point, pcl::PointXYZ& center);

        pcl::PointCloud<pcl::PointXYZ>::Ptr centerCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                        pcl::PointXYZ& minPt, 
                                                        pcl::PointXYZ& maxPt);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr centerColoredCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
                                                                  pcl::PointXYZ& minPt, 
                                                                  pcl::PointXYZ& maxPt, 
                                                                  std::string file_name);

        void removePointsInSpecificColor(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int color[3]);

        void regionGrowingSegmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

        void extractWalls(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        

        pcl::ModelCoefficients::Ptr computePlaneCoefficients(std::vector<pcl::PointXYZ> points);

        pcl::PointCloud<pcl::PointXYZ>::Ptr estimatePolygon(std::vector<pcl::PointXYZ> points, 
                                                            pcl::ModelCoefficients::Ptr coefficients);

        std::vector<std::vector<pcl::PointXYZ>> parsePolygonData(const std::string& filename);

        // occlusion level
        bool rayIntersectPolygon(const Ray3D& ray, 
                                 const pcl::PointCloud<pcl::PointXYZ>::Ptr& polygonCloud, 
                                 const pcl::ModelCoefficients::Ptr coefficients);
        
        Ray3D generateRay(const pcl::PointXYZ& center, const pcl::PointXYZ& surfacePoint);

        bool rayIntersectDisk(const Ray3D& ray, const Disk3D& disk);

        pcl::PointXYZ rayBoxIntersection(const Ray3D& ray, const pcl::PointXYZ& minPt, const pcl::PointXYZ& maxPt);

        bool rayIntersectPointCloud(const Ray3D& ray, double step, double radius, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree);
        
        std::vector<pcl::PointXYZ> getSphereLightSourceCenters(pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt);

        std::vector<pcl::PointXYZ> UniformSamplingSphere(pcl::PointXYZ center, 
                                                         double radius, 
                                                         size_t num_samples);

        double rayBasedOcclusionLevel(
            pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt,
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds,
            std::vector<pcl::ModelCoefficients::Ptr> allCoefficients
        );

};

