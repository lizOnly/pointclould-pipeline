#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "BaseStruct.h"


class Scanner {

    public:
        Scanner();

        ~Scanner();

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sphere_scanner(size_t num_rays_per_vp, int pattern, std::vector<pcl::PointXYZ> scanning_positions, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr gt_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, std::string file_name);
        
        void traverseOctree();

        bool rayBoxIntersection(const Ray3D& ray, const pcl::PointXYZ& minPt, const pcl::PointXYZ& maxPt);

        bool rayIntersectSpehre(pcl::PointXYZ& origin, pcl::PointXYZ& direction, pcl::PointXYZ& point);

        bool rayIntersectPointCloud(Ray3D& ray, pcl::PointXYZ& intersection, size_t& index);

        std::vector<pcl::PointXYZ> scanning_positions(pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt, int pattern);
        
        std::vector<pcl::PointXYZ> sample_square_points(const pcl::PointXYZ& scanner_position, int sample_step, double distance, double angle);


        pcl::PointCloud<pcl::PointXYZRGB>::Ptr multi_square_scanner(double step, double searchRadius, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, std::string file_name);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr random_scanner(double step, double searchRadius, size_t num_random_positions, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, std::string file_name);

    private:

        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr octree_cloud;

        std::vector<LeafBBox> octree_leaf_bbox;
            
};

