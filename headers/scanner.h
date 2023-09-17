#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "BaseStruct.h"


class Scanner {

    public:
        Scanner();

        ~Scanner();

        void setPointRadius(float radius) {
            point_radius = radius;
        }

        void setOctreeResolution(float resolution) {
            octree_resolution = resolution;
        }

        void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
            input_cloud = cloud;
        }

        void setInputCloudGT(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
            input_cloud_gt = cloud;
        }

        void setInputCloudColor(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
            input_cloud_color = cloud;
        }

        void setSamplingHor(size_t hor) {
            sampling_hor = hor;
        }

        void setSamplingVer(size_t ver) {
            sampling_ver = ver;
        }

        void sphere_scanner(int pattern, std::string path);
        
        void traverseOctree();

        void buildCompleteOctreeNodes();

        bool rayBoxIntersection(const Ray3D& ray, const pcl::PointXYZ& min_pt, const pcl::PointXYZ& max_pt);

        bool rayIntersectSpehre(pcl::PointXYZ& origin, pcl::PointXYZ& direction, pcl::PointXYZ& point);

        void checkRayOctreeIntersection(Ray3D& ray, OctreeNode& node);
        
        void checkFirstHitPoint(Ray3D& ray);

        std::vector<Eigen::Vector3d> create_scanning_pattern();

        void generateRays(std::vector<pcl::PointXYZ> origins);

        void generateRaysHalton(size_t num_rays_per_vp, std::vector<pcl::PointXYZ> origins);

        std::vector<pcl::PointXYZ> random_scanning_positions(pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt, int num_scanners);

        std::vector<pcl::PointXYZ> fixed_scanning_positions(pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt, int pattern);
        
        std::vector<pcl::PointXYZ> sample_square_points(const pcl::PointXYZ& scanner_position, int sample_step, double distance, double angle);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr multi_square_scanner(double step, double searchRadius, pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, std::string file_name);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr random_scanner(double step, double searchRadius, size_t num_random_positions, pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, std::string file_name);

    private:

        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
        pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_gt;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud_color;

        pcl::PointCloud<pcl::PointXYZ>::Ptr octree_cloud;

        size_t sampling_hor;
        size_t sampling_ver;
        

        std::unordered_map<size_t, OctreeNode> t_octree_nodes;
        std::unordered_map<size_t, Ray3D> t_rays;

        std::vector<LeafBBox> octree_leaf_bbox;

        float octree_resolution;
        double point_radius;
            
};

