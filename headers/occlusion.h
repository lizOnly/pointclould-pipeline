#include <string>
#include <unordered_map>
#include <tuple>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree.h>

#include "BaseStruct.h"

class Occlusion {

    public:

        Occlusion();
        ~Occlusion();
        

        template <typename PointT>

        typename pcl::PointCloud<PointT>::Ptr voxelizePointCloud(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file_name) {
            pcl::VoxelGrid<PointT> vg;
            vg.setInputCloud(cloud);
            vg.setLeafSize(0.03f, 0.03f, 0.03f);
            typename pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
            vg.filter(*cloud_filtered);
            std::cout << "Point cloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl;

            cloud_filtered->width = cloud_filtered->points.size();
            cloud_filtered->height = 1;
            cloud_filtered->is_dense = true;

            pcl::io::savePCDFileASCII("../files/v_" + file_name, *cloud_filtered);

            return cloud_filtered;
        };

        pcl::PointCloud<pcl::PointXYZ>::Ptr centerCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr centerColoredCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt, std::string file_name);

        void removePointsInSpecificColor(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int color[3]);

        void regionGrowingSegmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

        Eigen::Vector3d computeCentroid(pcl::PointCloud<pcl::PointXYZ>::Ptr polygon_cloud);

        void generateTriangleFromCluster();

        void extractWalls(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        

        pcl::ModelCoefficients::Ptr computePlaneCoefficients(std::vector<pcl::PointXYZ> points);

        pcl::PointCloud<pcl::PointXYZ>::Ptr estimatePolygon(std::vector<pcl::PointXYZ> points, pcl::ModelCoefficients::Ptr coefficients);

        std::vector<std::vector<pcl::PointXYZ>> parsePointString(const std::string& input);

        std::vector<pcl::PointXYZ> generateDefultPolygon();

        std::vector<std::vector<pcl::PointXYZ>> parsePolygonData(const std::string& filename);

        // occlusion level
        bool rayIntersectPolygon(const Ray3D& ray, const pcl::PointCloud<pcl::PointXYZ>::Ptr& polygonCloud, const pcl::ModelCoefficients::Ptr coefficients);
        
        Ray3D generateRay(const pcl::PointXYZ& center, const pcl::PointXYZ& surfacePoint);

        bool rayIntersectDisk(const Ray3D& ray, const Disk3D& disk);

        bool rayBoxIntersection(const Ray3D& ray, const pcl::PointXYZ& minPt, const pcl::PointXYZ& maxPt);

        bool rayIntersectSpehre(pcl::PointXYZ& origin, pcl::PointXYZ& direction, pcl::PointXYZ& point, double radius);

        bool rayIntersectPointCloud(const Ray3D& ray, double radius);
        
        std::vector<pcl::PointXYZ> getSphereLightSourceCenters(pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt);

        std::vector<pcl::PointXYZ> UniformSamplingSphere(pcl::PointXYZ center, size_t num_samples);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr computeMedianDistance(double radius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_density);

        void traverseOctree();

        double rayBasedOcclusionLevel(pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt, size_t num_rays_per_vp, double point_radius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds, std::vector<pcl::ModelCoefficients::Ptr> allCoefficients);
        /*-----------------------------------------------------------------------------------------------------------*/

        void parseTrianglesFromOBJ(const std::string& mesh_path);

        double calculateTriangleArea(Triangle& tr);

        void computeMeshBoundingBox();

        void generateRaysWithIdx(std::vector<Eigen::Vector3d>& origins, size_t num_samples);

        std::vector<Eigen::Vector3d> viewPointPattern(const int& pattern);

        bool rayTriangleIntersect(Triangle& tr, Ray& ray, Eigen::Vector3d& intersectionPoint);

        bool getRayTriangleIntersectionPt(Triangle& tr, Ray& ray, Eigen::Vector3d& origin, size_t idx, Intersection& intersection);

        bool rayIntersectLeafBbox(Ray& ray, LeafBBox& bbox);

        void isFirstHitIntersection(Ray& ray);

        double triangleBasedOcclusionLevel(Eigen::Vector3d& origin);

        void generateCloudFromIntersection();

        void generateCloudFromTriangle();

        Eigen::AlignedBox3d getBoundingBox() {
            return bbox;
        };

        void traverseOctreeTriangle();

        void buildOctreeCloud();

        private:

            std::vector<pcl::PointIndices> rg_clusters;
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
            std::vector<LeafBBox> octree_leaf_bbox;


            std::vector<Eigen::Vector3d> vertices; // all vertices of mesh
            Eigen::AlignedBox3d bbox; // bounding box of mesh
            std::unordered_map<size_t, Intersection> t_intersections; // table of intersections
            std::unordered_map<size_t, Triangle> t_triangles; // table of triangles
            std::unordered_map<size_t, Ray> t_rays; // table of rays

            pcl::PointCloud<pcl::PointXYZI>::Ptr t_octree_cloud; // octree cloud to store center of triangles
            pcl::PointCloud<pcl::PointXYZ>::Ptr t_pure_octree_cloud;
            std::vector<LeafBBox> t_octree_leaf_bbox; // bounding box of octree leaf nodes

            Eigen::Vector3d oc_cloud_min_pt;
            Eigen::Vector3d oc_cloud_max_pt;

};

