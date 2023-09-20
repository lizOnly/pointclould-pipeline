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

        Eigen::Vector3d getMeshMinVertex() {
            return mesh_min_vertex;
        };

        Eigen::Vector3d getMeshMaxVertex() {
            return mesh_max_vertex;
        };

        pcl::PointCloud<pcl::PointXYZI>::Ptr getEstimatedBoundCloud() {
            return estimated_bound_cloud;
        };
        
        void setPointRadius(double radius) {
            point_radius = radius;
        };


        void setOctreeResolution(float resolution) {
            octree_resolution = resolution;
        };


        void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
            input_cloud = cloud;
        };

        void setInputCloudRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {

            input_cloud_rgb = cloud;

        }

        void setInputCloudBound(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
            input_cloud_bound = cloud;
        };

        void setInputSampleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
            input_sample_cloud = cloud;
        };

        void setPolygonClouds(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds) {
            polygonClouds = clouds;
        };

        void setAllCoefficients(std::vector<pcl::ModelCoefficients::Ptr> coefficients) {
            allCoefficients = coefficients;
        };

        void setConfigInfo(int samples_per_unit_area, int pattern) {
            this->samples_per_unit_area = samples_per_unit_area;
            this->pattern = pattern;
        };


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

        void regionGrowingSegmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, size_t min_cluster_size, size_t max_cluster_size, int num_neighbours, int k_search_neighbours, double smoothness_threshold, double curvature_threshold);

        Eigen::Vector3d computeCentroid(pcl::PointCloud<pcl::PointXYZ>::Ptr polygon_cloud);

        void generateTriangleFromCluster();

        pcl::ModelCoefficients::Ptr computePlaneCoefficients(std::vector<pcl::PointXYZ> points);

        pcl::PointCloud<pcl::PointXYZ>::Ptr estimatePolygon(std::vector<pcl::PointXYZ> points, pcl::ModelCoefficients::Ptr coefficients);

        std::vector<std::vector<pcl::PointXYZ>> parsePointString(const std::string& input);

        std::vector<pcl::PointXYZ> generateDefaultPolygon();

        std::vector<std::vector<pcl::PointXYZ>> parsePolygonData(const std::string& filename);

        double halton(int index, int base);

        // occlusion level
        bool rayIntersectPolygon(const Ray3D& ray, const pcl::PointCloud<pcl::PointXYZ>::Ptr& polygonCloud, const pcl::ModelCoefficients::Ptr coefficients);
        
        Ray3D generateRay(const pcl::PointXYZ& center, const pcl::PointXYZ& surfacePoint);

        bool rayBoxIntersection(const Ray3D& ray, const pcl::PointXYZ& min_pt, const pcl::PointXYZ& max_pt);

        bool rayIntersectSpehre(pcl::PointXYZ& origin, pcl::PointXYZ& direction, pcl::PointXYZ& point, double radius);

        bool rayIntersectPointCloud(const Ray3D& ray);
        
        std::vector<pcl::PointXYZ> UniformSampleSphere(pcl::PointXYZ center, size_t num_samples);

        std::vector<pcl::PointXYZ> HaltonSampleSphere(pcl::PointXYZ center, size_t num_samples);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr computeMedianDistance(double radius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_density);

        void traverseOctree();

        void generateRandomRays(size_t num_rays, pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt);

        void checkRayOctreeIntersection(Ray3D& ray, pcl::PointXYZ& direction, OctreeNode& node);

        double randomRayBasedOcclusionLevel(bool use_openings);

        void parseTrianglesFromOBJ(const std::string& mesh_path);

        void parseTrianglesFromPLY(const std::string& ply_path);

        void uniformSampleTriangle(double samples_per_unit_area);

        void haltonSampleTriangle(double samples_per_unit_area);

        void estimateBoundary(int K_nearest);

        void estimateSemantics();

        double calculateTriangleArea(Triangle& tr);

        void generateRayFromTriangle(std::vector<Eigen::Vector3d>& origins);

        void generateRaysWithIdx(std::vector<Eigen::Vector3d>& origins, size_t num_rays_per_vp);

        std::vector<Eigen::Vector3d> viewPointPattern(Eigen::Vector3d& min, Eigen::Vector3d& max, Eigen::Vector3d& center);

        bool rayTriangleIntersect(Triangle& tr, Ray& ray, Eigen::Vector3d& intersection_point);

        bool getRayTriangleIntersectionPt(Triangle& tr, Ray& ray, size_t idx, Intersection& intersection);

        bool rayIntersectOctreeNode(Ray& ray, OctreeNode& node);

        void computeFirstHitIntersection(Ray& ray);

        void checkRayOctreeIntersectionTriangle(Ray& ray, OctreeNode& node, size_t& idx);

        double triangleBasedOcclusionLevel();

        void generateCloudFromIntersection();

        Eigen::AlignedBox3d getBoundingBox() {
            return bbox;
        };

        void buildLeafBBoxSet();

        void buildCompleteOctreeNodes();

        void buildCompleteOctreeNodesTriangle();

        private:

            std::string scene_name;
            int samples_per_unit_area;
            int pattern;

            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds;
            std::vector<pcl::ModelCoefficients::Ptr> allCoefficients;

            std::unordered_map<size_t, Ray3D> t_random_rays; // table of random rays
            double point_radius;
            // double point_radius_random;

            float octree_resolution;

            std::vector<pcl::PointIndices> rg_clusters;
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud_rgb;
            pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_bound;
            pcl::PointCloud<pcl::PointXYZ>::Ptr input_sample_cloud;
            pcl::PointCloud<pcl::PointXYZI>::Ptr estimated_bound_cloud;
            std::vector<LeafBBox> octree_leaf_bbox;


            std::vector<Eigen::Vector3d> vertices; // all vertices of mesh
            Eigen::AlignedBox3d bbox; // bounding box of mesh
            Eigen::Vector3d mesh_min_vertex; // min vertex of mesh
            Eigen::Vector3d mesh_max_vertex; // max vertex of mesh


            std::unordered_map<size_t, Intersection> t_intersections; // table of intersections
            std::unordered_map<size_t, Triangle> t_triangles; // table of triangles
            std::unordered_map<size_t, Ray> t_rays; // tables of rays
            std::unordered_map<size_t, Sample> t_samples; // table of samples
        
            pcl::PointCloud<pcl::PointXYZI>::Ptr t_octree_cloud; // octree cloud to store center of triangles
            pcl::PointCloud<pcl::PointXYZ>::Ptr t_pure_octree_cloud; // pure octree cloud to store center of triangles
            
            std::vector<LeafBBox> t_octree_leaf_bbox; // bounding box of octree leaf nodes
            std::vector<LeafBBox> t_octree_leaf_bbox_triangle;

            Eigen::Vector3d oc_cloud_min_pt;
            Eigen::Vector3d oc_cloud_max_pt;

            std::unordered_map<size_t, OctreeNode> t_octree_nodes; // table of octree nodes

};

