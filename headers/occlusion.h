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

        pcl::PointXYZ transformPoint(pcl::PointXYZ& point, pcl::PointXYZ& center);

        pcl::PointCloud<pcl::PointXYZ>::Ptr centerCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr centerColoredCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt, std::string file_name);

        void removePointsInSpecificColor(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int color[3]);

        void regionGrowingSegmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

        void extractWalls(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        

        pcl::ModelCoefficients::Ptr computePlaneCoefficients(std::vector<pcl::PointXYZ> points);

        pcl::PointCloud<pcl::PointXYZ>::Ptr estimatePolygon(std::vector<pcl::PointXYZ> points, 
                                                            pcl::ModelCoefficients::Ptr coefficients);

        std::vector<std::vector<pcl::PointXYZ>> parsePointString(const std::string& input);

        std::vector<std::vector<pcl::PointXYZ>> parsePolygonData(const std::string& filename);

        // occlusion level
        bool rayIntersectPolygon(const Ray3D& ray, const pcl::PointCloud<pcl::PointXYZ>::Ptr& polygonCloud, const pcl::ModelCoefficients::Ptr coefficients);
        
        Ray3D generateRay(const pcl::PointXYZ& center, const pcl::PointXYZ& surfacePoint);

        bool rayIntersectDisk(const Ray3D& ray, const Disk3D& disk);

        pcl::PointXYZ rayBoxIntersection(const Ray3D& ray, const pcl::PointXYZ& minPt, const pcl::PointXYZ& maxPt);

        bool rayIntersectPointCloud(const Ray3D& ray, double step, double radius, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt, 
                                    pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree);
        
        std::vector<pcl::PointXYZ> getSphereLightSourceCenters(pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt);

        std::vector<pcl::PointXYZ> UniformSamplingSphere(pcl::PointXYZ center, double radius, size_t num_samples);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr computeMedianDistance(double radius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_density);

        bool rayIntersectPcdMedianDistance(const Ray3D& ray, double step, double radius, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_median_distance);     

        pcl::PointCloud<pcl::PointXYZI>::Ptr computeDistanceVariance(double radius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);                                    

        double rayBasedOcclusionLevelMedian(pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt, double density, double radius, int pattern,
                                            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_median_distance,
                                            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds, std::vector<pcl::ModelCoefficients::Ptr> allCoefficients);

        double rayBasedOcclusionLevel(pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt, double radius, int pattern, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds, std::vector<pcl::ModelCoefficients::Ptr> allCoefficients);


        void parseTrianglesFromOBJ(const std::string& mesh_path);
        double calculateTriangleArea(Triangle& tr);
        void computeMeshBoundingBox();
        void generateRaysWithIdx(Eigen::Vector3d& origin, size_t num_samples);
        bool rayTriangleIntersect(Triangle& tr, Ray& ray, Eigen::Vector3d& intersectionPoint);
        bool getRayTriangleIntersectionPt(Triangle& tr, Ray& ray, Eigen::Vector3d& origin, size_t idx, Intersection& intersection);
        void isFirstHitIntersection(Ray& ray);
        double triangleBasedOcclusionLevel(Eigen::Vector3d& origin);
        Eigen::AlignedBox3d getBoundingBox() {
            return bbox;
        }
        
        private:
            std::vector<Eigen::Vector3d> vertices; // all vertices of mesh

            Eigen::AlignedBox3d bbox; // bounding box of mesh
            std::unordered_map<size_t, Intersection> t_intersections; // table of intersections
            std::unordered_map<size_t, Triangle> t_triangles; // table of triangles
            std::unordered_map<size_t, Ray> t_rays; // table of rays

};

