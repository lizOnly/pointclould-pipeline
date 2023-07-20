#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "BaseStruct.h"


class Scanner {

    private:
        /* data */
    public:
        Scanner();

        ~Scanner();

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr multi_sphere_scanner(double step, 
                                                                    double searchRadius,
                                                                    double sphereRadius, 
                                                                    size_t num_samples,
                                                                    pcl::PointXYZ& minPt,
                                                                    pcl::PointXYZ& maxPt,
                                                                    std::vector<pcl::PointXYZ> scanning_positions,
                                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
                                                                    std::string file_name);

        std::vector<pcl::PointXYZ> scanning_positions(size_t num_positions, 
                                                        pcl::PointXYZ& minPt, 
                                                        pcl::PointXYZ& maxPt,
                                                        int pattern);
        
        std::vector<pcl::PointXYZ> sample_square_points(const pcl::PointXYZ& scanner_position, int sample_step, double distance, double angle);


        pcl::PointCloud<pcl::PointXYZRGB>::Ptr multi_square_scanner(double step, 
                                                                    double searchRadius, // search radius
                                                                    pcl::PointXYZ& minPt, 
                                                                    pcl::PointXYZ& maxPt,
                                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
                                                                    std::string file_name);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr random_scanner(double step, 
                                                            double searchRadius, // search radius
                                                            size_t num_random_positions,
                                                            pcl::PointXYZ& minPt, 
                                                            pcl::PointXYZ& maxPt,
                                                            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
                                                            std::string file_name);
};

