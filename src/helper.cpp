#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/transforms.h>

#include "../headers/BaseStruct.h"
#include "../headers/helper.h"


Helper::Helper()
{
    // Constructor
}

Helper::~Helper()
{
    // Destructor
}


void Helper::displayRunningTime(std::chrono::high_resolution_clock::time_point start)
{
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Time taken by this run: " << duration.count() << " seconds" << std::endl;
}


/*
    This function is used exstract the walls from the point cloud
    and save them in a separate file
    @param cloud: the point cloud
    @return void
*/
void Helper::extractWalls(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch(0.05); 

    ne.compute(*cloud_normals);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_walls(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_walls_north(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_walls_south(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_walls_east(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_walls_west(new pcl::PointCloud<pcl::PointXYZ>);

    for (size_t i = 0; i < cloud->points.size(); ++i) {

        if (fabs(cloud_normals->points[i].normal_z) < 0.05) {

            if (fabs(cloud_normals->points[i].normal_x) > fabs(cloud_normals->points[i].normal_y)) {

                if (cloud_normals->points[i].normal_x > 0) {

                    cloud_walls_north->points.push_back(cloud->points[i]);

                } else {

                    cloud_walls_south->points.push_back(cloud->points[i]);

                }
            } else {

                if (cloud_normals->points[i].normal_y > 0) {

                    cloud_walls_east->points.push_back(cloud->points[i]);

                } else {

                    cloud_walls_west->points.push_back(cloud->points[i]);

                }
            }

            cloud_walls->points.push_back(cloud->points[i]);
        }

    }

    cloud_walls->width = cloud_walls->points.size();
    cloud_walls->height = 1;
    cloud_walls->is_dense = true;

    cloud_walls_north->width = cloud_walls_north->points.size();
    cloud_walls_north->height = 1;

    cloud_walls_south->width = cloud_walls_south->points.size();
    cloud_walls_south->height = 1;

    cloud_walls_east->width = cloud_walls_east->points.size();
    cloud_walls_east->height = 1;

    cloud_walls_west->width = cloud_walls_west->points.size();
    cloud_walls_west->height = 1;

}


void Helper::removePointsInSpecificColor(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int color[3]) {

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (const auto& point : cloud->points)
    {
        
        if (point.r != color[0] || point.g != color[1] || point.b != color[2])
        {
            std::cout << "Point " << point.x << " " << point.y << " " << point.z << " is not " << color[0] << " " << color[1] << " " << color[2] << std::endl;
            filtered_cloud->points.push_back(point);
        }
    }

    filtered_cloud->width = filtered_cloud->points.size();
    filtered_cloud->height = 1;
    filtered_cloud->is_dense = true;


}
