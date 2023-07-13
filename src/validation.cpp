#include <vector>
#include <iostream>
#include <unordered_set>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>

#include "../headers/BaseStruct.h"
#include "../headers/helper.h"
#include "../headers/validation.h"
#include "../headers/property.h"

Validation::Validation()
{
    // empty constructor
}

Validation::~Validation()
{
    // empty destructor
}

/*
    This method is used to generate a sampled cloud using ray sampling method. We cast a ray from a light source to a point on the sphere. 
    Then we sample points along the ray with given step, for each point we search for its nearest neighbor within a given search radius.
    The neighbor points are added to the sampled cloud.
*/
pcl::PointCloud<pcl::PointXYZRGB>::Ptr Validation::raySampleCloud(double step, 
                                                                  double searchRadius, // search radius
                                                                  double sphereRadius, // radius of sphere
                                                                  size_t numSamples, // number of samples
                                                                  pcl::PointXYZ& minPt, 
                                                                  pcl::PointXYZ& maxPt,
                                                                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
                                                                  double density,
                                                                  std::string file_name) {

    Helper helper;
    Property prop;

    if (density <= 10) {
        
        density = density / 10.0 + 1.0;
    
    } else if (density > 10){

        density = 2.0;

    }
    
    size_t newNumSamples = (double) numSamples / density;
    std::cout << "number of samples adjusted by density: " << newNumSamples << std::endl;

    std::vector<pcl::PointXYZ> centers = helper.getSphereLightSourceCenters(minPt, maxPt);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    kdtree.setInputCloud(cloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sampledCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::cout << "total rays: " << newNumSamples * centers.size() << std::endl;

    std::unordered_set<int> addedPoints;

    for (int k = 0; k < centers.size(); k++) {

        std::vector<pcl::PointXYZ> sampledPoints = helper.UniformSamplingSphere(centers[k], sphereRadius, newNumSamples);
        std::cout << "center " << k << std::endl;

        // store all index of points that should be added to the sampled cloud
        for (int i = 0; i < sampledPoints.size(); i++) {

            pcl::PointXYZ sampledPoint = sampledPoints[i];
            Ray3D ray = helper.generateRay(centers[k], sampledPoint);
            
            while ( sampledPoint.x < maxPt.x && sampledPoint.y < maxPt.y && sampledPoint.z < maxPt.z && sampledPoint.x > minPt.x && sampledPoint.y > minPt.y && sampledPoint.z > minPt.z) {
                
                std::vector<int> pointIdxRadiusSearch;
                std::vector<float> pointRadiusSquaredDistance;
                
                if ( kdtree.radiusSearch(sampledPoint, searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
              
                    addedPoints.insert(pointIdxRadiusSearch[0]);
                    break;
                    
                }
                
                sampledPoint.x += step * ray.direction.x;
                sampledPoint.y += step * ray.direction.y;
                sampledPoint.z += step * ray.direction.z;
                
            }
        }
    }

    std::cout << "total points after downsampling: " << addedPoints.size() << std::endl;

    for (const auto& ptIdx : addedPoints) {

        pcl::PointXYZRGB point;

        point.x = cloud->points[ptIdx].x;
        point.y = cloud->points[ptIdx].y;
        point.z = cloud->points[ptIdx].z;

        point.r = coloredCloud->points[ptIdx].r;
        point.g = coloredCloud->points[ptIdx].g;
        point.b = coloredCloud->points[ptIdx].b;

        sampledCloud->push_back(point);

    }

    sampledCloud->width = sampledCloud->size();
    sampledCloud->height = 1;
    sampledCloud->is_dense = true;

    std::string outputPath = "../files/rs_" + file_name.substr(0, file_name.length() - 4) + "-";
    // std::string outputPath = "../files/rs-";
    outputPath += std::to_string(numSamples) + ".pcd";
    pcl::io::savePCDFileASCII (outputPath, *sampledCloud);

    return sampledCloud;

}

