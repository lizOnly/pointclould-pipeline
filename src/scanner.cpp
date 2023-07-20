#include <vector>
#include <iostream>
#include <unordered_set>
#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>

#include "../headers/BaseStruct.h"
#include "../headers/helper.h"
#include "../headers/scanner.h"
#include "../headers/property.h"


#define DEG_TO_RAD 0.0174533


Scanner::Scanner()
{
    // empty constructor
}

Scanner::~Scanner()
{
    // empty destructor
}


/*
    This method is used to generate a sampled cloud using ray sampling method. We cast a ray from a light source to a point on the sphere. 
    Then we sample points along the ray with given step, for each point we search for its nearest neighbor within a given search radius.
    The neighbor points are added to the sampled cloud.
*/
pcl::PointCloud<pcl::PointXYZRGB>::Ptr Scanner::multi_sphere_scanner(double step, 
                                                                     double searchRadius, // search radius
                                                                     double sphereRadius, // radius of sphere
                                                                     size_t num_samples, // number of samples
                                                                     pcl::PointXYZ& minPt,
                                                                     pcl::PointXYZ& maxPt,
                                                                     std::vector<pcl::PointXYZ> scanning_positions,
                                                                     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
                                                                     std::string file_name) {

    Property prop;
    Helper helper;
    
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sampledCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::cout << "total rays: " << num_samples * scanning_positions.size() << std::endl;

    std::unordered_set<int> addedPoints;

    for (int k = 0; k < scanning_positions.size(); k++) {

        std::vector<pcl::PointXYZ> sampledPoints = helper.UniformSamplingSphere(scanning_positions[k], sphereRadius, num_samples);
        std::cout << "scanning position " << k << std::endl;

        // store all index of points that should be added to the sampled cloud
        for (int i = 0; i < sampledPoints.size(); i++) {

            pcl::PointXYZ sampledPoint = sampledPoints[i];
            Ray3D ray = helper.generateRay(scanning_positions[k], sampledPoint);
            
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

    std::cout << "total points after scanning: " << addedPoints.size() << std::endl;

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
    outputPath += std::to_string(num_samples) + ".pcd";
    pcl::io::savePCDFileASCII (outputPath, *sampledCloud);

    return sampledCloud;

}



std::vector<pcl::PointXYZ> Scanner::scanning_positions( size_t num_positions, 
                                                        pcl::PointXYZ& minPt, 
                                                        pcl::PointXYZ& maxPt,
                                                        int pattern) {

    Helper helper;

    std::vector<pcl::PointXYZ> positions;
    
    pcl::PointXYZ center;
    center.x = (minPt.x + maxPt.x) / 2;
    center.y = (minPt.y + maxPt.y) / 2;
    center.z = (minPt.z + maxPt.z) / 2;

    // pattern 0: fixed 5 positions for square scanner                                            
    if (pattern == 0) {

        pcl::PointXYZ position1;
        position1.x = (center.x + minPt.x) / 2;
        position1.y = (center.y + minPt.y) / 2;
        position1.z = center.z;

        pcl::PointXYZ position2;
        position2.x = (center.x + maxPt.x) / 2;
        position2.y = (center.y + maxPt.y) / 2;
        position2.z = center.z;

        pcl::PointXYZ position3;
        position3.x = (center.x + minPt.x) / 2;
        position3.y = (center.y + maxPt.y) / 2;
        position3.z = center.z;

        pcl::PointXYZ position4;
        position4.x = (center.x + maxPt.x) / 2;
        position4.y = (center.y + minPt.y) / 2;
        position4.z = center.z;

        positions.push_back(center);
        positions.push_back(position1);
        positions.push_back(position2);
        positions.push_back(position3);
        positions.push_back(position4);

    } else if (pattern == 1) { // random positions

        pcl::PointXYZ position;
        for (int i = 0; i < num_positions; i++) {

            position.x = minPt.x + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxPt.x-minPt.x)));
            position.y = minPt.y + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxPt.y-minPt.y)));
            position.z = minPt.z + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxPt.z-minPt.z)));

            positions.push_back(position);

        }
    } else if (pattern == 2) { // one center position

        positions.push_back(center);
        
    } else if (pattern == 3) {
        positions = helper.getSphereLightSourceCenters(minPt, maxPt);
    }

    return positions;

}


std::vector<pcl::PointXYZ> Scanner::sample_square_points(const pcl::PointXYZ& scanner_position, int sample_step, double distance, double angle) {

    double half_distance = distance / 2;
    std::vector<pcl::PointXYZ> points;

    pcl::PointXYZ square_center; // looking at the center of the square
    square_center.x = scanner_position.x + distance * cos(angle * DEG_TO_RAD);
    square_center.y = scanner_position.y + distance * sin(angle * DEG_TO_RAD);
    square_center.z = scanner_position.z;

    double step = half_distance / (double)sample_step; // 5 points

    for (int i = 0; i < sample_step + 1; i++) {
        
        if (i == 0) {
            points.push_back(square_center);

            for (int j = 1; j < sample_step + 1; j++) {

                pcl::PointXYZ center_up;
                center_up.x = square_center.x;
                center_up.y = square_center.y;
                center_up.z = square_center.z + step * j;

                points.push_back(center_up);

                pcl::PointXYZ center_down;

                center_down.x = square_center.x;
                center_down.y = square_center.y;
                center_down.z = square_center.z - step * j;

                points.push_back(center_down);

            }
            continue;
        }

        pcl::PointXYZ point_hor_left;
        point_hor_left.x = square_center.x + step * i * sin(angle * DEG_TO_RAD);
        point_hor_left.y = square_center.y - step * i * cos(angle * DEG_TO_RAD);
        point_hor_left.z = square_center.z;

        points.push_back(point_hor_left);

        pcl::PointXYZ point_hor_right;
        point_hor_right.x = square_center.x - step * i * sin(angle * DEG_TO_RAD);
        point_hor_right.y = square_center.y + step * i * cos(angle * DEG_TO_RAD);
        point_hor_right.z = square_center.z;

        points.push_back(point_hor_right);

        for (int j = 1; j < sample_step + 1; j++) {

            pcl::PointXYZ point_left_vert_up;
            point_left_vert_up.x = point_hor_left.x;
            point_left_vert_up.y = point_hor_left.y;
            point_left_vert_up.z = point_hor_left.z + step * j;
            points.push_back(point_left_vert_up);

            pcl::PointXYZ point_left_vert_down;
            point_left_vert_down.x = point_hor_left.x;
            point_left_vert_down.y = point_hor_left.y;
            point_left_vert_down.z = point_hor_left.z - step * j;
            points.push_back(point_left_vert_down);


            pcl::PointXYZ point_right_vert_up;
            point_right_vert_up.x = point_hor_right.x;
            point_right_vert_up.y = point_hor_right.y;
            point_right_vert_up.z = point_hor_right.z + step * j;
            points.push_back(point_right_vert_up);

            pcl::PointXYZ point_right_vert_down;
            point_right_vert_down.x = point_hor_right.x;
            point_right_vert_down.y = point_hor_right.y;
            point_right_vert_down.z = point_hor_right.z - step * j;
            points.push_back(point_right_vert_down);

        }
    }
    
    std::cout << points.size() << " points generated for a square"<< std::endl;

    return points;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr Scanner::multi_square_scanner(double step, 
                                                                    double searchRadius, // search radius
                                                                    pcl::PointXYZ& minPt, 
                                                                    pcl::PointXYZ& maxPt,
                                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
                                                                    std::string file_name)
{
    Helper helper;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scanned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    std::vector<pcl::PointXYZ> scanner_positions = scanning_positions(5, minPt, maxPt, 0); // generate fixed scanners
    std::cout << "total scanners: " << scanner_positions.size() << std::endl;

    std::unordered_set<int> addedPoints;

    for (int i = 0; i < scanner_positions.size(); i++) {

        pcl::PointXYZ scanner_position = scanner_positions[i];
        for (double angle = 0.0; angle <= 360.0; angle += 10.0) {
            std::vector<pcl::PointXYZ> points = sample_square_points(scanner_position, 10, 0.1, angle);

            for (int j = 0; j < points.size(); j++) {

                pcl::PointXYZ point = points[j];
                Ray3D ray = helper.generateRay(scanner_position, point);
                
                while ( point.x < maxPt.x && point.y < maxPt.y && point.z < maxPt.z && point.x > minPt.x && point.y > minPt.y && point.z > minPt.z) {
                    
                    std::vector<int> pointIdxRadiusSearch;
                    std::vector<float> pointRadiusSquaredDistance;
                    
                    if ( kdtree.radiusSearch(point, searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
                
                        addedPoints.insert(pointIdxRadiusSearch[0]);
                        break;
                        
                    }
                    
                    point.x += step * ray.direction.x;
                    point.y += step * ray.direction.y;
                    point.z += step * ray.direction.z;
                    
                }
            }
        }
    }    

    std::cout << "total points after scanning: " << addedPoints.size() << std::endl;

    for (const auto& ptIdx : addedPoints) {

        pcl::PointXYZRGB point;

        point.x = coloredCloud->points[ptIdx].x;
        point.y = coloredCloud->points[ptIdx].y;
        point.z = coloredCloud->points[ptIdx].z;

        point.r = coloredCloud->points[ptIdx].r;
        point.g = coloredCloud->points[ptIdx].g;
        point.b = coloredCloud->points[ptIdx].b;

        scanned_cloud->push_back(point);

    }                            

    scanned_cloud->width = scanned_cloud->size();
    scanned_cloud->height = 1;
    scanned_cloud->is_dense = true;

    std::string outputPath = "../files/scanned_" + file_name.substr(0, file_name.length() - 4) + ".pcd";
    pcl::io::savePCDFileASCII (outputPath, *scanned_cloud); 

    return scanned_cloud;   
}

std::vector<pcl::PointXYZ> random_look_at_direction(int num_directions, pcl::PointXYZ& minPt, pcl::PointXYZ& maxPt) {

    std::vector<pcl::PointXYZ> points;

    for (int i = 0; i < num_directions; i++) {

        pcl::PointXYZ point;
        point.x = minPt.x + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxPt.x-minPt.x)));
        point.y = minPt.y + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxPt.y-minPt.y)));
        point.z = minPt.z + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxPt.z-minPt.z)));

        points.push_back(point);

    }

    return points;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr Scanner::random_scanner(double step,
                                                                double searchRadius, // search radius
                                                                size_t num_random_positions,
                                                                pcl::PointXYZ& minPt, 
                                                                pcl::PointXYZ& maxPt,
                                                                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
                                                                std::string file_name)
{

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scanned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    Helper helper;

    std::vector<pcl::PointXYZ> scanner_positions = scanning_positions(num_random_positions, minPt, maxPt, 1); // generate random scanners

    std::unordered_set<int> addedPoints;

    for (size_t i = 0; i < scanner_positions.size(); i++) {
        
        pcl::PointXYZ scanner_position = scanner_positions[i];
        std::vector<pcl::PointXYZ> look_at_directions = random_look_at_direction(10, minPt, maxPt);

        for (size_t j = 0; j < look_at_directions.size(); j++) {

            pcl::PointXYZ look_at_direction = look_at_directions[j];
            Ray3D ray = helper.generateRay(scanner_position, look_at_direction);

            pcl::PointXYZ point = scanner_position;
            while ( point.x < maxPt.x && point.y < maxPt.y && point.z < maxPt.z && point.x > minPt.x && point.y > minPt.y && point.z > minPt.z) {
                
                std::vector<int> pointIdxRadiusSearch;
                std::vector<float> pointRadiusSquaredDistance;
                
                if ( kdtree.radiusSearch(point, searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
            
                    addedPoints.insert(pointIdxRadiusSearch[0]);
                    break;
                }
                
                point.x += step * ray.direction.x;
                point.y += step * ray.direction.y;
                point.z += step * ray.direction.z;
                
            }
        }
    }

    std::cout << "total points after scanning: " << addedPoints.size() << std::endl;

    for (const auto& ptIdx : addedPoints) {

        pcl::PointXYZRGB point;

        point.x = coloredCloud->points[ptIdx].x;
        point.y = coloredCloud->points[ptIdx].y;
        point.z = coloredCloud->points[ptIdx].z;

        point.r = coloredCloud->points[ptIdx].r;
        point.g = coloredCloud->points[ptIdx].g;
        point.b = coloredCloud->points[ptIdx].b;

        scanned_cloud->push_back(point);

    }

    scanned_cloud->width = scanned_cloud->size();
    scanned_cloud->height = 1;
    scanned_cloud->is_dense = true;

    std::string outputPath = "../files/random_scanned_" + file_name.substr(0, file_name.length() - 4) + "-" + std::to_string(num_random_positions) + ".pcd";
    pcl::io::savePCDFileASCII (outputPath, *scanned_cloud);

    return scanned_cloud;
}


// pcl::PointCloud<pcl::PointXYZ>::Ptr Scanner::scan_visible_area() {
//     pcl::PointXYZ scan_position = Scanner::scanning_positions(1, minPt, maxPt, 2)[0];
// }
