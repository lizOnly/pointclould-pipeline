#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "../headers/BaseStruct.h"


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
    std::cout << " Time taken by this run: " << duration.count() << " seconds" << std::endl;
}