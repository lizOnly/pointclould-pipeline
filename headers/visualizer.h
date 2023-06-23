#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>

class visualizer
{
    private:
        /* data */
    public:
        visualizer(/* args */);
        ~visualizer();

        void visualizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        void pointPickingEventOccurred(const pcl::visualization::PointPickingEvent& event, void* viewer_void);
        void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* viewer_void);        
};

