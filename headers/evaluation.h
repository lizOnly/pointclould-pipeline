#include <vector>
#include <stdexcept>
#include <unordered_map>

class Evaluation {
    public:
        Evaluation();
        ~Evaluation();    
    
        void compareClouds(pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmented_cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr ground_truth_cloud);

        float calculateIoU();
        float calculateAccuracy();
        float calculatePrecision();
        float calculateRecall();
        float calculateF1Score();
        

    private:

        size_t tp, fp, fn, tn;
        std::vector<bool> ground_truth_labels;
        std::vector<bool> predicted_labels;

        void updateProperties(size_t& tp, size_t& fp, size_t& fn, size_t& tn);

};