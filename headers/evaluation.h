#include <vector>
#include <stdexcept>


class Evaluation {
    public:
        Evaluation();
        ~Evaluation();    
        
        float calculateIoU();
        float calculateAccuracy();
        float calculatePrecision();
        float calculateRecall();
        float calculateF1Score();
        

    private:
        std::vector<int> ground_truth_labels;
        std::vector<int> predicted_labels;
        
        // Point cloud data members
        std::vector<std::vector<int>> ground_truth_point_cloud;
        std::vector<std::vector<int>> predicted_point_cloud;
        
        void updateProperties(size_t& tp, size_t& fp, size_t& fn, size_t& tn);
        
        // Helper functions for point cloud operations
        float calculateDistance(const std::vector<bool>& point1, const std::vector<bool>& point2);
        std::vector<size_t> findNearestNeighbors(const std::vector<bool>& query_point, size_t k);
};