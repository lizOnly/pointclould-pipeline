#include <vector>
#include <stdexcept>
#include <unordered_map>

class Evaluation {
    public:
        Evaluation();
        ~Evaluation();    
        
        // labels trained in Minkowski Engine
        // std::map<std::string semantic_label, int color[4]> label_color_map;
        // label_color_map["wall"] = {255, 0, 0, 0};
        // label_color_map["floor"] = {0, 255, 0, 1};
        // label_color_map["cabinet"] = {0, 0, 255 ,2};
        // label_color_map["bed"] = {255, 255, 0 ,3};
        // label_color_map["chair"] = {255, 0, 255, 4};
        // label_color_map["sofa"] = {0, 255, 255, 5};
        // label_color_map["table"] = {255, 255, 255, 6};
        // label_color_map["door"] = {192, 192, 192, 7};
        // label_color_map["window"] = {128, 128, 128, 8};
        // label_color_map["bookshelf"] = {128, 0, 0, 9};
        // label_color_map["picture"] = {128, 128, 0, 10};
        // label_color_map["counter"] = {0, 128, 0, 11};
        // label_color_map["blinds"] = {128, 0, 128, 12};
        // label_color_map["desk"] = {0, 128, 128, 13};
        // label_color_map["shelves"] = {0, 0, 128, 14};
        // label_color_map["curtain"] = {255, 215, 0, 15};
        // label_color_map["dresser"] = {0, 0, 0, 16};
        // label_color_map["pillow"] = {220, 20, 60, 17};
        // label_color_map["mirror"] = {255, 0, 0, 18};
        // label_color_map["floor mat"] = {0, 255, 0, 19};
        // label_color_map["clothes"] = {0, 0, 255, 20};
        // label_color_map["ceiling"] = {255, 255, 0, 21};
        // label_color_map["books"] = {255, 0, 255, 22};
        // label_color_map["fridge"] = {0, 255, 255, 23};
        // label_color_map["tv"] = {192, 192, 192, 24};
        // label_color_map["paper"] = {128, 128, 128, 25};

        

        int findUnderScore(std::string& str);
        
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