#include <vector>
#include <stdexcept>
#include <unordered_map>

class Evaluation {
    public:

        Evaluation();

        ~Evaluation();    

        void setColorLabelMap() {
            
            color_label_map["wall"] = {174, 199, 232, 0};
            color_label_map["floor"] = {152, 223, 138, 1};
            // color_label_map["ceiling"] = {152, 223, 138, 1};
            color_label_map["cabinet"] = {31, 119, 180, 2};
            color_label_map["bed"] = {255, 187, 120, 3};
            color_label_map["chair"] = {188, 189, 34, 4};
            color_label_map["sofa"] = {140, 86, 75, 5};
            color_label_map["table"] = {255, 152, 150, 6};
            color_label_map["door"] = {214, 39, 40, 7};
            color_label_map["window"] = {197, 156, 148, 8};
            color_label_map["bookshelf"] = {148, 103, 189, 9};
            color_label_map["picture"] = {196, 156, 148, 10};
            color_label_map["counter"] = {23, 190, 207, 11};
            color_label_map["desk"] = {247, 182, 210, 12};
            color_label_map["curtain"] = {219, 219, 141, 13};
            color_label_map["refrigerator"] = {255, 127, 14, 14};
            color_label_map["shower curtain"] = {158, 218, 229, 15};
            color_label_map["toilet"] = {44, 160, 44, 16};
            color_label_map["sink"] = {112, 128, 144, 17};
            color_label_map["bathtub"] = {227, 119, 194, 18};
            color_label_map["otherfurniture"] = {82, 84, 163, 19};

        }

        void setGroundTruthMap() {

            ground_truth_map["wall"] = {0};
            ground_truth_map["floor"] = {1};
            ground_truth_map["ceiling"] = {1}; // we define its class same as floor
            ground_truth_map["chair"] = {4};
            ground_truth_map["sofa"] = {5};
            ground_truth_map["table"] = {6};
            ground_truth_map["door"] = {7};
            ground_truth_map["window"] = {8};
            ground_truth_map["bookcase"] = {9};
            ground_truth_map["beam"] = {20};
            ground_truth_map["board"] = {21};
            ground_truth_map["clutter"] = {25};
            ground_truth_map["column"] = {26};

        }

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

        std::map<std::string, std::vector<int>> color_label_map;
        std::map<std::string, std::vector<int>> ground_truth_map;

        void updateProperties();

};