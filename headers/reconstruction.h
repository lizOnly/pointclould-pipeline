

class Reconstruction
{
    public:

        Reconstruction();

        ~Reconstruction();

        double getInteriorRatio() { return interior_ratio; };

        void setGroundTruthMap() {

            ground_truth_map["wall"] = {0};
            ground_truth_map["column"] = {0};
            ground_truth_map["board"] = {0};
            ground_truth_map["window"] = {0};

            ground_truth_map["floor"] = {1};
            ground_truth_map["ceiling"] = {1}; // we define its class same as floor
            ground_truth_map["beam"] = {1};

            ground_truth_map["door"] = {7};
            
            ground_truth_map["chair"] = {4};
            ground_truth_map["sofa"] = {5};
            ground_truth_map["table"] = {6};
            ground_truth_map["bookcase"] = {9};
            
            
            ground_truth_map["clutter"] = {25};

        }

        void poissonReconstruction(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        
        void marchingCubesReconstruction(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        
        void pointCloudReconstructionFromTxt(std::string path);

        void buildGroundTruthCloud(std::string folder);

        void pcd2ply(std::string path);

        void ply2pcd(std::string path);

        void createGT(std::string path);

        int findUnderScore(std::string& str);

    private:   

        double interior_ratio;

        std::map<std::string, std::vector<int>> ground_truth_map;
    
};
