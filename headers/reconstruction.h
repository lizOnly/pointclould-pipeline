

class Reconstruction
{
    public:
        Reconstruction();
        ~Reconstruction();

        void poissonReconstruction(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        
        void marchingCubesReconstruction(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        
        void pointCloudReconstructionFromTxt(std::string path);

        void batchReconstructionFromTxt(std::string folder);

        void pcd2ply(std::string path);

        void ply2pcd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string file_name);

        int findUnderScore(std::string& str);

    private:   
    
};
