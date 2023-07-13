

class Reconstruction
{
    public:
        Reconstruction();
        ~Reconstruction();

        void poissonReconstruction(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        
        void marchingCubesReconstruction(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        
        void pointCloudReconstructionFromTxt(std::string path);

        void batchReconstructionFromTxt(std::string folder);

        void pcd2ply(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string file_name);

    private:   
        void saveMeshAsOBJWithMTL(const pcl::PolygonMesh& mesh, const std::string& obj_filename, const std::string& mtl_filename);
};
