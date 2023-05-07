

class Reconstruction
{
    public:
        Reconstruction();
        ~Reconstruction();
        void poissonReconstruction(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    
    private:   
        void saveMeshAsOBJWithMTL(const pcl::PolygonMesh& mesh, const std::string& obj_filename, const std::string& mtl_filename);
};
