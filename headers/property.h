class Property
{ 
    public:
        Property();
        ~Property();
        void calculateDensity(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        void calculateDistribution(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        void calculateLocalPointNeighborhood(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        void boundaryEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double angle_threshold, std::string input_file);
        double volumeEstimation(pcl::PolygonMesh mesh);

    private:
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        std::vector<double> densities;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_density;

};