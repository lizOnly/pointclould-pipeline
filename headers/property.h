class Property
{ 
    public:
        Property();
        ~Property();
        double computeDensity(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr computeDensityGaussian(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        
        void calculateLocalPointNeighborhood(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        
        void boundaryEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double angle_threshold, std::string input_file);

    private:
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        std::vector<double> densities;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_density;

};