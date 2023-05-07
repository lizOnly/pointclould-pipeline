class Property
{ 
    public:
        Property();
        ~Property();
        void calculateDensity(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        void calculateDistribution(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        void calculateLocalPointNeighborhood(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    private:
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        std::vector<double> densities;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_density;

};