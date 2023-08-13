#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <sys/socket.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/octree/octree.h>

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <nlohmann/json.hpp>

#include "../headers/reconstruction.h"
#include "../headers/evaluation.h"
#include "../headers/property.h"
#include "../headers/occlusion.h"
#include "../headers/scanner.h"


using json = nlohmann::json;


class DataHolder {

    private:
        std::string file_name;
        std::string input_path;
        std::string polygon_data;
        std::vector<std::vector<pcl::PointXYZ>> polygons;
        std::string segmentation_path;
        std::string gt_path;

    public:
        DataHolder() {};
        ~DataHolder() {};

        void setFileName(std::string file_name) {
            this->file_name = file_name;
        }

        void setInputPath(std::string input_path) {
            this->input_path = input_path;
        }

        void setPolygonData(std::string polygon_data) {
            this->polygon_data = polygon_data;
        }

        void setPolygons(std::vector<std::vector<pcl::PointXYZ>> polygons) {
            this->polygons = polygons;
        }

        void setSegmentationPath(std::string segmentation_path) {
            this->segmentation_path = segmentation_path;
        }

        void setGtPath(std::string gt_path) {
            this->gt_path = gt_path;
        }

        std::string getFileName() {
            return this->file_name;
        }

        std::string getInputPath() {
            return this->input_path;
        }

        std::string getPolygonData() {
            return this->polygon_data;
        }

        std::vector<std::vector<pcl::PointXYZ>> getPolygons() {
            return this->polygons;
        }

        std::string getSegmentationPath() {
            return this->segmentation_path;
        }

        std::string getGtPath() {
            return this->gt_path;
        }

};

typedef websocketpp::server<websocketpp::config::asio> server;

void on_message(server& s, websocketpp::connection_hdl hdl, server::message_ptr msg, DataHolder& data_holder) {
    std::cout << "on_message called with hdl: " << hdl.lock().get()
              << " and message: " << msg->get_payload()
              << std::endl;
    Occlusion occlusion;
    // Echo the message back
    try {
        s.send(hdl, msg->get_payload(), msg->get_opcode());

        std::string payload = msg->get_payload();

        if (payload.substr(0, 3) == "-i=") {
            data_holder.setFileName(payload.substr(3, payload.length()));
            data_holder.setInputPath("../files/" + payload.substr(3, payload.length()));
        }
        if (payload.substr(0, 3) == "-p=") {
            data_holder.setPolygonData(payload.substr(3, payload.length()));
            std::string polygon_data = data_holder.getPolygonData();
            std::vector<std::vector<pcl::PointXYZ>> polygons = occlusion.parsePointString(polygon_data);
            data_holder.setPolygons(polygons);
        }
        if (payload.substr(0, 2) == "-o") {
            int pattern;
            std::string file_name = data_holder.getFileName();
            std::string input_path = data_holder.getInputPath();
            std::vector<std::vector<pcl::PointXYZ>> polygons = data_holder.getPolygons();

            int length = file_name.length();
            if (file_name.substr(length - 6, length - 4) == "v1") {
                pattern = 4; // max scanning
            } else if (file_name.substr(length - 6, length - 4) == "v2") {
                pattern = 5; // min scanning
            } else {
                pattern = 2; // center scanning
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_path, *cloud) == -1) {
                PCL_ERROR("Couldn't read file\n");
            }

            pcl::PointXYZ minPt, maxPt;
            pcl::getMinMax3D(*cloud, minPt, maxPt);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);          
            if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(input_path, *colored_cloud) == -1) {
                PCL_ERROR("Couldn't read file\n");
            }

            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds;
            std::vector<pcl::ModelCoefficients::Ptr> allCoefficients;
            // if user does not specify polygons, use default polygon


            std::vector<pcl::PointXYZ> polygon;

            if (polygons.size() == 0) {
                polygons.push_back(occlusion.generateDefaultPolygon());
            }

            for (int i = 0; i < polygons.size(); i++) {
                pcl::ModelCoefficients::Ptr coefficients = occlusion.computePlaneCoefficients(polygons[i]);
                allCoefficients.push_back(coefficients);

                pcl::PointCloud<pcl::PointXYZ>::Ptr polygon = occlusion.estimatePolygon(polygons[i], coefficients);
                polygonClouds.push_back(polygon);
            }

            double rayOcclusionLevel = occlusion.rayBasedOcclusionLevel(minPt, maxPt, pattern, cloud, polygonClouds, allCoefficients);

            std::cout << "rayOcclusionLevel: " << rayOcclusionLevel << std::endl;

            std::string ray_occlusion_level = "-o=" + std::to_string(rayOcclusionLevel);
            s.send(hdl, ray_occlusion_level, msg->get_opcode());

        }
        if (payload.substr(0, 3) == "-s=") {
            data_holder.setSegmentationPath("../files/" + payload.substr(3, payload.length()));
        }
        if (payload.substr(0, 4) == "-gt=") {
            data_holder.setGtPath("../files/" + payload.substr(4, payload.length()));
        }
        if (payload.substr(0, 2) == "-e") {
            std::cout << "Calculating evaluation metrics ... " << std::endl;
            std::string gt_path = data_holder.getGtPath();
            pcl::PointCloud<pcl::PointXYZI>::Ptr ground_truth_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::io::loadPCDFile<pcl::PointXYZI>(gt_path, *ground_truth_cloud);
            std::cout << "ground_truth_cloud loaded " << ground_truth_cloud->size() << std::endl;

            std::string segmentation_path = data_holder.getSegmentationPath();
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::io::loadPCDFile<pcl::PointXYZRGB>(segmentation_path, *segmented_cloud);
            std::cout << "segmented_cloud loaded " << segmented_cloud->size() << std::endl;

            Evaluation eval;
            eval.compareClouds(segmented_cloud, ground_truth_cloud);
            
            float iou = eval.calculateIoU();
            std::string iou_str = "-iou=" + std::to_string(iou);
            s.send(hdl, iou_str, msg->get_opcode());
            
            float accuracy = eval.calculateAccuracy();
            std::string accuracy_str = "-accuracy=" + std::to_string(accuracy);
            s.send(hdl, accuracy_str, msg->get_opcode());
            
            float precision = eval.calculatePrecision();
            std::string precision_str = "-precision=" + std::to_string(precision);
            s.send(hdl, precision_str, msg->get_opcode());

            float recall = eval.calculateRecall();
            std::string recall_str = "-recall=" + std::to_string(recall);
            s.send(hdl, recall_str, msg->get_opcode());
            
            float f1_score = eval.calculateF1Score();
            std::string f1_score_str = "-f1_score=" + std::to_string(f1_score);
            s.send(hdl, f1_score_str, msg->get_opcode());

        }

    } catch (const websocketpp::lib::error_code& e) {
        std::cout << "Echo failed because: " << e
                  << "(" << e.message() << ")" << std::endl;
    }
}


int main(int argc, char *argv[])
{   
    if (argc < 2) {
        std::cout << "You have to provide at least two arguments" << std::endl;
        return 0;
    }

    std::string arg1 = argv[1];

    if (arg1 == "-b") {
        std::cout << "Program now running as a backend server on port 8080 ..." << std::endl;
        DataHolder data_holder;
        server print_server;

        print_server.set_message_handler([&print_server, &data_holder](websocketpp::connection_hdl hdl, server::message_ptr msg) {
            on_message(print_server, hdl, msg, data_holder);
        });

        print_server.init_asio();
        print_server.listen(8080);
        print_server.start_accept();

        print_server.run();        

    }

    std::ifstream f("../config.json", std::ifstream::in);
    json j;
    f >> j;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Parsing arguments ... " << std::endl;

    Helper helper;

    // compute mesh based occlusion level
    switch (arg1) {
        case "-moc":
        {
            auto occlusion_mesh = j.at("occlusion").at("mesh");
            std::string mesh_path = occlusion_mesh.at("path");
            std::cout << "mesh path is: " << mesh_path << std::endl;

            int pattern = occlusion_mesh.at("pattern");
            size_t num_rays_per_vp = occlusion_mesh.at("num_rays_per_vp");

            Occlusion occlusion;

            occlusion.parseTrianglesFromOBJ(mesh_path);
            occlusion.buildOctreeCloud();
            occlusion.traverseOctreeTriangle();

            occlusion.generateCloudFromTriangle();
            occlusion.computeMeshBoundingBox();

            Eigen::AlignedBox3d bbox = occlusion.getBoundingBox();
            Eigen::Vector3d center = bbox.center();
            Eigen::Vector3d min = bbox.min();
            Eigen::Vector3d max = bbox.max();

            Eigen::Vector3d min_mid = (min + center) / 2;
            Eigen::Vector3d max_mid = (max + center) / 2;

            std::vector<Eigen::Vector3d> origins = occlusion.viewPointPattern(pattern);
            
            occlusion.generateRaysWithIdx(origins, num_rays_per_vp);
            double occlusion_level = occlusion.triangleBasedOcclusionLevel(center);
            
            std::cout << "Mesh based occlusion level is: " << occlusion_level << std::endl;
            occlusion.generateCloudFromIntersection();

            helper.displayRunningTime(start);
        }
        case "-poc":
        { 
            auto occlusion_point_cloud = j.at("occlusion").at("point_cloud");

            std::string cloud_path = occlusion_point_cloud.at("path");
            std::cout << "input cloud path is: " << cloud_path << std::endl;

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud);

            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*cloud, min_pt, max_pt);

            std::string polygon_path = occlusion_point_cloud.at("polygon_path");
            std::cout << "polygon path is: " << polygon_path << std::endl;

            size_t num_rays_per_vp = occlusion_point_cloud.at("num_rays_per_vp");
            int pattern = occlusion_point_cloud.at("pattern");
            double point_radius = occlusion_point_cloud.at("point_radius");

            Occlusion occlusion;

            std::vector<std::vector<pcl::PointXYZ>> polygons = occlusion.parsePolygonData(polygon_path);
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds;
            std::vector<pcl::ModelCoefficients::Ptr> allCoefficients;

            for (int i = 0; i < polygons.size(); i++) {
                pcl::ModelCoefficients::Ptr coefficients = occlusion.computePlaneCoefficients(polygons[i]);
                allCoefficients.push_back(coefficients);

                pcl::PointCloud<pcl::PointXYZ>::Ptr polygon = occlusion.estimatePolygon(polygons[i], coefficients);
                polygonClouds.push_back(polygon);
            }

            double rayOcclusionLevel = occlusion.rayBasedOcclusionLevel(min_pt, max_pt, num_rays_per_vp, pattern, cloud, polygonClouds, allCoefficients);  

            helper.displayRunningTime(start);
        }
        case "-rgoc":
        {
            auto occlusion_rg_mesh = j.at("occlusion").at("rg_mesh");
            
            std::string path = occlusion_rg_mesh.at("path");
            std::cout << "input cloud path is: " << path << std::endl;
            size_t num_rays_per_vp = occlusion_rg_mesh.at("num_rays_per_vp");
            int pattern = occlusion_rg_mesh.at("pattern");

            // region growing segmentation configuration
            auto seg_config = occlusion_rg_mesh.at("seg_config");
            size_t min_cluster_size = seg_config.at("min_cluster_size");
            size_t max_cluster_size = seg_config.at("max_cluster_size");
            int num_neighbours = seg_config.at("num_neighbours");
            double smoothness_threshold = seg_config.at("smoothness_threshold");
            double curvature_threshold = seg_config.at("curvature_threshold");

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud);

            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*cloud, min_pt, max_pt);
            Eigen::Vector3d center = Eigen::Vector3d((min_pt.x + max_pt.x) / 2, (min_pt.y + max_pt.y) / 2, (min_pt.z + max_pt.z) / 2);
            Eigen::Vector3d max = Eigen::Vector3d(max_pt.x, max_pt.y, max_pt.z);
            Eigen::Vector3d min = Eigen::Vector3d(min_pt.x, min_pt.y, min_pt.z);

            Eigen::Vector3d min_mid = (min + center) / 2;
            Eigen::Vector3d max_mid = (max + center) / 2;

            std::vector<Eigen::Vector3d> origins = occlusion.viewPointPattern(pattern);

            Occlusion occlusion;
            occlusion.regionGrowingSegmentation(cloud, min_cluster_size, max_cluster_size, num_neighbours, smoothness_threshold, curvature_threshold);
            occlusion.generateTriangleFromCluster();
            occlusion.buildOctreeCloud();
            occlusion.traverseOctreeTriangle();
            occlusion.generateRaysWithIdx(origins, num_rays_per_vp);
            double occlulsion_level = occlusion.triangleBasedOcclusionLevel(center);
            
            std::cout << "triangle based occlulsion level is: " << occlulsion_level << std::endl;

            helper.displayRunningTime(start);
        }
        case "-scan":
        {
            auto scanner = j.at("scanner");
            std::string path = scanner.at("path");

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::io::loadPCDFile<pcl::PointXYZRGB>(path, *colored_cloud);

            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*cloud, min_pt, max_pt);

            std::string gt_path = scanner.at("gt_path");
            std::cout << "input cloud path is: " << path << std::endl;
            size_t num_rays_per_vp = scanner.at("num_rays_per_vp");
            int pattern = scanner.at("pattern");
            int pattern_v1 = scanner.at("pattern_v1");
            int pattern_v2 = scanner.at("pattern_v2");

            pcl::PointCloud<pcl::PointXYZI>::Ptr gt_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::io::loadPCDFile<pcl::PointXYZI>(gt_path, *gt_cloud);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr sacnned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

            std::vector<pcl::PointXYZ> center_scanning = scanner.scanning_positions(min_pt, max_pt, pattern);
            sacnned_cloud = scanner.multi_sphere_scanner(num_ray_sample, pattern, center_scanning, cloud, gt_cloud, colored_cloud, file_name);
            
            num_scanned_points = sacnned_cloud->width * sacnned_cloud->height;
            double sample_rate = (double) num_scanned_points / (double) num_points;
            std::cout << "center scanning sample rate: " << sample_rate << std::endl;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr sacnned_cloud_v1(new pcl::PointCloud<pcl::PointXYZRGB>);
            std::vector<pcl::PointXYZ> v1_scanning = scanner.scanning_positions(min_pt, max_pt, pattern_v1);
            sacnned_cloud_v1 = scanner.multi_sphere_scanner(num_ray_sample, pattern_v1, v1_scanning, cloud, gt_cloud, colored_cloud, file_name);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr sacnned_cloud_v2(new pcl::PointCloud<pcl::PointXYZRGB>);
            std::vector<pcl::PointXYZ> v2_scanning = scanner.scanning_positions(min_pt, max_pt, pattern_v2);
            sacnned_cloud_v2 = scanner.multi_sphere_scanner(num_ray_sample, pattern_v2, v2_scanning, cloud, gt_cloud, colored_cloud, file_name);

            helper.displayRunningTime(start);
        }
        case "-recon":
        {
            auto recon = j.at("recon");
            std::string path = recon.at("path");
            std::string gt_path = recon.at("gt_path");

            Reconstruction reconstruction;

            helper.displayRunningTime(start);
            return 0;
        }
        case "-eval":
        { 
            Evaluation eval;
            auto evaluation = j.at("eval");
            std::string seg_path = evaluation.at("seg_path");
            std::string gt_path = evaluation.at("gt_path");

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::io::loadPCDFile<pcl::PointXYZRGB>(seg_path, *segmented_cloud);

            pcl::PointCloud<pcl::PointXYZI>::Ptr ground_truth_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::io::loadPCDFile<pcl::PointXYZI>(gt_path, *ground_truth_cloud);

            eval.compareClouds(segmented_cloud, ground_truth_cloud);
            eval.calculateIoU();
            eval.calculateAccuracy();
            eval.calculatePrecision();
            eval.calculateRecall();
            eval.calculateF1Score();

            helper.displayRunningTime(start);
        }
        case "-t2ply":
        {
            Reconstruction recon;
            std::string path = j.at("transfer").at("path_pcd");
            std::cout << "input cloud path is: " << path << std::endl;

            recon.pcd2ply(path);

            helper.displayRunningTime(start);
        }
        case "-h":
        {
            std::map<std::string, std::string> args_map;
            args_map["-moc"] = "";
            args_map["-poc"] = "";
            args_map["-rgoc"] = "";
            args_map["-h"] = "";
            args_map["-rc="] = "";
        }

        default:
            std::cout << "Invalid argument" << std::endl;
            return 0;

        return 0;
    }

}
