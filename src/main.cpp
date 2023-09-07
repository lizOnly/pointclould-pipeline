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
#include "../headers/helper.h"


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

            occlusion.setPointRadius(0.025);
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

            double rayOcclusionLevel = occlusion.rayBasedOcclusionLevel(minPt, maxPt, 10000, polygonClouds, allCoefficients);

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
        std::cout << "You have to provide at least 1 argument" << std::endl;
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
    std::cout << "" << std::endl;

    Helper helper;
    // compute mesh based occlusion level
    if (arg1 == "-moc") {

        auto occlusion_mesh = j.at("occlusion").at("mesh");

        int pattern = occlusion_mesh.at("pattern");
        float octree_resolution = occlusion_mesh.at("octree_resolution");
        bool enable_acceleration = occlusion_mesh.at("enable_acceleration");
        double samples_per_unit_area = occlusion_mesh.at("samples_per_unit_area");
        bool use_ply = occlusion_mesh.at("use_ply");

        Occlusion occlusion;

        if (use_ply) {
            std::string ply_path = occlusion_mesh.at("ply_path");
            std::cout << "ply path is: " << ply_path << std::endl;
            std::cout << "" << std::endl;
            occlusion.parseTrianglesFromPLY(ply_path);
        } else {
            std::string mesh_path = occlusion_mesh.at("path");
            std::cout << "mesh path is: " << mesh_path << std::endl;
            std::cout << "" << std::endl;
            occlusion.parseTrianglesFromOBJ(mesh_path);
        }
        occlusion.setOctreeResolution(octree_resolution);
        occlusion.haltonSampleTriangle(samples_per_unit_area);
        occlusion.buildOctreeCloud();
        occlusion.buildCompleteOctreeNodesTriangle();
        
        Eigen::Vector3d min = occlusion.getMeshMinVertex();
        Eigen::Vector3d max = occlusion.getMeshMaxVertex();
        Eigen::Vector3d center = (min + max) / 2.0;

        std::vector<Eigen::Vector3d> origins = occlusion.viewPointPattern(pattern, min, max, center);
        
        occlusion.generateRayFromTriangle(origins);
        double occlusion_level = occlusion.triangleBasedOcclusionLevel(enable_acceleration);
        
        std::cout << "" << std::endl;
        std::cout << "Mesh based occlusion level is: " << occlusion_level << std::endl;
        std::cout << "" << std::endl;
        occlusion.generateCloudFromIntersection();

        helper.displayRunningTime(start);

    } else if (arg1 == "-bounoc") {
        // calculate based on how many intersections a ray has with the boundary of the point cloud

        auto occlusion_boundary = j.at("occlusion").at("boundary_cloud");

        bool use_estimated_cloud = occlusion_boundary.at("use_estimated_cloud");
        std::string path = occlusion_boundary.at("path");   
        std::cout << "input cloud path is: " << path << std::endl;
        std::cout << "" << std::endl;

        pcl::PointCloud<pcl::PointXYZI>::Ptr bound_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::io::loadPCDFile<pcl::PointXYZI>(path, *bound_cloud);

        std::string polygon_path = occlusion_boundary.at("polygon_path");
        std::cout << "polygon path is: " << polygon_path << std::endl;
        std::cout << "" << std::endl;

        size_t num_rays = occlusion_boundary.at("num_rays");
        double point_radius = occlusion_boundary.at("point_radius");
        float octree_resolution = occlusion_boundary.at("octree_resolution");
        bool use_openings = occlusion_boundary.at("use_openings");
        int K_nearest = occlusion_boundary.at("K_nearest");

        Occlusion occlusion;
        occlusion.setPointRadius(point_radius);
        occlusion.setOctreeResolution(octree_resolution);
        occlusion.setInputCloudBound(bound_cloud);

        pcl::PointXYZI min_pt, max_pt;
        

        if (use_estimated_cloud) {
            
            std::string sample_cloud_path = occlusion_boundary.at("sample_cloud_path");
            std::cout << "sample cloud path is: " << sample_cloud_path << std::endl;
            pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::io::loadPCDFile<pcl::PointXYZ>(sample_cloud_path, *sample_cloud);
            occlusion.setInputSampleCloud(sample_cloud);
            occlusion.estimateSemantics(K_nearest);

            pcl::PointCloud<pcl::PointXYZI>::Ptr estimated_bound_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            estimated_bound_cloud = occlusion.getEstimatedBoundCloud();
            pcl::getMinMax3D(*estimated_bound_cloud, min_pt, max_pt);

        } else {
            pcl::getMinMax3D(*bound_cloud, min_pt, max_pt);
        }

        pcl::PointXYZ min_pt_bound(min_pt.x, min_pt.y, min_pt.z);
        pcl::PointXYZ max_pt_bound(max_pt.x, max_pt.y, max_pt.z);

        occlusion.buildCompleteOctreeNodes(use_estimated_cloud);

        if (use_openings) {
            std::cout << "Using openings" << std::endl;
            std::vector<std::vector<pcl::PointXYZ>> polygons = occlusion.parsePolygonData(polygon_path);
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygon_clouds;
            std::vector<pcl::ModelCoefficients::Ptr> all_coefficients;

            for (int i = 0; i < polygons.size(); i++) {
                pcl::ModelCoefficients::Ptr coefficients = occlusion.computePlaneCoefficients(polygons[i]);
                all_coefficients.push_back(coefficients);
                pcl::PointCloud<pcl::PointXYZ>::Ptr polygon = occlusion.estimatePolygon(polygons[i], coefficients);
                polygon_clouds.push_back(polygon);
            }

            occlusion.setPolygonClouds(polygon_clouds);
            occlusion.setAllCoefficients(all_coefficients);
            
        } else {
            std::cout << "Not using openings" << std::endl;
        }

        occlusion.generateRandomRays(num_rays, min_pt_bound, max_pt_bound);

        double randomRayBasedOcclusionLevel = occlusion.randomRayBasedOcclusionLevel(use_openings, use_estimated_cloud);
        std::cout << "" << std::endl;
        std::cout << "Random ray based occlusion level is: " << randomRayBasedOcclusionLevel << std::endl;
        std::cout << "" << std::endl;
        helper.displayRunningTime(start);
    
    } else if (arg1 == "-fscan") {
        
        Scanner scanner;
        auto scan = j.at("fixed_sphere_scanner");
        std::string path = scan.at("path");

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::io::loadPCDFile<pcl::PointXYZRGB>(path, *colored_cloud);

        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);

        std::string gt_path = scan.at("gt_path");
        std::cout << "input cloud path is: " << path << std::endl;
        size_t num_rays_per_vp = scan.at("num_rays_per_vp");
        int pattern = scan.at("pattern");
        int pattern_v1 = scan.at("pattern_v1");
        int pattern_v2 = scan.at("pattern_v2");
        float octree_resolution = scan.at("octree_resolution");
        double point_radius = scan.at("point_radius");

        scanner.setOctreeResolution(octree_resolution);
        scanner.setPointRadius(point_radius);

        pcl::PointCloud<pcl::PointXYZI>::Ptr gt_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::io::loadPCDFile<pcl::PointXYZI>(gt_path, *gt_cloud);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sacnned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        std::vector<pcl::PointXYZ> center_scanning = scanner.fixed_scanning_positions(min_pt, max_pt, pattern);
        sacnned_cloud = scanner.sphere_scanner(num_rays_per_vp, pattern, center_scanning, cloud, gt_cloud, colored_cloud, path);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sacnned_cloud_v1(new pcl::PointCloud<pcl::PointXYZRGB>);
        std::vector<pcl::PointXYZ> v1_scanning = scanner.fixed_scanning_positions(min_pt, max_pt, pattern_v1);
        sacnned_cloud_v1 = scanner.sphere_scanner(num_rays_per_vp, pattern_v1, v1_scanning, cloud, gt_cloud, colored_cloud, path);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sacnned_cloud_v2(new pcl::PointCloud<pcl::PointXYZRGB>);
        std::vector<pcl::PointXYZ> v2_scanning = scanner.fixed_scanning_positions(min_pt, max_pt, pattern_v2);
        sacnned_cloud_v2 = scanner.sphere_scanner(num_rays_per_vp, pattern_v2, v2_scanning, cloud, gt_cloud, colored_cloud, path);

        helper.displayRunningTime(start);

    } else if (arg1 == "-recon") {
        
        // Since we use S3d as our dataset, we have to reconstruct the ground truth point cloud from .txt file

        auto recon = j.at("recon");
        std::string path = recon.at("path");

        Reconstruction reconstruct;

        reconstruct.pointCloudReconstructionFromTxt(path);

        helper.displayRunningTime(start);

    } else if (arg1 == "-recongt") {

        // Since we use S3d as our dataset, we have to reconstruct the ground truth point cloud from .txt file

        auto recon = j.at("recon");
        std::string gt_path = recon.at("gt_path");

        Reconstruction reconstruct;

        reconstruct.setGroundTruthMap();
        reconstruct.buildGroundTruthCloud(gt_path);

        helper.displayRunningTime(start);

    } else if (arg1 == "-eval") {
        Evaluation eval;
        auto evaluation = j.at("evaluation");
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

    } else if (arg1 == "-t2ply") {
        Reconstruction recon;
        std::string path = j.at("transfer").at("path_pcd");
        std::cout << "input cloud path is: " << path << std::endl;

        recon.pcd2ply(path);

        helper.displayRunningTime(start);

    } else if (arg1 == "-h") {
        
        std::map<std::string, std::string> instructions;
        instructions["-moc"] = "Compute occlusion level of a mesh";
        instructions["-poc"] = "Compute occlusion level of a point cloud";
        instructions["-rgoc"] = "Compute occlusion level of region growing segmentation generated mesh";
        instructions["-h"] = "help";
        instructions["-scan="] = "Scan a point cloud from a certain viewpoint";

        for (auto const& x : instructions) {
            std::cout << x.first << ": " << x.second << std::endl;
        }
        
    } else if (arg1 == "-ts") {
        
        std::string path = "../files/triangle.obj";
        int samples_per_unit_area = 50;

        Occlusion occlusion;
        occlusion.parseTrianglesFromOBJ(path);
        occlusion.uniformSampleTriangle(samples_per_unit_area);
        occlusion.haltonSampleTriangle(samples_per_unit_area);

        pcl::PointXYZ center;
        center.x = 0.0;
        center.y = 0.0;
        center.z = 0.0;

        occlusion.HaltonSampleSphere(center, 5000);


    } else {
            std::cout << "Invalid argument" << std::endl;
    }

   return 0;
}
