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
        std::string boundary_cloud_path;
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

        void setBoundaryCloudPath(std::string boundary_cloud_path) {
            this->boundary_cloud_path = boundary_cloud_path;
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

        std::string getBoundaryCloudPath() {
            return this->boundary_cloud_path;
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

    std::ifstream f("../config.json", std::ifstream::in);
    json j;
    f >> j;

    std::string input_root_path = j.at("globals").at("input_root_path");

    Occlusion occlusion;
    // Echo the message back
    try {
        s.send(hdl, msg->get_payload(), msg->get_opcode());

        std::string payload = msg->get_payload();

        // each point in the input cloud should be labeled as boundary or non-boundary
        if (payload.substr(0, 3) == "-i=") {

            data_holder.setFileName(payload.substr(3, payload.length()));
            data_holder.setInputPath(input_root_path + payload.substr(3, payload.length()));

        }
        
        if (payload.substr(0, 3) == "-p=") {

            data_holder.setPolygonData(payload.substr(3, payload.length()));
            std::string polygon_data = data_holder.getPolygonData();
            std::vector<std::vector<pcl::PointXYZ>> polygons = occlusion.parsePointString(polygon_data);
            data_holder.setPolygons(polygons);

        }

        if (payload.substr(0, 2) == "-o") {

            std::string bound_path = data_holder.getBoundaryCloudPath();
            std::vector<std::vector<pcl::PointXYZ>> polygons = data_holder.getPolygons();

            pcl::PointCloud<pcl::PointXYZI>::Ptr bound_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::io::loadPCDFile<pcl::PointXYZI>(bound_path, *bound_cloud);

            size_t clutter_count = 0;
            for (auto& p : bound_cloud->points) {
                if (p.intensity == 0) {
                    clutter_count++;
                }
            }

            if (clutter_count == bound_cloud->size()) {
                std::cout << "Indicating that input cloud has no intensity field, now we have to change all i value to 1" << std::endl;
                for (auto& p : bound_cloud->points) {
                    p.intensity = 1;
                }
            }

            pcl::PointXYZI min_pt, max_pt;
            pcl::getMinMax3D(*bound_cloud, min_pt, max_pt);
            pcl::PointXYZ min_pt_bound(min_pt.x, min_pt.y, min_pt.z);
            pcl::PointXYZ max_pt_bound(max_pt.x, max_pt.y, max_pt.z);

            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds;
            std::vector<pcl::ModelCoefficients::Ptr> allCoefficients;
            // if user does not specify polygons, use default polygon

            occlusion.setPointRadius(0.2);
            occlusion.setOctreeResolution(0.5);
            occlusion.setInputCloudBound(bound_cloud);
            occlusion.buildCompleteOctreeNodes();

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

            occlusion.setPolygonClouds(polygonClouds);
            occlusion.setAllCoefficients(allCoefficients);

            occlusion.generateRandomRays(10000, min_pt_bound, max_pt_bound);

            double occlusion_level = occlusion.randomRayBasedOcclusionLevel(true);
            std::string occlusion_level_str = "-occlusion_level=" + std::to_string(occlusion_level);

            s.send(hdl, occlusion_level_str, msg->get_opcode());
        }

        if (payload.substr(0, 3) ==  "-b=") {

            data_holder.setFileName(payload.substr(3, payload.length()));
            data_holder.setBoundaryCloudPath(input_root_path + payload.substr(3, payload.length()));

        }

        if (payload.substr(0, 3) == "-s=") {

            data_holder.setSegmentationPath(input_root_path + payload.substr(3, payload.length()));

        }
        if (payload.substr(0, 4) == "-gt=") {

            data_holder.setGtPath(input_root_path + payload.substr(4, payload.length()));

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

            eval.setColorLabelMap();
            eval.setGroundTruthMap();

            eval.compareClouds(segmented_cloud, ground_truth_cloud, false);
            
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

    std::string input_root_path = j.at("globals").at("input_root_path");
    std::cout << "input root path is: " << input_root_path << std::endl;
    std::cout << "" << std::endl;

    std::string output_root_path = j.at("globals").at("output_root_path");
    std::cout << "output path is: " << output_root_path << std::endl;
    std::cout << "" << std::endl;

    Helper helper;
    // compute mesh based occlusion level
    if (arg1 == "-moc") {

        //create occlusion levels array
        std::vector<double> occlusion_levels;

        for (int i = 0; i < 6; i++)
        {

            auto occlusion_mesh = j.at("occlusion").at("mesh");

            int pattern = occlusion_mesh.at("pattern");
            float octree_resolution = occlusion_mesh.at("octree_resolution");
            double samples_per_unit_area = occlusion_mesh.at("samples_per_unit_area");
            bool use_ply = occlusion_mesh.at("use_ply");

            Occlusion occlusion;
            std::string shape_name;

            if (use_ply) {

                std::string ply_path = occlusion_mesh.at("ply_path");
                ply_path = input_root_path + ply_path;
                std::cout << "Estimated .ply mesh path is: " << ply_path << std::endl;
                std::cout << "" << std::endl;
                occlusion.parseTrianglesFromPLY(ply_path);
                shape_name = ply_path;

            } else {

                std::string mesh_path = occlusion_mesh.at("path");
                mesh_path = input_root_path + mesh_path;
                std::cout << "mesh path is: " << mesh_path << std::endl;
                std::cout << "" << std::endl;
                occlusion.parseTrianglesFromOBJ(mesh_path);
                shape_name = mesh_path;

            }

            //remove .ply and all path extension from ply_path
            shape_name = shape_name.substr(0, shape_name.find_last_of("."));
            shape_name = shape_name.substr(shape_name.find_last_of("/") + 1, shape_name.length());

            occlusion.setOutputRootPath(output_root_path);
            occlusion.setShapeName(shape_name);
            occlusion.setOctreeResolution(octree_resolution);
            occlusion.setPattern(i);
            occlusion.setSamplesPerUnitArea(samples_per_unit_area);
            occlusion.haltonSampleTriangle(samples_per_unit_area);
            occlusion.buildCompleteOctreeNodesTriangle();

            Eigen::Vector3d min = occlusion.getMeshMinVertex();
            Eigen::Vector3d max = occlusion.getMeshMaxVertex();
            Eigen::Vector3d center = (min + max) / 2.0;


            std::vector<Eigen::Vector3d> origins = occlusion.viewPointPattern(min, max, center);

            occlusion.generateRayFromTriangle(origins);

            double occlusion_level = occlusion.triangleBasedOcclusionLevel();
            occlusion_levels.push_back(occlusion_level);

            std::cout << "" << std::endl;
            std::cout << "Mesh based occlusion level is: " << occlusion_level << std::endl;
            std::cout << "" << std::endl;
        }
        //print all occlusion levels with the pattern
        for (int i = 0; i < occlusion_levels.size(); i++) {
            std::cout << "Pattern " << i << " occlusion level is: " << occlusion_levels[i] << std::endl;
        }

        helper.displayRunningTime(start);


    } else if (arg1 == "-bounoc") {

        auto occlusion_boundary = j.at("occlusion").at("boundary_cloud");

        std::string path = occlusion_boundary.at("path");
        path = input_root_path + path;
        std::cout << "input cloud path is: " << input_root_path + path << std::endl;
        std::cout << "" << std::endl;

        pcl::PointCloud<pcl::PointXYZI>::Ptr bound_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::io::loadPCDFile<pcl::PointXYZI>(path, *bound_cloud);

        size_t clutter_count = 0;
        for (auto& p : bound_cloud->points) {
            if (p.intensity == 0) {
                clutter_count++;
            }
        }

        if (clutter_count == bound_cloud->size()) {
            std::cout << "Indicating that input cloud has no intensity field, now we have to change all i value to 1" << std::endl;
            for (auto& p : bound_cloud->points) {
                p.intensity = 1;
            }
        }

        std::string polygon_path = occlusion_boundary.at("polygon_path");
        polygon_path = input_root_path + polygon_path;
        std::cout << "polygon path is: " << polygon_path << std::endl;
        std::cout << "" << std::endl;

        size_t num_rays = occlusion_boundary.at("num_rays");
        double point_radius = occlusion_boundary.at("point_radius");
        float octree_resolution = occlusion_boundary.at("octree_resolution");
        bool use_openings = occlusion_boundary.at("use_openings");
        int K_nearest = occlusion_boundary.at("K_nearest");

        Occlusion occlusion;
        occlusion.setOutputRootPath(output_root_path);
        occlusion.setPointRadius(point_radius);
        occlusion.setOctreeResolution(octree_resolution);
        occlusion.setInputCloudBound(bound_cloud);

        pcl::PointXYZI min_pt, max_pt;
        pcl::getMinMax3D(*bound_cloud, min_pt, max_pt);
        pcl::PointXYZ min_pt_bound(min_pt.x, min_pt.y, min_pt.z);
        pcl::PointXYZ max_pt_bound(max_pt.x, max_pt.y, max_pt.z);

        occlusion.buildCompleteOctreeNodes();

        if (use_openings) {

            std::cout << "Using openings ... " << std::endl;
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

        double randomRayBasedOcclusionLevel = occlusion.randomRayBasedOcclusionLevel(use_openings);
       
        std::cout << "" << std::endl;
        std::cout << "Random ray based occlusion level is: " << randomRayBasedOcclusionLevel << std::endl;
        std::cout << "" << std::endl;
        
        helper.displayRunningTime(start);
    
    }
    else if (arg1 == "-bounoc_ratios") {

        //create a simple matrix
        double occlusion_ratios [5][3];
        std::vector<double> radius = {0.5, 1, 0.5};
        //iterate the patterns from 0 to 6
        for (int pattern = 0; pattern < 6; ++pattern) {
            //iterate the radius
            for (int r = 0; r < radius.size(); ++r) {

                auto occlusion_boundary = j.at("occlusion").at("boundary_cloud");

                std::string path = occlusion_boundary.at("path");
                path = input_root_path + path;
                std::cout << "input cloud path is: " << input_root_path + path << std::endl;
                std::cout << "" << std::endl;

                //replace path with the new pattern
                std::string pattern_str = std::to_string(pattern);
                path.replace(path.find("0"), 1, pattern_str);

                pcl::PointCloud<pcl::PointXYZI>::Ptr bound_cloud(new pcl::PointCloud<pcl::PointXYZI>);
                pcl::io::loadPCDFile<pcl::PointXYZI>(path, *bound_cloud);

                size_t clutter_count = 0;
                for (auto &p: bound_cloud->points) {
                    if (p.intensity == 0) {
                        clutter_count++;
                    }
                }

                if (clutter_count == bound_cloud->size()) {
                    std::cout
                            << "Indicating that input cloud has no intensity field, now we have to change all i value to 1"
                            << std::endl;
                    for (auto &p: bound_cloud->points) {
                        p.intensity = 1;
                    }
                }

                std::string polygon_path = occlusion_boundary.at("polygon_path");
                polygon_path = input_root_path + polygon_path;
                std::cout << "polygon path is: " << polygon_path << std::endl;
                std::cout << "" << std::endl;

                size_t num_rays = occlusion_boundary.at("num_rays");
                //double point_radius = occlusion_boundary.at("point_radius");
                double point_radius = radius[r];
                float octree_resolution = occlusion_boundary.at("octree_resolution");
                bool use_openings = occlusion_boundary.at("use_openings");
                int K_nearest = occlusion_boundary.at("K_nearest");

                Occlusion occlusion;
                occlusion.setOutputRootPath(output_root_path);
                occlusion.setPointRadius(point_radius);
                occlusion.setOctreeResolution(octree_resolution);
                occlusion.setInputCloudBound(bound_cloud);

                pcl::PointXYZI min_pt, max_pt;
                pcl::getMinMax3D(*bound_cloud, min_pt, max_pt);
                pcl::PointXYZ min_pt_bound(min_pt.x, min_pt.y, min_pt.z);
                pcl::PointXYZ max_pt_bound(max_pt.x, max_pt.y, max_pt.z);

                occlusion.buildCompleteOctreeNodes();

                if (use_openings) {

                    std::cout << "Using openings ... " << std::endl;
                    std::vector<std::vector<pcl::PointXYZ>> polygons = occlusion.parsePolygonData(polygon_path);
                    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygon_clouds;
                    std::vector<pcl::ModelCoefficients::Ptr> all_coefficients;

                    for (int i = 0; i < polygons.size(); i++) {
                        pcl::ModelCoefficients::Ptr coefficients = occlusion.computePlaneCoefficients(polygons[i]);
                        all_coefficients.push_back(coefficients);
                        pcl::PointCloud<pcl::PointXYZ>::Ptr polygon = occlusion.estimatePolygon(polygons[i],
                                                                                                coefficients);
                        polygon_clouds.push_back(polygon);
                    }

                    occlusion.setPolygonClouds(polygon_clouds);
                    occlusion.setAllCoefficients(all_coefficients);

                } else {
                    std::cout << "Not using openings" << std::endl;
                }

                occlusion.generateRandomRays(num_rays, min_pt_bound, max_pt_bound);

                double randomRayBasedOcclusionLevel = occlusion.randomRayBasedOcclusionLevel(use_openings);

                std::cout << "" << std::endl;
                std::cout << "Random ray based occlusion level is: " << randomRayBasedOcclusionLevel << std::endl;
                std::cout << "" << std::endl;

                // fill randomRayBasedOcclusionLevel into the matrix
                occlusion_ratios[pattern][r] = randomRayBasedOcclusionLevel;

                helper.displayRunningTime(start);
            }
        }

        // print occlusion ratios matrix
        for (int i = 0; i < 6; ++i) {
            std::cout << "Pattern " << i << " occlusion level is: " << std::endl;
            for (int j = 0; j < 3; ++j) {
                std::cout << occlusion_ratios[i][j] << " ";
            }
            std::cout << std::endl;
        }

    }

    else if (arg1 == "-bounoc_rays") {

        //create a simple matrix
        double occlusion_ratios [5][4];
        std::vector<double> rays = {10, 100, 1000, 10000};
        //iterate the patterns from 0 to 6
        for (int pattern = 0; pattern < 6; ++pattern) {
            //iterate the radius
            for (int r = 0; r < rays.size(); ++r) {

                auto occlusion_boundary = j.at("occlusion").at("boundary_cloud");

                std::string path = occlusion_boundary.at("path");
                path = input_root_path + path;
                std::cout << "input cloud path is: " << input_root_path + path << std::endl;
                std::cout << "" << std::endl;

                //replace path with the new pattern
                std::string pattern_str = std::to_string(pattern);
                path.replace(path.find("0"), 1, pattern_str);

                pcl::PointCloud<pcl::PointXYZI>::Ptr bound_cloud(new pcl::PointCloud<pcl::PointXYZI>);
                pcl::io::loadPCDFile<pcl::PointXYZI>(path, *bound_cloud);

                size_t clutter_count = 0;
                for (auto &p: bound_cloud->points) {
                    if (p.intensity == 0) {
                        clutter_count++;
                    }
                }

                if (clutter_count == bound_cloud->size()) {
                    std::cout
                            << "Indicating that input cloud has no intensity field, now we have to change all i value to 1"
                            << std::endl;
                    for (auto &p: bound_cloud->points) {
                        p.intensity = 1;
                    }
                }

                std::string polygon_path = occlusion_boundary.at("polygon_path");
                polygon_path = input_root_path + polygon_path;
                std::cout << "polygon path is: " << polygon_path << std::endl;
                std::cout << "" << std::endl;

                //size_t num_rays = occlusion_boundary.at("num_rays");
                size_t num_rays = rays[r];
                double point_radius = occlusion_boundary.at("point_radius");
                //double point_radius = r;
                float octree_resolution = occlusion_boundary.at("octree_resolution");
                bool use_openings = occlusion_boundary.at("use_openings");
                int K_nearest = occlusion_boundary.at("K_nearest");

                Occlusion occlusion;
                occlusion.setOutputRootPath(output_root_path);
                occlusion.setPointRadius(point_radius);
                occlusion.setOctreeResolution(octree_resolution);
                occlusion.setInputCloudBound(bound_cloud);

                pcl::PointXYZI min_pt, max_pt;
                pcl::getMinMax3D(*bound_cloud, min_pt, max_pt);
                pcl::PointXYZ min_pt_bound(min_pt.x, min_pt.y, min_pt.z);
                pcl::PointXYZ max_pt_bound(max_pt.x, max_pt.y, max_pt.z);

                occlusion.buildCompleteOctreeNodes();

                if (use_openings) {

                    std::cout << "Using openings ... " << std::endl;
                    std::vector<std::vector<pcl::PointXYZ>> polygons = occlusion.parsePolygonData(polygon_path);
                    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygon_clouds;
                    std::vector<pcl::ModelCoefficients::Ptr> all_coefficients;

                    for (int i = 0; i < polygons.size(); i++) {
                        pcl::ModelCoefficients::Ptr coefficients = occlusion.computePlaneCoefficients(polygons[i]);
                        all_coefficients.push_back(coefficients);
                        pcl::PointCloud<pcl::PointXYZ>::Ptr polygon = occlusion.estimatePolygon(polygons[i],
                                                                                                coefficients);
                        polygon_clouds.push_back(polygon);
                    }

                    occlusion.setPolygonClouds(polygon_clouds);
                    occlusion.setAllCoefficients(all_coefficients);

                } else {
                    std::cout << "Not using openings" << std::endl;
                }

                occlusion.generateRandomRays(num_rays, min_pt_bound, max_pt_bound);

                double randomRayBasedOcclusionLevel = occlusion.randomRayBasedOcclusionLevel(use_openings);

                std::cout << "" << std::endl;
                std::cout << "Random ray based occlusion level is: " << randomRayBasedOcclusionLevel << std::endl;
                std::cout << "" << std::endl;

                // fill randomRayBasedOcclusionLevel into the matrix
                occlusion_ratios[pattern][r] = randomRayBasedOcclusionLevel;

                helper.displayRunningTime(start);
            }
        }

        // print occlusion ratios matrix
        for (int i = 0; i < 6; ++i) {
            std::cout << "Pattern " << i << " occlusion level is: " << std::endl;
            for (int j = 0; j < 3; ++j) {
                std::cout << occlusion_ratios[i][j] << " ";
            }
            std::cout << std::endl;
        }

    }
    else if (arg1 == "-fscan") {
        
        Scanner scanner;
        auto scan = j.at("fixed_sphere_scanner");
        std::string path = scan.at("path");
        path = input_root_path + path;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud);

        scanner.setInputCloud(cloud);

        std::cout << "input cloud path is: " << path << std::endl;
        std::cout << "Loaded " << cloud->size() << " data points" << std::endl;
        std::cout << "" << std::endl;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::io::loadPCDFile<pcl::PointXYZRGB>(path, *colored_cloud);

        scanner.setInputCloudColor(colored_cloud);

        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);

        std::cout << "min_pt: " << min_pt << std::endl;
        std::cout << "max_pt: " << max_pt << std::endl;
        std::cout << "" << std::endl;

        std::string gt_path = scan.at("gt_path");
        gt_path = input_root_path + gt_path;

        scanner.setOutputRootPath(output_root_path);
        pcl::PointCloud<pcl::PointXYZI>::Ptr gt_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::io::loadPCDFile<pcl::PointXYZI>(gt_path, *gt_cloud);

        scanner.setInputCloudGT(gt_cloud);

        size_t sampling_hor = scan.at("sampling_hor");
        scanner.setSamplingHor(sampling_hor);

        size_t sampling_ver = scan.at("sampling_ver");
        scanner.setSamplingVer(sampling_ver);

        std::string scene_name = scan.at("scene_name");

        float octree_resolution = scan.at("octree_resolution");
        double point_radius = scan.at("point_radius");

        scanner.setOctreeResolution(octree_resolution);
        scanner.setPointRadius(point_radius);
        
        int pattern = scan.at("pattern");
        std::vector<pcl::PointXYZ> origins = scanner.fixed_scanning_positions(min_pt, max_pt, pattern);
        scanner.generateRays(origins);

        scanner.buildCompleteOctreeNodes();

        scanner.sphere_scanner(pattern, scene_name);
        
        helper.displayRunningTime(start);

    } else if (arg1 == "-scanm"){ // scan mesh

        Occlusion occlusion;

        auto scan_mesh = j.at("scan_mesh");
        
        std::string mesh_path = scan_mesh.at("mesh_path");
        mesh_path = input_root_path + mesh_path;
        std::cout << "mesh path is: " << mesh_path << std::endl;
        std::cout << "" << std::endl;

        //add shape name to occlusion object
        std::string shape_name;
        shape_name = mesh_path;
        shape_name = shape_name.substr(0, shape_name.find_last_of("."));
        shape_name = shape_name.substr(shape_name.find_last_of("/") + 1, shape_name.length());
        occlusion.setShapeName(shape_name);


        occlusion.parseTrianglesFromPLY(mesh_path);

        occlusion.setOutputRootPath(output_root_path);
        double octree_resolution = scan_mesh.at("octree_resolution");
        occlusion.setOctreeResolution(octree_resolution);
        
        size_t sampling_hor = scan_mesh.at("sampling_hor");
        occlusion.setSamplingHor(sampling_hor);

        size_t sampling_ver = scan_mesh.at("sampling_ver");
        occlusion.setSamplingVer(sampling_ver);

        int pattern = scan_mesh.at("pattern");
        occlusion.setPattern(pattern);

        Eigen::Vector3d min = occlusion.getMeshMinVertex();
        Eigen::Vector3d max = occlusion.getMeshMaxVertex();
        Eigen::Vector3d center = (min + max) / 2.0;

        std::vector<Eigen::Vector3d> origins = occlusion.viewPointPattern(min, max, center);

        occlusion.generateScannerRays(origins);

        occlusion.buildCompleteOctreeNodesTriangle();

        occlusion.scannerIntersectTriangle();

        helper.displayRunningTime(start);

    }
    else
        if (arg1 == "-scanm_all"){ // scan mesh
        //iterate all patterns from 0 to 6
        for (int i = 0; i < 6; i++) {

            Occlusion occlusion;

            auto scan_mesh = j.at("scan_mesh");

            std::string mesh_path = scan_mesh.at("mesh_path");
            mesh_path = input_root_path + mesh_path;
            std::cout << "mesh path is: " << mesh_path << std::endl;
            std::cout << "" << std::endl;

            //add shape name to occlusion object
            std::string shape_name;
            shape_name = mesh_path;
            shape_name = shape_name.substr(0, shape_name.find_last_of("."));
            shape_name = shape_name.substr(shape_name.find_last_of("/") + 1, shape_name.length());
            occlusion.setShapeName(shape_name);


            occlusion.parseTrianglesFromPLY(mesh_path);

            occlusion.setOutputRootPath(output_root_path);
            double octree_resolution = scan_mesh.at("octree_resolution");
            occlusion.setOctreeResolution(octree_resolution);

            size_t sampling_hor = scan_mesh.at("sampling_hor");
            occlusion.setSamplingHor(sampling_hor);

            size_t sampling_ver = scan_mesh.at("sampling_ver");
            occlusion.setSamplingVer(sampling_ver);

            //int pattern = scan_mesh.at("pattern");
            int pattern = i;
            occlusion.setPattern(pattern);

            Eigen::Vector3d min = occlusion.getMeshMinVertex();
            Eigen::Vector3d max = occlusion.getMeshMaxVertex();
            Eigen::Vector3d center = (min + max) / 2.0;

            std::vector<Eigen::Vector3d> origins = occlusion.viewPointPattern(min, max, center);

            occlusion.generateScannerRays(origins);

            occlusion.buildCompleteOctreeNodesTriangle();

            occlusion.scannerIntersectTriangle();

            helper.displayRunningTime(start);
        }

    }
    else if (arg1 == "-recon") {
        
        // Since we use S3d as our dataset, we have to reconstruct the ground truth point cloud from .txt file

        auto recon = j.at("recon");
        std::string path = recon.at("path");
        path = input_root_path + path;

        Reconstruction reconstruct;

        reconstruct.pointCloudReconstructionFromTxt(path);

        helper.displayRunningTime(start);

    } else if (arg1 == "-recongt") {

        // Since we use S3d as our dataset, we have to reconstruct the ground truth point cloud from .txt file

        auto recon = j.at("recon");
        std::string gt_path = recon.at("gt_path");
        gt_path = input_root_path + gt_path;

        Reconstruction reconstruct;

        reconstruct.setGroundTruthMap();
        reconstruct.buildGroundTruthCloud(gt_path);

        helper.displayRunningTime(start);

    } else if (arg1 == "-eval") {
        Evaluation eval;
        auto evaluation = j.at("evaluation");
        std::string seg_path = evaluation.at("seg_path");
        seg_path = input_root_path + seg_path;
        std::string gt_path = evaluation.at("gt_path");
        gt_path = input_root_path + gt_path;
        bool compare_bound = evaluation.at("compare_bound");

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::io::loadPCDFile<pcl::PointXYZRGB>(seg_path, *segmented_cloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr ground_truth_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::io::loadPCDFile<pcl::PointXYZI>(gt_path, *ground_truth_cloud);

        eval.setColorLabelMap();
        eval.setGroundTruthMap();

        eval.compareClouds(segmented_cloud, ground_truth_cloud, compare_bound);
        
        eval.calculateAccuracy();
        eval.calculateIoU();
        eval.calculatePrecision();
        eval.calculateRecall();
        eval.calculateF1Score();

        helper.displayRunningTime(start);

    } else if (arg1 == "-t2ply") {
        Reconstruction recon;
        std::string path = j.at("transfer").at("path_pcd");
        path = input_root_path + path;
        std::cout << "input cloud path is: " << path << std::endl;

        recon.pcd2ply(path);

        helper.displayRunningTime(start);

    } else if (arg1 == "-t2pcd"){

        Reconstruction recon;
        std::string path = j.at("transfer").at("path_ply");
        path = input_root_path + path;
        std::cout << "input cloud path is: " << path << std::endl;

        recon.ply2pcd(path);

        helper.displayRunningTime(start);
    
    } else if (arg1 == "-gt"){

        Reconstruction recon;
        std::string path = j.at("transfer").at("path_pcd");
        path = input_root_path + path;
        std::cout << "input cloud path is: " << path << std::endl;

        recon.createGT(path);

        helper.displayRunningTime(start);

    } else if (arg1 == "-h") {
        
        std::map<std::string, std::string> instructions;
        instructions["-moc"] = "Compute occlusion level of a mesh";
        instructions["-bounoc"] = "Compute occlusion level of a point cloud";
        instructions["-fscan"] = "Compute occlusion level of a point cloud using fixed sphere scanning";
        instructions["-scanm"] = "Scan mesh using fixed sphere scanning";
        instructions["-recon"] = "Reconstruct point cloud from .txt file";
        instructions["-recongt"] = "Reconstruct ground truth point cloud from .txt file";
        instructions["-eval"] = "Evaluate segmentation results";
        instructions["-t2ply"] = "Convert .pcd file to .ply file";
        instructions["-h"] = "help";

        for (auto const& x : instructions) {
            std::cout << x.first << ": " << x.second << std::endl;
        }
        
    } else {
            std::cout << "Invalid argument" << std::endl;
    }

   return 0;
}
