#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fstream>
#include <sstream>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>

#include "../headers/reconstruction.h"
#include "../headers/evaluation.h"
#include "../headers/property.h"
#include "../headers/helper.h"
#include "../headers/visualizer.h"
#include "../headers/scanner.h"

#include <boost/asio.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

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
    Helper helper;
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
            std::vector<std::vector<pcl::PointXYZ>> polygons = helper.parsePointString(polygon_data);
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

            pcl::PointXYZ default_point;
            default_point.x = 0;
            default_point.y = 0;
            default_point.z = 0;
            pcl::PointXYZ default_point2;
            default_point2.x = 1;
            default_point2.y = 1;
            default_point2.z = 1;
            pcl::PointXYZ default_point3;
            default_point3.x = 1;
            default_point3.y = 0;
            default_point3.z = 1;

            if (polygons.size() == 0) {
                std::vector<pcl::PointXYZ> default_polygon;
                default_polygon.push_back(default_point);
                default_polygon.push_back(default_point2);
                default_polygon.push_back(default_point3);

                pcl::ModelCoefficients::Ptr coefficients = helper.computePlaneCoefficients(default_polygon);
                allCoefficients.push_back(coefficients);

                pcl::PointCloud<pcl::PointXYZ>::Ptr polygon = helper.estimatePolygon(default_polygon, coefficients);
                polygonClouds.push_back(polygon);

            } else {
                for (int i = 0; i < polygons.size(); i++) {
                    pcl::ModelCoefficients::Ptr coefficients = helper.computePlaneCoefficients(polygons[i]);
                    allCoefficients.push_back(coefficients);

                    pcl::PointCloud<pcl::PointXYZ>::Ptr polygon = helper.estimatePolygon(polygons[i], coefficients);
                    polygonClouds.push_back(polygon);
                }
            }
            

            double search_radius = 0.1;

            double rayOcclusionLevel = helper.rayBasedOcclusionLevel(minPt, maxPt, search_radius, pattern, 
                                                                    cloud, polygonClouds, allCoefficients);

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

    // parse arguments related to input file, or --help or backend server
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

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Parsing arguments ... " << std::endl;

    std::map<std::string, std::string> args_map;
    args_map["-i="] = "--input==";
    args_map["-rs="] = "--raysample==";
    args_map["-o"] = "--occlusion";
    args_map["-sr="] = "--searchradius==";
    args_map["-h"] = "--help";
    args_map["-rc="] = "--reconstruct";
    args_map["-s="] = "--segmentation==";
    args_map["-p="] = "--polygon==";
    args_map["-sc="] = "--scan==";
    args_map["-e"] = "--evaluate"; // calculate the evaluation metrics, IoU, F1, etc.
    args_map["-c"] = "--center";
    args_map["-f"] = "--filter";
    args_map["-rt"] = "--rotate";  // rotate the cloud around the x-axis by 90 degrees clockwise
    args_map["-d"] = "--density"; // compute density of the cloud
    args_map["-t2ply"] = "--transfer2ply"; // transfer pcd file to ply file
    args_map["-t2pcd"] = "--transfer2pcd"; // transfer ply file to pcd file
    args_map["-t"] = "--test"; // test the executable



    int num_ray_sample = 1000; // default value, ray downsampling cloud
    bool filter_cloud = false;
    double epsilon = 0.1;
    size_t num_points = 0;
    size_t num_scanned_points = 0;

    std::string file_name = "";
    std::string input_path = "";
    std::string segmentation_path = "";
    std::string polygon_path = "";
    std::string recon_path = "";
    std::string recon_gt_path = "";
    std::string gt_path = "";

    Property prop;
    Reconstruction recon;
    Helper helper;
    Scanner scanner;

    std::string folder_path = "/mnt/c/Users/51932/Desktop/s3d/Area_1/conferenceRoom_1/Annotations/"; // default value, batch reconstruction
    // recon.batchReconstructionFromTxt(folder_path);

    Evaluation eval;
    if (arg1.substr(0, 3) == "-i=") {

        file_name = arg1.substr(3, arg1.length());

    } else if (arg1.substr(0, 9) == "--input==") {

        file_name = arg1.substr(9, arg1.length());

    } else if (arg1.substr(0, 4) == "-rc=" || arg1.substr(0, 15) == "--reconstruct==") {

        if (arg1.substr(0, 4) == "-rc=") {
            recon_path = "../files/" + arg1.substr(4, arg1.length());
        } else {
            recon_path = "../files/" + arg1.substr(14, arg1.length());
        }

        recon.pointCloudReconstructionFromTxt(recon_path);
        std::cout << "recon_path: " << recon_path << std::endl;

        return 0;

    } else if (arg1.substr(0, 5) == "-rcg=") {
            recon_gt_path = "../files/" + arg1.substr(5, arg1.length());
            std::cout << "recon_gt_path: " << recon_gt_path << std::endl;
            recon.batchReconstructionFromTxt(recon_gt_path);
            return 0;
    } else if (arg1 == "-h" || arg1 == "--help") {

        for (auto it = args_map.begin(); it != args_map.end(); it++) {
            std::cout << it->first << " " << it->second << std::endl;
        }

        return 0;
    } else if (arg1 == "-t") {

        std::cout << "Executable running successfully in test" << std::endl;
        return 0;

    } else {

        std::cout << "You have to provide at least two arguments" << std::endl;
        return 0;
    }

    input_path = "../files/" + file_name;
    std::cout << "input_path: " << input_path << std::endl;
   
    // load cloud from file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_path, *cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return (-1);
    }

    double density = 1.0; 

    num_points = cloud->width * cloud->height;
    std::cout << "Loaded " << num_points << " data points from " << file_name << std::endl;

    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cloud, minPt, maxPt);

    pcl::PointXYZ center;
    center.x = (minPt.x + maxPt.x) / 2;
    center.y = (minPt.y + maxPt.y) / 2;
    center.z = (minPt.z + maxPt.z) / 2;

    std::cout << "center: " << center.x << " " << center.y << " " << center.z << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);          
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(input_path, *colored_cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return (-1);
    }

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr centered_colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // centered_colored_cloud = helper.centerColoredCloud(colored_cloud, minPt, maxPt, file_name);

    // parse arguments related to functionality
    for (int i = 2; i < argc; i++) {

        std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
        std::string argi = argv[i];

        //  use ray to downsample the cloud
        if (argi.substr(0, 4) == "-rs=" || argi.substr(0, 13) == "--raysample==") {

            if (argi.substr(0, 4) == "-rs=") {

                num_ray_sample = std::stoi(argi.substr(4, argi.length()));

            } else if (argi.substr(0, 13) == "--raysample==") {

                num_ray_sample = std::stoi(argi.substr(13, argi.length()));

            }
            pcl::PointCloud<pcl::PointXYZI>::Ptr gt_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::io::loadPCDFile<pcl::PointXYZI>(gt_path, *gt_cloud);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr sacnned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            int pattern_center = 2;
            std::vector<pcl::PointXYZ> center_scanning = scanner.scanning_positions(0, minPt, maxPt, pattern_center);
            sacnned_cloud = scanner.multi_sphere_scanner(0.05, 0.05, 0.1, num_ray_sample, minPt, maxPt, pattern_center, center_scanning, cloud, gt_cloud, colored_cloud, file_name);
            num_scanned_points = sacnned_cloud->width * sacnned_cloud->height;
            double sample_rate = (double) num_scanned_points / (double) num_points;
            std::cout << "center scanning sample rate: " << sample_rate << std::endl;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr sacnned_cloud_v1(new pcl::PointCloud<pcl::PointXYZRGB>);
            int pattern_v1 = 4; // max scanning
            std::vector<pcl::PointXYZ> v1_scanning = scanner.scanning_positions(0, minPt, maxPt, pattern_v1);
            sacnned_cloud_v1 = scanner.multi_sphere_scanner(0.05, 0.05, 0.1, num_ray_sample, minPt, maxPt, pattern_v1, v1_scanning, cloud, gt_cloud, colored_cloud, file_name);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr sacnned_cloud_v2(new pcl::PointCloud<pcl::PointXYZRGB>);
            int pattern_v2 = 5; // min scanning
            std::vector<pcl::PointXYZ> v2_scanning = scanner.scanning_positions(0, minPt, maxPt, pattern_v2);
            sacnned_cloud_v2 = scanner.multi_sphere_scanner(0.05, 0.05, 0.1, num_ray_sample, minPt, maxPt, pattern_v2, v2_scanning, cloud, gt_cloud, colored_cloud, file_name);

        // compute occlusionlevel
        } else if (argi.substr(0, 4) == "-rsc=") {
            
            size_t num_random_positions = std::stoi(argi.substr(4, argi.length()));
            scanner.random_scanner(0.05, 0.1, num_random_positions, minPt, maxPt, cloud, colored_cloud, file_name);

        } else if (argi == "-o" || argi == "-om") {
            
            std::vector<std::vector<pcl::PointXYZ>> polygons = helper.parsePolygonData(polygon_path);
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> polygonClouds;
            std::vector<pcl::ModelCoefficients::Ptr> allCoefficients;

            for (int i = 0; i < polygons.size(); i++) {
                pcl::ModelCoefficients::Ptr coefficients = helper.computePlaneCoefficients(polygons[i]);
                allCoefficients.push_back(coefficients);

                pcl::PointCloud<pcl::PointXYZ>::Ptr polygon = helper.estimatePolygon(polygons[i], coefficients);
                polygonClouds.push_back(polygon);
            }

            double search_radius = 0.1;
            int pattern = 2;
            if (argi == "-o") {

                double rayOcclusionLevel = helper.rayBasedOcclusionLevel(minPt, maxPt, search_radius, pattern, 
                                                                        cloud, polygonClouds, allCoefficients);
    
            } else if (argi == "-om") {

                pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_density(new pcl::PointCloud<pcl::PointXYZI>);
                cloud_with_density = prop.computeDensityGaussian(cloud);

                pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_median_distance(new pcl::PointCloud<pcl::PointXYZI>);
                cloud_with_median_distance = helper.computeMedianDistance(search_radius, cloud, cloud_with_density);

                double rayOcclusionLevel = helper.rayBasedOcclusionLevelMedian(minPt, maxPt, density, search_radius, pattern, 
                                                                            cloud, cloud_with_median_distance,
                                                                            polygonClouds, allCoefficients);
            }

        } else if (argi == "-d" || argi == "--density") {

            prop.computeDensityGaussian(cloud);

        } else if (argi == "-t2ply") {

            recon.pcd2ply(colored_cloud, file_name);
            
        } else if (argi == "-f" || argi == "--filter") {

            helper.voxelizePointCloud<pcl::PointXYZRGB>(colored_cloud, file_name);

        } else if (argi.substr(0, 3) == "-p=") {

            polygon_path = "../files/" + argi.substr(3, argi.length());
            std::cout << "polygon_path: " << polygon_path << std::endl;

        } else if (argi.substr(0, 3) == "-s=") {

            segmentation_path = "../files/" + argi.substr(3, argi.length());
            std::cout << "segmentation_path: " << segmentation_path << std::endl;

        } else if (argi.substr(0, 4) == "-gt=") {

            gt_path = "../files/" + argi.substr(4, argi.length());
            std::cout << "gt_path: " << gt_path << std::endl;

        } else if (argi == "-e") {

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::io::loadPCDFile<pcl::PointXYZRGB>(segmentation_path, *segmented_cloud);

            pcl::PointCloud<pcl::PointXYZI>::Ptr ground_truth_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::io::loadPCDFile<pcl::PointXYZI>(gt_path, *ground_truth_cloud);

            eval.compareClouds(segmented_cloud, ground_truth_cloud);
            eval.calculateIoU();
            eval.calculateAccuracy();
            eval.calculatePrecision();
            eval.calculateRecall();
            eval.calculateF1Score();
        }

    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << " Time taken by this run: " << duration.count() << " seconds" << std::endl;

    return 0;
}
