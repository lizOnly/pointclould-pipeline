
#include <iostream>
#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "../headers/evaluation.h"

Evaluation::Evaluation() {
    // empty constructor
}

Evaluation::~Evaluation() {
    // empty destructor
}


void Evaluation::compareClouds(pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmented_cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr ground_truth_cloud, bool compare_bound) {

    std::vector<std::vector<bool>> labels;

    size_t segmented_cloud_size = segmented_cloud->size();
    size_t ground_truth_cloud_size = ground_truth_cloud->size();

    std::cout << "Segmented cloud size: " << segmented_cloud_size << std::endl;
    std::cout << "Ground truth cloud size: " << ground_truth_cloud_size << std::endl;
    std::cout << std::endl;

    if (segmented_cloud_size != ground_truth_cloud_size) {
        std::cout << "Segmented cloud and ground truth cloud sizes are not equal!" << std::endl;
        return;
    }

    std::cout << "Comparing clouds... " << std::endl;
    std::cout << std::endl;
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(ground_truth_cloud);
    
    if (compare_bound) {
        std::cout << "Comparing only boundary points... " << std::endl;
    } else {
        std::cout << "Comparing all points... " << std::endl;
    }

    std::cout << std::endl;

    for (auto it = ground_truth_map.begin(); it != ground_truth_map.end(); ++it) {

        int ground_truth = it->second[0];

        if (compare_bound) {
            if (ground_truth == 4 || ground_truth == 5 || ground_truth == 6 || ground_truth == 9 || ground_truth == 25) {
                continue;
            }
        }
    
        for (size_t i = 0; i < segmented_cloud->size(); ++i) {

            pcl::PointXYZRGB segmented_point = segmented_cloud->points[i];
            std::vector<int> segmented_point_color = {segmented_point.r, segmented_point.g, segmented_point.b};
            int segmented_point_label = -1;

            for (auto it = color_label_map.begin(); it != color_label_map.end(); ++it) {

                if (it->second[0] == segmented_point_color[0] && it->second[1] == segmented_point_color[1] && it->second[2] == segmented_point_color[2]) {
                    
                    if (compare_bound) {
                        if (it->second[3] == 4 || it->second[3] == 5 || it->second[3] == 6 || it->second[3] == 9 || it->second[3] == 25) {
                            continue;
                        } else {
                            segmented_point_label = it->second[3];
                        }
                    } else {
                        segmented_point_label = it->second[3];
                    }

                }

            }

            pcl::PointXYZI searchPoint;
            searchPoint.x = segmented_point.x;
            searchPoint.y = segmented_point.y;
            searchPoint.z = segmented_point.z;
            searchPoint.intensity = segmented_point_label;

            if (segmented_point_label != -1) {

                std::vector<int> indices;
                std::vector<float> distances;
                kdtree.nearestKSearch(searchPoint, 1, indices, distances);
                
                pcl::PointXYZI ground_truth_point = ground_truth_cloud->points[indices[0]];

                if (ground_truth_point.intensity == ground_truth) {

                    ground_truth_labels.push_back(true);

                    if (segmented_point_label == ground_truth) {
                        predicted_labels.push_back(true);
                    } else {
                        predicted_labels.push_back(false);
                    }
                    
                } else {

                    ground_truth_labels.push_back(false);

                    if (segmented_point_label == ground_truth) {
                        predicted_labels.push_back(true);
                    } else {
                        predicted_labels.push_back(false);
                    }

                }
            }
        }
    }

    labels.push_back(ground_truth_labels);
    labels.push_back(predicted_labels);

    std::cout << "Updating properties... " << std::endl;
    updateProperties();
    std::cout << std::endl;
}

void Evaluation::updateProperties() {

    tp = fp = fn = tn = 0;

    for (size_t i = 0; i < ground_truth_labels.size(); ++i) {
        if (ground_truth_labels[i] && predicted_labels[i]) {
            ++tp;
        } else if (!ground_truth_labels[i] && predicted_labels[i]) {
            ++fp;
        } else if (ground_truth_labels[i] && !predicted_labels[i]) {
            ++fn;
        } else {
            ++tn;
        }
    }

    std::cout << "True positives: " << tp << std::endl;
    std::cout << "False positives: " << fp << std::endl;
    std::cout << "False negatives: " << fn << std::endl;
    std::cout << "True negatives: " << tn << std::endl;
    std::cout << std::endl;
}

float Evaluation::calculateIoU() {

    std::cout << "Calculating IoU... " << std::endl;
    double iou = static_cast<double>(tp) / (tp + fp + fn);
    std::cout << "IoU: " << iou << std::endl;
    std::cout << std::endl;

    return iou;
}

float Evaluation::calculateAccuracy() {

    std::cout << "Calculating accuracy... " << std::endl;
    double accuracy = static_cast<double>(tp + tn) / (tp + fp + fn + tn);
    std::cout << "Accuracy: " << accuracy << std::endl;
    std::cout << std::endl;

    return accuracy;
}

float Evaluation::calculatePrecision() {

    std::cout << "Calculating precision... " << std::endl;
    double precision = static_cast<double>(tp) / (tp + fp);
    std::cout << "Precision: " << precision << std::endl;
    std::cout << std::endl;

    return precision;
}

float Evaluation::calculateRecall() {

    std::cout << "Calculating recall... " << std::endl;
    double recall = static_cast<double>(tp) / (tp + fn);
    std::cout << "Recall: " << recall << std::endl;
    std::cout << std::endl;

    return recall;
}

float Evaluation::calculateF1Score() {

    std::cout << "Calculating F1 score... " << std::endl;
    double f1_score = 2 * static_cast<double>(tp) / (2 * tp + fp + fn);
    std::cout << "F1 score: " << f1_score << std::endl;
    std::cout << std::endl;

    return f1_score;
}
