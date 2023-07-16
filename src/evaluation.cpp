
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


std::vector<int> Evaluation::compareClouds(pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmented_cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr ground_truth_cloud) {

    std::map<std::string, std::vector<int>> color_label_map;
    color_label_map["wall"] = {174, 199, 232, 0};
    color_label_map["floor"] = {0, 255, 0, 1};
    color_label_map["cabinet"] = {0, 0, 255, 2};
    color_label_map["bed"] = {255, 255, 0, 3};
    color_label_map["chair"] = {255, 0, 255, 4};
    color_label_map["sofa"] = {0, 255, 255, 5};
    color_label_map["table"] = {255, 255, 255, 6};
    color_label_map["door"] = {192, 192, 192, 7};
    color_label_map["window"] = {128, 128, 128, 8};
    color_label_map["bookshelf"] = {128, 0, 0, 9};
    color_label_map["picture"] = {128, 128, 0, 10};
    color_label_map["counter"] = {0, 128, 0, 11};
    color_label_map["desk"] = {128, 0, 128, 12};
    color_label_map["curtain"] = {0, 128, 128, 13};
    color_label_map["refrigerator"] = {255, 0, 0, 14};
    color_label_map["shower curtain"] = {255, 255, 0, 15};
    color_label_map["toilet"] = {0, 255, 0, 16};
    color_label_map["sink"] = {0, 0, 255, 17};
    color_label_map["bathtub"] = {255, 0, 255, 18};
    color_label_map["otherfurniture"] = {0, 255, 255, 19};

    std::cout << "Comparing clouds... " << std::endl;
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(ground_truth_cloud);

    std::vector<int> ground_truth_labels;

    for (size_t i = 0; i < segmented_cloud->size(); ++i) {

        pcl::PointXYZRGB segmented_point = segmented_cloud->points[i];
        std::vector<int> segmented_point_color = {segmented_point.r, segmented_point.g, segmented_point.b};
        int segmented_point_label = -1;

        for (auto it = color_label_map.begin(); it != color_label_map.end(); ++it) {
            if (it->second[0] == segmented_point_color[0] && it->second[1] == segmented_point_color[1] && it->second[2] == segmented_point_color[2]) {
                segmented_point_label = it->second[3];
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

            if (ground_truth_point.intensity == segmented_point_label) {
                ground_truth_labels.push_back(1);
            } else {
                ground_truth_labels.push_back(0);
            }
        }
    }

    return ground_truth_labels;
}

void Evaluation::updateProperties(size_t& tp, size_t& fp, size_t& fn, size_t& tn) {

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
}

float Evaluation::calculateIoU() {
    size_t tp, fp, fn, tn;
    updateProperties(tp, fp, fn, tn);
    return static_cast<float>(tp) / (tp + fp + fn);
}

float Evaluation::calculateAccuracy() {
    size_t tp, fp, fn, tn;
    updateProperties(tp, fp, fn, tn);
    return static_cast<float>(tp + tn) / (tp + fp + fn + tn);
}

float Evaluation::calculatePrecision() {
    size_t tp, fp, fn, tn;
    updateProperties(tp, fp, fn, tn);
    return static_cast<float>(tp) / (tp + fp);
}

float Evaluation::calculateRecall() {
    size_t tp, fp, fn, tn;
    updateProperties(tp, fp, fn, tn);
    return static_cast<float>(tp) / (tp + fn);
}

float Evaluation::calculateF1Score() {
    size_t tp, fp, fn, tn;
    updateProperties(tp, fp, fn, tn);
    return 2 * static_cast<float>(tp) / (2 * tp + fp + fn);
}
