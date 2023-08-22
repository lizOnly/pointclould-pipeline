
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


void Evaluation::compareClouds(pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmented_cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr ground_truth_cloud) {

    std::vector<std::vector<bool>> labels;

    std::cout << "Comparing clouds... " << std::endl;
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(ground_truth_cloud);
    
    for (auto it = ground_truth_map.begin(); it != ground_truth_map.end(); ++it) {

        int ground_truth = it->second[0];
    
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

}

void Evaluation::updateProperties(size_t& tp, size_t& fp, size_t& fn, size_t& tn) {

    std::cout << "Updating properties... " << std::endl;
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

    std::cout << "Calculating IoU... " << std::endl;
    size_t tp, fp, fn, tn;
    updateProperties(tp, fp, fn, tn);
    double iou = static_cast<double>(tp) / (tp + fp + fn);
    std::cout << "IoU: " << iou << std::endl;
    std::cout << std::endl;

    return iou;
}

float Evaluation::calculateAccuracy() {

    std::cout << "Calculating accuracy... " << std::endl;
    size_t tp, fp, fn, tn;
    updateProperties(tp, fp, fn, tn);
    double accuracy = static_cast<double>(tp + tn) / (tp + fp + fn + tn);
    std::cout << "Accuracy: " << accuracy << std::endl;
    std::cout << std::endl;

    return accuracy;
}

float Evaluation::calculatePrecision() {

    std::cout << "Calculating precision... " << std::endl;
    size_t tp, fp, fn, tn;
    updateProperties(tp, fp, fn, tn);
    double precision = static_cast<double>(tp) / (tp + fp);
    std::cout << "Precision: " << precision << std::endl;
    std::cout << std::endl;

    return precision;
}

float Evaluation::calculateRecall() {

    std::cout << "Calculating recall... " << std::endl;
    size_t tp, fp, fn, tn;
    updateProperties(tp, fp, fn, tn);
    double recall = static_cast<double>(tp) / (tp + fn);
    std::cout << "Recall: " << recall << std::endl;
    std::cout << std::endl;

    return recall;
}

float Evaluation::calculateF1Score() {

    std::cout << "Calculating F1 score... " << std::endl;
    size_t tp, fp, fn, tn;
    updateProperties(tp, fp, fn, tn);
    double f1_score = 2 * static_cast<double>(tp) / (2 * tp + fp + fn);
    std::cout << "F1 score: " << f1_score << std::endl;
    std::cout << std::endl;

    return f1_score;
}
