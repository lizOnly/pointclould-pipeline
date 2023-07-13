
#include "../headers/evaluation.h"
#include <iostream>


Evaluation::Evaluation() {
    // empty constructor
}

Evaluation::~Evaluation() {
    // empty destructor
}


/*
    * Find the index of the first underscore in the string
    * @param str: the string to be searched
    * @return: the index of the first underscore in the string
    *          if no underscore is found, return -1
*/
int Evaluation::findUnderScore(std::string& str) {
    for (int i = 0; i < str.size(); ++i) {
        if (str[i] == '_') {
            return i;
        }
    }
    return -1;
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
