#include "utils.hpp"

// Interpolates between two colors using Euclidean distance in RGB space.
// Returns the normalized Euclidean distance between the colors.
float colorInterpol(color a, color b) {
    return sqrt(pow((b.r - a.r), 2) + pow((b.g - a.g), 2) + pow((b.b - a.b), 2)) / 441.672943;
}

// Interpolates between two OpenCV Scalar colors using Euclidean distance in RGB space.
// Returns the normalized Euclidean distance between the colors.
float colorInterpol(cv::Scalar a, cv::Scalar b) {
    return sqrt(pow((b[0] - a[0]), 2) + pow((b[1] - a[1]), 2) + pow((b[2] - a[2]), 2)) / 441.672943;
}

void progressBar(uint progress, uint total, uint width, uint iteration) {
    // From ChatGPT
    float percentage = (float)progress / total;
    uint filledWidth = ( uint)(percentage * width);
    printf("[");
    for ( uint i = 0; i < width; i++) {
        if (i < filledWidth) {
            printf("=");
        } else {
            printf(" ");
        }
    }
    printf("] %.2f%% : %d Frames\r", percentage * 100, iteration);
    fflush(stdout);
}