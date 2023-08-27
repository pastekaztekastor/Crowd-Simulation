#ifndef UTILS_HPP
#define UTILS_HPP

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <random> // Pour std::default_random_engine

#include <time.h>
#include <unistd.h>        // chdir
#include <sys/stat.h>      // mkdir

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

// #include "matrixMove.cpp"

// Define
#define uint unsigned int

// Define des valeur nominales
// #define __POPULATION_NOMINAL__

#define __POPULATION_NOMINAL_SIZE__     10
#define __POPULATION_NOMINAL_NB_EXIT__  1

#define __PRINT_DEBUG__                 true

// Map
#define __MAP_EMPTY__                    -1
#define __MAP_EXIT__                     -2
#define __MAP_WALL__                     -3
#define __MAP_HUMAN_QUITE__              {-1.f,-1.f,0.f}
#define __MAP_NOMINALE_X_DIM__           4
#define __MAP_NOMINALE_Y_DIM__           4
#define __MAP_NOMINALE_WALL__            2
#define __MAP_NOMINALE_POPULATION__      1
#define __MAP_NOMINALE_POPULATION_SIZE__ 5

// POPULATION


// Export 
#define __EXPORT_TYPE_VIDEO__           0    
#define __EXPORT_TYPE_VALUE__           1    
#define __EXPORT_TYPE_ALL__             2 

#define __VIDEO_FPS__                   40
#define __VIDEO_CALC_COST_PLOT_ON__     1
#define __VIDEO_CALC_COST_PLOT_OFF__    0
#define __VIDEO_RATIO_FRAME__           1

#define __SIM_MAX_WAITING__             100

#define __MAX_X_DIM_JPEG__              1920
#define __MAX_Y_DIM_JPEG__              1080

// Print 
#define __DEBUG_PRINT_NONE__            0 
#define __DEBUG_PRINT_STEP__            1 
#define __DEBUG_PRINT_ALL__             2 
#define __DEBUG_PRINT_DEBUG__           3 
#define __DEBUG_PRINT_KERNEL__          4

// Color
#define __COLOR_WHITE__                 cv::Scalar(255, 255, 255)
#define __COLOR_RANDOM__                cv::Scalar(rand()%255, rand()%255, rand()%255)
#define __COLOR_BLUE__                  cv::Scalar(255, 0, 0)
#define __COLOR_GREEN__                 cv::Scalar(0, 255, 0)
#define __COLOR_RED__                   cv::Scalar(0, 0, 255)
#define __COLOR_BLACK__                 cv::Scalar(0, 0, 0)
#define __COLOR_GREY__                  cv::Scalar(125, 125, 125)
#define __COLOR_ALPHA__                 cv::Scalar(0, 0, 0, 0)

// Path
#define __PATH_DENSITY__                "../../input Image/2/population.png"
#define __PATH_EXIT__                   "../../input Image/2/exit.png"

// Struct 
typedef struct { int x ; int y ; }int2;
typedef struct { uint x; uint y; }uint2;
typedef struct { int x ; int y ; int z ; }int3;
typedef struct { uint x; uint y; uint z; }uint3;
typedef struct { uint r; uint g; uint b; }color;

typedef struct { int3 position; int from; int id; }individu;

// Matrix
std::vector<float> getMatrixMove();

/**
 * Interpolates between two colors using Euclidean distance in RGB space.
 * Calculates the normalized Euclidean distance between the colors.
 *
 * @param a First color.
 * @param b Second color.
 * @return Normalized Euclidean distance between the colors.
 */
float colorInterpol(color a, color b);

/**
 * Interpolates between two OpenCV Scalar colors using Euclidean distance in RGB space.
 * Calculates the normalized Euclidean distance between the colors.
 *
 * @param a First OpenCV Scalar color.
 * @param b Second OpenCV Scalar color.
 * @return Normalized Euclidean distance between the colors.
 */
float colorInterpol(cv::Scalar a, cv::Scalar b);

#endif // UTILS_HPP
