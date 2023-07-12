#include <stdio.h>
#include <stdlib.h>
#include <vector>

// Define
#define uint usigned int

// Map
#define __MAP_EMPTY__                   -1
#define __MAP_EXIT__                    -2
#define __MAP_WALL__                    -3
#define __MAP_HUMAN_QUITE__             make_float3(-1.f,-1.f,0.f)

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

// Struct 
typedef struct { int x ; int y ; }int2;
typedef struct { uint x; uint y; }uint2;
typedef struct { int x ; int y ; int z }int3;
typedef struct { uint x; uint y; uint z}uint3;