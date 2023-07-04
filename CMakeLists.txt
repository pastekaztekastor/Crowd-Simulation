cmake_minimum_required(VERSION 3.10)
project(GPUSimulation VERSION 0.2.0 LANGUAGES CXX CUDA) 

# Spécifier la version de C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Inclure la bibliothèque nlohmann_json
set(CMAKE_PREFIX_PATH "/home/cartron3/StageXLIM/extern/nlohmann_json" CACHE PATH "Path to nlohmann_json installation")
find_package(nlohmann_json 3.2.0 REQUIRED)

# Activer le langage C
enable_language(C)

# Inclure les bibliothèques HDF5
find_package(HDF5 REQUIRED COMPONENTS C HL)
include_directories(${HDF5_INCLUDE_DIRS})
link_directories(${HDF5_LIBRARY_DIRS})


# Spécifier les fichiers source
set(SOURCES
    src/gpu/simulation.cpp
    src/gpu/kernel.cu
    src/gpu/main.cu
    )
    
    # Spécifier les fichiers d'en-tête
set(HEADERS
    src/gpu/kernel.hpp
    src/gpu/simulation.hpp
    src/gpu/utils/commonCUDA.hpp
)

# Recherchez le package OpenCV
set(OpenCV_DIR /usr/local)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Spécifier le répertoire de sortie pour l'exécutable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/exe)

# Ajouter les options de compilation pour CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_30)

# Compiler les fichiers source avec g++
add_executable(gpuSimulation ${SOURCES} ${HEADERS})

# Lier les librairies CUDA et HDF5
target_link_libraries(gpuSimulation PRIVATE ${CUDA_LIBRARIES} ${HDF5_LIBRARIES})
target_link_libraries(gpuSimulation PRIVATE opencv_core opencv_highgui opencv_imgproc)
target_link_libraries(gpuSimulation PRIVATE ${OpenCV_LIBS})
