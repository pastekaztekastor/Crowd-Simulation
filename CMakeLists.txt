cmake_minimum_required(VERSION 3.0)
project(GPUObjSimulation VERSION 0.1.0 LANGUAGES CXX)

# Définir le standard C++11
set(CMAKE_CXX_STANDARD 11)

# Spécifier les fichiers source
set(SOURCES
    src/gpuObjectVersion/main.cpp

    src/gpuObjectVersion/utils/utils.cpp

    # src/gpuObjectVersion/Kernel.cu
    src/gpuObjectVersion/Map.cpp
    src/gpuObjectVersion/Population.cpp
    src/gpuObjectVersion/Settings.cpp
    src/gpuObjectVersion/Simulation.cpp
)

# Spécifier les fichiers d'en-tête
set(HEADERS
    src/gpuObjectVersion/utils/utils.hpp

    # src/gpuObjectVersion/Kernel.hpp
    src/gpuObjectVersion/Map.hpp
    src/gpuObjectVersion/Population.hpp
    src/gpuObjectVersion/Settings.hpp
    src/gpuObjectVersion/Simulation.hpp
)

# Recherchez le package OpenCV
# set(OpenCV_DIR /usr/local)
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# Spécifier le répertoire de sortie pour l'exécutable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/exe)

# Ajouter les options de compilation pour CUDA
#find_package(CUDA REQUIRED)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_30)

# Activer le langage C
enable_language(C)

# Inclure les bibliothèques HDF5
# find_package(HDF5 REQUIRED COMPONENTS C HL)
# include_directories(${HDF5_INCLUDE_DIRS})
# link_directories(${HDF5_LIBRARY_DIRS})

# Compiler les fichiers source avec g++
add_executable(gpuObjSimulation ${SOURCES} ${HEADERS})

# Lier les librairies CUDA et HDF5
#target_link_libraries(gpuSimulation PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(gpuObjSimulation PRIVATE ${HDF5_LIBRARIES})
#target_link_libraries(gpuObjSimulation PRIVATE opencv_highgui opencv_imgproc)
target_link_libraries(gpuObjSimulation PRIVATE ${OpenCV_LIBS})