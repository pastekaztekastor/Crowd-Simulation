cmake_minimum_required(VERSION 3.0)
project(GPUObjSimulation VERSION 0.1.0 LANGUAGES CXX)

# Définir le standard C++11
set(CMAKE_CXX_STANDARD 11)

# Définir la macro GRAPHICS_CARD_PRESENT si la carte graphique est présente
# Trouver CUDA
find_package(CUDA QUIET)

if (CUDA_FOUND)
    set(GRAPHICS_CARD_PRESENT 1)
    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_30)
else()
    set(GRAPHICS_CARD_PRESENT 0)
endif()
# Afficher le résultat
message(STATUS "CUDA found: ${CUDA_FOUND}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")

# Spécifier les fichiers source
if (CUDA_FOUND)
    set(SOURCES
            src/gpuObjectVersion/main.cpp

            src/gpuObjectVersion/utils/utils.cpp

            src/gpuObjectVersion/Kernel.cu
            src/gpuObjectVersion/Map.cpp
            src/gpuObjectVersion/Population.cpp
            src/gpuObjectVersion/Settings.cpp
            src/gpuObjectVersion/Simulation.cpp)
else()
    set(SOURCES
            src/gpuObjectVersion/main.cpp

            src/gpuObjectVersion/utils/utils.cpp

            src/gpuObjectVersion/Kernel.cpp
            src/gpuObjectVersion/Map.cpp
            src/gpuObjectVersion/Population.cpp
            src/gpuObjectVersion/Settings.cpp
            src/gpuObjectVersion/Simulation.cpp)
endif()


# Spécifier les fichiers d'en-tête
set(HEADERS
    src/gpuObjectVersion/utils/utils.hpp

    src/gpuObjectVersion/Kernel.hpp
    src/gpuObjectVersion/Map.hpp
    src/gpuObjectVersion/Population.hpp
    src/gpuObjectVersion/Settings.hpp
    src/gpuObjectVersion/Simulation.hpp)

# Spécifier le répertoire de sortie pour l'exécutable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/exe)

# Compiler les fichiers source avec g++
add_executable(GPUObjSimulation ${SOURCES} ${HEADERS})

if (CUDA_FOUND)
    target_link_libraries(GPUObjSimulation PRIVATE ${CUDA_LIBRARIES})
endif()

target_link_libraries(GPUObjSimulation PRIVATE ${OpenCV_LIBS})