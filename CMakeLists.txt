cmake_minimum_required(VERSION 3.0)
project(GPUSimulation VERSION 0.1.0 LANGUAGES CXX)

# Spécifier les fichiers source
set(SOURCES
    src/cpu/onlyCPU.cpp
    )
    
    # Spécifier les fichiers d'en-tête
set(HEADERS
)

# Spécifier le répertoire de sortie pour l'exécutable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/exe)

# Compiler les fichiers source avec g++
add_executable(cpuSimulation ${SOURCES} ${HEADERS})
