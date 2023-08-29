/*******************************************************************************
* File Name: main.cpp
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: main file of the crowd simulation with parallelization on GPU. Contains only the main program.
*******************************************************************************/


// Include necessary libraries here

#include "utils/utils.hpp"

#include "Kernel.hpp"
#include "Map.hpp"
#include "Population.hpp"
#include "Settings.hpp"
#include "Simulation.hpp"
#include "Export.hpp"



#include <iostream>

// Main function
int main(int argc, char const *argv[])
{
    srand(time(NULL));

    // Output the title for the simulation
    std::cout << "---------- Crowd Simulation ----------" << std::endl;

    clock_t start = clock();

    // Create a Map object
    // Note: The usage of 'new' is not recommended as it can cause memory leaks.
    // Instead, consider using smart pointers or stack-based objects.
    std::string pathMap = std::string("../../input Image/") +
                std::string(__NAME_DIR_INPUT_FILES__) +
                std::string("/map.png");
    Map map = *new Map(pathMap);

    // Create a Population object named "Test1"
    for (int i = 0; i < __NB_POPULATION__; i++) {
        std::string pathPopulation = std::string("../../input Image/") +
                std::string(__NAME_DIR_INPUT_FILES__) +
                std::string("/population") +
                std::to_string(i) +
                std::string(".png");
        std::string pathExit = std::string("../../input Image/") +
                std::string(__NAME_DIR_INPUT_FILES__) +
                std::string("/exit") +
                std::to_string(i) +
                std::string(".png");
        Population population = *new Population(pathPopulation,pathExit);
        map.addPopulation(population);
    }
    //calculat Map Cost
    map.initCostMap();

    // Output "Map.Print" to indicate that the map will be printed
    if(__PRINT_DEBUG__)std::cout << "Map.Print 1" << std::endl;

    // Print the Map and its contents
    if(__PRINT_DEBUG__)map.print();

    // initialisaiton du kernel
    Kernel kernel(map);
    Export extp(map);

    int minExit = INT_MAX;
    for(auto &i : map.getPopulations()){
        minExit = std::min(minExit, (int)i.getExits().size());
    }
    long int secu = std::sqrt(std::pow(map.getDimensions().x,2) * std::pow(map.getDimensions().y,2))*kernel.getPopulation().size();
    if (__NB_FRAMES_MAX__ > 0) {
        secu = (secu < __NB_FRAMES_MAX__) ? secu : __NB_FRAMES_MAX__;
    }

    std::cout << " - Nb mure : " << map.getWallPositions().size() << std::endl
            << " - Nb population : " << map.getPopulations().size() << std::endl
            << " - Nb individu total : " << kernel.getPopulation().size() << std::endl
            << " - Frames max : " << secu << std::endl;
              //<< " - Nb sorti total" << kernel. << std::endl;

    if (minExit == 0) {
        std::cout << "#ERREUR : l'une des populations ne possède aucune sortie"<< std::endl;
        return 0;
    }

    int testFin = 1;
    int frame = 0;
    extp.creatFrame(kernel); // enregistrer la frame 0

    clock_t end = clock();
    double init_time = (double)(end - start) / CLOCKS_PER_SEC;

    std::cout << "DÉBUT DE LA SIMULATION "<< std::endl;

    start = clock();
    while (testFin > 0 && frame < secu){
        frame ++;
        testFin = kernel.computeNextFrame();
        extp.creatFrame(kernel);
        progressBar((int)kernel.getPopulation().size()-testFin, (int)kernel.getPopulation().size(), 200, frame);
    }
    end = clock();
    double simul_time = (double)(end - start) / CLOCKS_PER_SEC;

    std::cout << std::endl;
    std::cout << "CRÉATION DE LA VIDÉO "<< std::endl;
    start = clock();
    extp.compileFramesToVid(map);
    std::cout << std::endl;
    end = clock();
    double expt_time = (double)(end - start) / CLOCKS_PER_SEC;

    std::cout << "RÉSUMÉ " << std::endl
              << " - calculé en " << frame << " frames " << std::endl;
    if(init_time > 60) {
        init_time /= 60;
        std::cout << " - temps pour l'initialisation " << init_time << " min " << std::endl;
    } else {
        std::cout << " - temps pour l'initialisation " << init_time << " sec " << std::endl;
    }
    if(simul_time > 60) {
        simul_time /= 60;
        std::cout << " - temps pour la simulation " << simul_time << " min " << std::endl;
    } else {
        std::cout << " - temps pour la simulation " << simul_time << " sec " << std::endl;
    }
    if(expt_time > 60) {
        expt_time /= 60;
        std::cout << " - temps pour l'export vidéo " << expt_time << " min " << std::endl;
    } else {
        std::cout << " - temps pour l'export vidéo " << expt_time << " sec " << std::endl;
    }

    // End the main function and return 0, indicating successful execution
    return 0;
}
