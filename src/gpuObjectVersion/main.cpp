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
    Map map = *new Map(__PATH_MAP__);

    // Create a Population object named "Test1"
    Population population = *new Population(__PATH_DENSITY__,__PATH_EXIT__);
    //Population population2 = *new Population(__PATH_DENSITY2__,__PATH_EXIT2__);

    // Add the created Population to the Map and calculat Map Cost
    map.addPopulation(population);
    //map.addPopulation(population2);
    //map.initCostMap();

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

    std::cout << " - Nb mure : " << map.getWallPositions().size() << std::endl
              << " - Nb population : " << map.getPopulations().size() << std::endl
              << " - Nb individu total : " << kernel.getPopulation().size() << std::endl;
              //<< " - Nb sorti total" << kernel. << std::endl;

    if (minExit == 0) {
        std::cout << "#ERREUR : l'une des population ne possède aucune sortie"<< std::endl;
        return 0;
    }

    int testFin = 1;
    int secu = 0;
    extp.creatFrame(kernel); // enregistrer la frame 0

    clock_t end = clock();
    double init_time = (double)(end - start) / CLOCKS_PER_SEC;

    std::cout << "DÉBUT DE LA SIMULATION "<< std::endl;

    start = clock();
    while (testFin > 0 && secu < 10000){
        secu ++;
        testFin = kernel.computeNextFrame();
        extp.creatFrame(kernel);
        progressBar((int)kernel.getPopulation().size()-testFin,(int)kernel.getPopulation().size(),200, secu);
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

    std::cout   << "RÉSUMÉ "<<std::endl
            << " - calculé en " << secu << " frames " << std::endl
            << " - temps pour l'initialisation " << init_time << " s " << std::endl
            << " - temps pour la simualtion " << simul_time << " s " << std::endl
            << " - temps pour l'export " << expt_time << " s " << std::endl;

    // End the main function and return 0, indicating successful execution
    return 0;
}
