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


int main(int argc, char const *argv[])
{
    // TEST CLASS Population
    std::cout << "Simulaiton de foule" << std::endl;
    // TEST CLASS Map
    Map map = *new Map();
    Population population = *new Population("Test1");
    population.initRandom(10,1,map.getDimensions(), map.getMap());
    map.addPopulation(population);

    std::cout << "Map.Print" << std::endl;
    map.print();

    std::cout << "Map.Print" << std::endl;
    map.getPopulations()[0].printEtats();
    map.getPopulations()[0].printExits();
    map.getPopulations()[0].printMapCost(map.getDimensions());

    std::cout << "Map.Print" << std::endl;
    map.print();

    return 0;
}
