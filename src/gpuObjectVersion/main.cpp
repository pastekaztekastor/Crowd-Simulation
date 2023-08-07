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
    // Output the title for the simulation
    std::cout << "---------- Crowd Simulation ----------" << std::endl;

    // Create a Map object
    // Note: The usage of 'new' is not recommended as it can cause memory leaks.
    // Instead, consider using smart pointers or stack-based objects.
    Map map = *new Map();

    // Create a Population object named "Test1"
    Population population = *new Population("Test1");

    // Initialize the Population with random positions for 10 individuals
    // The '1' here might be a parameter specifying the seed for randomness.
    // 'map.getDimensions()' returns the dimensions of the map, and 'map.getMap()' returns the map data.
    population.initRandom(__MAP_NOMINALE_POPULATION_SIZE__, 1, map.getDimensions(), map.getMap());

    // Add the created Population to the Map and calculat Map Cost
    map.addPopulation(population);
    map.initCostMap();

    // Output "Map.Print" to indicate that the map will be printed
    std::cout << "Map.Print 1" << std::endl;

    // Print the Map and its contents
    map.print();

    // End the main function and return 0, indicating successful execution
    return 0;
}
