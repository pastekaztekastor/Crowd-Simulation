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
    std::cout << "Crowd Simulation" << std::endl;

    // Create a Map object
    // Note: The usage of 'new' is not recommended as it can cause memory leaks.
    // Instead, consider using smart pointers or stack-based objects.
    Map map = *new Map();

    // Create a Population object named "Test1"
    Population population = *new Population("Test1");

    // Initialize the Population with random positions for 10 individuals
    // The '1' here might be a parameter specifying the seed for randomness.
    // 'map.getDimensions()' returns the dimensions of the map, and 'map.getMap()' returns the map data.
    population.initRandom(10, 1, map.getDimensions(), map.getMap());

    // Add the created Population to the Map
    map.addPopulation(population);

    // Output "Map.Print" to indicate that the map will be printed
    std::cout << "Map.Print" << std::endl;

    // Print the Map and its contents
    map.print();

    // Output "Map.Print" again
    std::cout << "Map.Print" << std::endl;

    // Print the states of the individuals in the first Population
    map.getPopulations()[0].printStates();

    // Print the exit positions of the individuals in the first Population
    map.getPopulations()[0].printExits();

    // Print the cost of each position on the map for the first Population
    // The 'map.getDimensions()' is likely used to specify the dimensions of the map for printing costs.
    map.getPopulations()[0].printMapCost(map.getDimensions());

    // Output "Map.Print" again
    std::cout << "Map.Print" << std::endl;

    // Print the Map and its contents again
    map.print();

    // End the main function and return 0, indicating successful execution
    return 0;
}
