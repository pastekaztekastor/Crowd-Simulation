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

    // Create a Map object
    // Note: The usage of 'new' is not recommended as it can cause memory leaks.
    // Instead, consider using smart pointers or stack-based objects.
    Map map = *new Map("../../input Image/2/map.png");

    // Create a Population object named "Test1"
    Population population = *new Population(__PATH_DENSITY__,__PATH_EXIT__);

    // Add the created Population to the Map and calculat Map Cost
    map.addPopulation(population);
    map.initCostMap();

    // Output "Map.Print" to indicate that the map will be printed
    std::cout << "Map.Print 1" << std::endl;

    // Print the Map and its contents
    map.print();

    // initialisaiton du kernel
    Kernel kernel(map);
    Export extp(map);

    int testFin = 1;
    int secu = 0;

    while (testFin > 0 || secu < 10){
        testFin = kernel.computeNextFrame();
        if(__PRINT_DEBUG__)std::cout << testFin << std::endl;
        extp.creatFrame(kernel);
        secu ++;
    }

    extp.compileFramesToVid(map);

    // End the main function and return 0, indicating successful execution
    return 0;
}
