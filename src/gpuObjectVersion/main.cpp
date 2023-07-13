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
    Map map;
    for (int i = 0; i < 10; ++i) {
        std::cout << i ;
    }
    map.print();

    return 0;
}
