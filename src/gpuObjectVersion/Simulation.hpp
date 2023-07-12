#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "utils/utils.hpp"
#include "Map.hpp"
#include "Export.hpp"
#include "Kernel.hpp"

class Simulation
{
private:
    Map map;
    Export expt; 

public:
    Simulation();

    ~Simulation();
};
#endif //SIMULATION_HPP