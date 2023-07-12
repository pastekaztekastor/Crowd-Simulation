#include "utils/utils.hpp"
#include "Map.hpp"
#include "Export.hpp"
#include "Kernel.hpp"

class Simulation

{
private:
    Population populations;
    Export export; 

public:
    Simulation();

    ~Simulation();
};