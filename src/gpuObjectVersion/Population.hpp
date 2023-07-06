#include "utils/utils.hpp"

class Population

{
private:
    /* data */
    std::vector<int3> etats;
    int nbPopulations;

public:
    Population(int nbPopulation, uint2 simulationDim, std::vector<int> mapElements );

    int2 getPosOf(int index) const;
    int getWaitOf(int index) const;
    int getNbPopulations() const;

    void setNbPopulations(int newNbPopulations); 

    ~Population();
};