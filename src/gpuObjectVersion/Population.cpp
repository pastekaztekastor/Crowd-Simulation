#include "Population.hpp"

Population::Population()
{
    
}
Population::Population(int newNbPopulations, uint2 simulationDim, vector<int> mapElements)
{
    nbPopulations = newNbPopulations;
    
    for (size_t i = 0; i < nbPopulations; i++)
    {
        int3 coord;
        bool isPicked = false;
        do 
        {
            coord = make_int3(rand()%simulationDim.x, rand()%simulationDim.y, 0);
            for (auto && etat : etats)
            {
                if (etat.x == coord.x && etat.y == coord.y)
                {
                    isPicked = true;
                    break;
                }
            }
            
        } while ( (mapElements[coord.y * simulationDim.x + coord.y] != __MAP_EMPTY__) || isPicked );

        etats.push_back(coord);
    }
}

// GETTER
int2 Population::getPosOf(int index) const
{
    return make_int2(etats[index].x, etats[index].y);
}
int Population::getWaitOf(int index) const
{
    return (etats[index].z);
}
int Population::getNbPopulations() const
{
    return nbPopulations;
}

// SETTER
void Population::setNbPopulations(int newNbPopulations)
{
    nbPopulations = newNbPopulations;
}


Population::~Population()
{
}



