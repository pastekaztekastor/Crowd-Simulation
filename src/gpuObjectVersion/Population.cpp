#include "Population.hpp"

Population::Population()
{

}
Population::Population(int newNbPopulations, int nbExit, uint2 simulationDim, std::vector<int> mapElements)
{
    Population::initRandom(newNbPopulations, nbExit, simulationDim, mapElements);
}
void Population::initRandomEtats(int nbPopulations, uint2 simulationDim, vector<int> mapElements){
    for (size_t i = 0; i < nbPopulations; i++)
    {
        int3 coord;
        bool isPicked = false;
        do 
        {
            coord = make_int3(rand()%simulationDim.x, rand()%simulationDim.y, 0);
            for (auto && etat : this->etats)
            {
                if (etat.x == coord.x && etat.y == coord.y)
                {
                    isPicked = true;
                    break;
                }
            }
            
        } while ( (mapElements[coord.y * simulationDim.x + coord.y] != __MAP_EMPTY__) || isPicked );

        this->etats.push_back(coord);
    }
}
void Population::initRandomExits(int nbExit, uint2 simulationDim, vector<int> mapElements){
    for (size_t i = 0; i < nbExit; i++)
    {
        int3 coord;
        bool isPicked = false;
        do 
        {
            coord = make_int3(rand()%simulationDim.x, rand()%simulationDim.y, 0);
            for (auto && exit : this->exits)
            {
                if (exit.x == coord.x && exit.y == coord.y)
                {
                    isPicked = true;
                    break;
                }
            }
            
        } while ( (mapElements[coord.y * simulationDim.x + coord.y] != __MAP_EMPTY__) || isPicked );

        this->exits.push_back(coord);
    }
}

void Population::initRandom(int nbPopulations, int nbExit, uint2 simulationDim, vector<int> mapElements){
    Population::initRandomExits(nbExit, simulationDim, mapElements);
    Population::initRandomEtats(nbPopulations,simulationDim, mapElements);
}
void Population::setName(std::string name) const{
    this.name = name;
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
    return etats.size();
}


void Population::printEtats(){
    std::cout " Liste des position de : " << this.name << std::endl;
    for (auto && etat : this->etats.size())
    {
        std::cout << "x : " << etat.x << "y : " << etat.y << "z : " << etat.z << std::endl;
    }
}
void Population::printExits(){
    // TO DO
}
void Population::printMapCost(){
    // TO DO
}
void Population::print(){
    // TO DO
}

Population::~Population()
{
}



