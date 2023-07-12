/*******************************************************************************
// Fichier : Population.cpp
// Description : Fichier de définition de la classe Population qui représente une population dans la simulation de foule.
// Auteur : Mathurin Champémont
// Date : 06/07/2023
*******************************************************************************/

#include "Population.hpp"

Population::Population(std::string name, int nbPopulations, int nbExit, uint2 simulationDim, std::vector<int> mapElements)
{
    this->name = name;
    this->col = {(uint)(rand()%255), (uint)(rand()%255), (uint)(rand()%255)};
    Population::initRandom(nbPopulations, nbExit, simulationDim, mapElements);
}
void Population::initRandomEtats(int nbPopulations, uint2 simulationDim, std::vector<int> mapElements){
    for (size_t i = 0; i < nbPopulations; i++)
    {
        int3 coord;
        bool isPicked = false;
        do 
        {
            coord = {(int)(rand()%simulationDim.x), (int)(rand()%simulationDim.y), 0};
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
void Population::initRandomExits(int nbExit, uint2 simulationDim, std::vector<int> mapElements){
    for (size_t i = 0; i < nbExit; i++)
    {
        int2 coord;
        bool isPicked = false;
        do 
        {
            coord = {(int)(rand()%simulationDim.x), (int)(rand()%simulationDim.y)};
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

void Population::initRandom(int nbPopulations, int nbExit, uint2 simulationDim, std::vector<int> mapElements){
    Population::initRandomExits(nbExit, simulationDim, mapElements);
    Population::initRandomEtats(nbPopulations,simulationDim, mapElements);
}
void Population::setName(std::string name){
    this->name = name;
}
void Population::setColor(uint r, uint g, uint b){
    this->col = {r,g,b};
}

// GETTER
int2 Population::getPosOf(int index) const
{
    return {etats[index].x, etats[index].y};
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
    std::cout << " Liste de position des individus de de la population : " << this->name << std::endl;
    for (auto && coord : this->etats)
    {
        std::cout << "x : " << coord.x << "y : " << coord.y << "z : " << coord.z << std::endl;
    }
}
void Population::printExits(){
    std::cout << " Liste de position des sorties de la population : " << this->name << std::endl;
    for (auto && coord : this->exits)
    {
        std::cout << "x : " << coord.x << "y : " << coord.y << "z : " << std::endl;
    }
}
void Population::printMapCost(uint2 dimension){
    std::cout << " Carte de cout de la population : " << this->name << std::endl;
    std::cout <<"  ";
        for (int x = 0; x < dimension.x; x++)
        {
            printf("%2d  ",x); 
        }
        std::cout<<"  "<<std::endl;

    for (int y = 0; y < dimension.y; y++)
    {
        printf("%2d ",y); 
        for (int x = 0; x < dimension.x; x++)
        {
            printf(" %2d ", this->mapCost[y * dimension.x + x]);
        }
        std::cout << std::endl;
    }
}
void Population::print(uint2 dimension){
    std::cout << "POPULATION : " << this->name << std::endl;

    Population::printEtats();
    Population::printExits();
    Population::printMapCost(dimension);

    std::cout << std::endl;
}

Population::~Population()
{
}



