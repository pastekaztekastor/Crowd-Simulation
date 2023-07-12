/*******************************************************************************
// Fichier : Population.hpp
// Description : Fichier d'en-tête de la classe Population qui représente une population dans la simulation de foule.
// Auteur : Mathurin Champémont
// Date : 06/07/2023
*******************************************************************************/

#ifndef POPULATION_HPP
#define POPULATION_HPP

// Déclaration de la classe Population

#include "utils/utils.hpp"

class Population
{
private:
    std::string name;
    std::vector<int3> etats;
    std::vector<int2> exits;
    std::vector<uint> mapCost;
    color col;

public:
    Population(std::string name);
    Population(std::string name, int nbPopulations, int nbExit, uint2 simulationDim, std::vector<int> mapElements);
    
    void initRandomEtats(int nbPopulations, uint2 simulationDim, std::vector<int> mapElements);
    void initRandomExits(int nbExit, uint2 simulationDim, std::vector<int> mapElements);

    void initRandom(int nbPopulations, int nbExit, uint2 simulationDim, std::vector<int> mapElements);

    void setName(std::string name);
    void setColor(uint r, uint g, uint b);

    int2 getPosOf(int index) const;
    int  getWaitOf(int index) const;
    int  getNbPopulations() const;

    void printEtats();
    void printExits();
    void printMapCost(uint2 dimension);
    void print(uint2 dimension);

    ~Population();
};

#endif // POPULATION_HPP