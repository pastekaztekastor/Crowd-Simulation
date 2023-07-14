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
    std::vector<float> pDeplacement;
    color col;

public:
    Population();
    Population(std::string name);
    Population(std::string name, int nbPopulations, int nbExit, uint2 simulationDim, std::vector<int> mapElements);

    void initRandomEtats(int nbPopulations, uint2 simulationDim, std::vector<int> mapElements);
    void initRandomExits(int nbExit, uint2 simulationDim, std::vector<int> mapElements);
    void initRandom(int nbPopulations, int nbExit, uint2 simulationDim, std::vector<int> mapElements);

    const std::string &getName() const;
    void setName(const std::string &name);
    const std::vector<int3> &getEtats() const;
    void setEtats(const std::vector<int3> &etats);
    const std::vector<int2> &getExits() const;
    void setExits(const std::vector<int2> &exits);
    const std::vector<unsigned int> &getMapCost() const;
    void setMapCost(const std::vector<unsigned int> &mapCost);
    const color &getCol() const;
    void setCol(const color &col);
    void setCol(uint r, uint g, uint b);

    void printEtats() const;
    void printExits() const;
    void printMapCost(uint2 dimension) const;
    void print(uint2 dimension);

    ~Population();
};

#endif // POPULATION_HPP