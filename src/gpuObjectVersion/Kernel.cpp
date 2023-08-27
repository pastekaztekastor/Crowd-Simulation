/*******************************************************************************
// Fichier : Kernel.cpp
// Description : This file enables parallel computation when possible, otherwise, it performs the equivalent on the CPU. It calculates the displacement of each individual on the displacement grid for each frame.
// Auteur : Mathurin Champémont
// Date : 02/08/2023
*******************************************************************************/

#include "Kernel.hpp"

#ifndef GRAPHICS_CARD_PRESENT

// CONSTRUCTOR CPU
Kernel::Kernel(const Map& map)
{
    Kernel::initKerel(map);
}

// FUNCTIONS CPU
void Kernel::initKerel(const Map& map) {
    this->isEmptyMapWall= map.getMapWall();
    this->isEmptyMapPopulation.resize(map.getMap().size(), true);

    for (int i = 0; i< map.getPopulations().size(); i++) {
        //Création du vecteur population
        for(auto &&individu : map.getPopulations()[i].getStates()){
            this->population.push_back({individu, i});
            this->isEmptyMapPopulation[individu.x + individu.y * map.getDimensions().x] = false;
        }
        // copy des cartes de couts
        this->costMaps_s.push_back(map.getPopulations()[i].getMapCost());
        // cpoy des déplacements
        this->pMovements_s.push_back(map.getPopulations()[i].getPMovement());
        this->directions_s.push_back(map.getPopulations()[i].getDirections());
    }

    for (int i = 0; i < this->population.size(); ++i) {
        this->shuffleIndex.push_back((uint)i);
    }
    this->dimentionMap = map.getDimensions();
}

// Computes the next frame of the simulation.
int Kernel::computeNextFrame() { // TODO expliquer la sortie qui est le nb personne restantte dans la simulation
    int out = 0;
    // Create a random number generator
    std::random_device rd;
    std::default_random_engine rng(rd());

    // Use std::shuffle to shuffle the shuffleIndex vector with the random number generator
    std::shuffle(shuffleIndex.begin(), shuffleIndex.end(), rng);

    // Iterate through the shuffled index vector
    for (auto &i : shuffleIndex) {
        // Check if the individual is already at the exit
        if (this->population[i].position.x == -1 && this->population[i].position.y == -1) {
            continue; // Move to the next individual
        }
        else {
            out ++;
        }
        // Initialize variables for position calculations
        int dimensionDep = static_cast<int>(std::sqrt(this->pMovements_s[this->population[i].from].size()));

        int2 coordAccMapPxy = {(this->population[i].position.x), (this->population[i].position.y)};
        int2 coordAccDepPxy = {(dimensionDep-1)/2, (dimensionDep-1)/2};

        int2 coordNextMapPxy = {0, 0};
        int2 coordNextDepPxy = {0, 0};

        uint coordAccMapPi = coordAccMapPxy.x + coordAccMapPxy.y * dimentionMap.x;
        uint coordAccDepPi = coordAccDepPxy.x + coordAccDepPxy.y * dimensionDep;
        uint coordNextMapPi = 0;
        uint coordNextDepPi = 0;

        // Step -1: Generate a list of available next positions
        std::vector<int2> allNextPositions;
        for (auto &nextPossiblePosition : this->directions_s[this->population[i].from]) {
            // Check if the position is within simulation boundaries
            coordNextMapPxy = {coordAccMapPxy.x + nextPossiblePosition.first, coordAccMapPxy.y + nextPossiblePosition.second};
            coordNextMapPi = coordNextMapPxy.x + coordNextMapPxy.y * dimentionMap.x;

            if (coordNextMapPxy.x < 0 || coordNextMapPxy.x >= dimentionMap.x ||
                coordNextMapPxy.y < 0 || coordNextMapPxy.y >= dimentionMap.y ||
                !isEmptyMapWall[coordNextMapPi] ||
                costMaps_s[population[i].from][coordNextMapPi] >= costMaps_s[population[i].from][coordAccMapPi]) {
                continue;
            }

            if (isEmptyMapPopulation[coordNextMapPi]) {
                allNextPositions.push_back({nextPossiblePosition.first, nextPossiblePosition.second});
            }
        }

        // Step -2: Calculate movement probabilities for available positions
        std::vector<std::pair<int2, float>> weightCumulCroissant;
        float sumWeight = 0.f;

        for (auto &nextPosition : allNextPositions) {
            coordNextMapPxy = {coordAccMapPxy.x + nextPosition.x, coordAccMapPxy.y + nextPosition.y};
            coordNextDepPxy = {coordAccDepPxy.x + nextPosition.x, coordAccDepPxy.y + nextPosition.y};
            coordNextMapPi = coordNextMapPxy.x + coordNextMapPxy.y * dimentionMap.x;
            coordNextDepPi = coordNextDepPxy.x + coordNextDepPxy.y * dimensionDep;

            float weight = this->costMaps_s[this->population[i].from][coordNextMapPi];
            float costWeight = weight * this->pMovements_s[this->population[i].from][coordNextDepPi];
            sumWeight += costWeight;
            weightCumulCroissant.push_back({coordNextMapPxy, costWeight});
        }

        // Step -3: Select the next position based on cumulative weights
        uint2 nextPos = {0, 0};
        float r = ((float)rand()/RAND_MAX) * sumWeight;

        for (auto &weightByPossibleIndex : weightCumulCroissant) {
            if (r < weightByPossibleIndex.second) {
                this->population[i].position.x = weightByPossibleIndex.first.x;
                this->population[i].position.y = weightByPossibleIndex.first.y;
                break;
            }
        }

        // TODO: Count the number of frames the individual has waited
        // Update if the individual has reached the exit
        if (this->costMaps_s[this->population[i].from][coordNextMapPi] == 0) {
            this->population[i].position.x = -1;
            this->population[i].position.y = -1;
        }
    }
    return out;
}

// GETTER AND SETTER CPU
const std::vector<individu> &Kernel::getPopulation() const {
    return population;
}
void Kernel::setPopulation(const std::vector<individu> &population) {
    Kernel::population = population;
}
const std::vector<std::vector<uint>> &Kernel::getCostMap() const {
    return costMaps_s;
}
void Kernel::setCostMap(const std::vector<std::vector<uint>> &costMap) {
    Kernel::costMaps_s = costMap;
}

// DESTRUCTOR CPU
Kernel::~Kernel()
{
}
#endif //KERNEL_CPU
