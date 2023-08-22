/*******************************************************************************
// Fichier : Kernel.hpp
// Description : Header of the code Kernel.cpp. This file enables parallel computation when possible, otherwise, it performs the equivalent on the CPU. It calculates the displacement of each individual on the displacement grid for each frame.
// Auteur : Mathurin Champémont
// Date : 02/08/2023
*******************************************************************************/

#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "utils/utils.hpp"
#include "Map.hpp"

class Kernel
{
private:
    uint2 dimentionMap;
    std::vector<individu> population;
    std::vector<uint> shuffleIndex;
    std::vector<bool> isEmptyMap;
    std::vector<std::vector<uint>> costMaps_s;
    std::vector<std::vector<float>> pMovements_s;
    std::vector<std::vector<std::pair<int, int>>> directions_s

#ifdef GRAPHICS_CARD_PRESENT
    // Graphique card Stuff
#endif

public:
    Kernel(const Map& map);

    void initKerel(const Map& map);

    const std::vector<std::vector<uint>> &getCostMap() const;
    void setCostMap(const std::vector<std::vector<uint>> &costMap);

    const std::vector<individu> &getPopulation() const;
    void setPopulation(const std::vector<individu> &population);

    // Computes the next frame of the simulation.
    /**
     * Computes the next frame of the simulation for each individual in a shuffled order.
     * Uses movement probabilities, cost maps, and available directions to determine the next positions.
     */
    void computeNextFrame();

    ~Kernel();
};

#endif //KERNEL_HPP