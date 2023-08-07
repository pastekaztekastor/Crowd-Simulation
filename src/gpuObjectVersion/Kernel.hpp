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
    std::vector<individu> population;
    std::vector<std::vector<uint>> costMap;
    std::vector<std::vector<float>> pMovement;

#ifdef GRAPHICS_CARD_PRESENT
    // Graphique card Stuff
#endif

public:
    Kernel();

    void initKerel(const Map& map);

    const std::vector<std::vector<uint>> &getCostMap() const;
    void setCostMap(const std::vector<std::vector<uint>> &costMap);

    const std::vector<individu> &getPopulation() const;
    void setPopulation(const std::vector<individu> &population);

    void computeNextFrame();

    ~Kernel();
};

#endif //KERNEL_HPP