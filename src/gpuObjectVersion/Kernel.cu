/*******************************************************************************
// Fichier : Kernel.cpp
// Description : This file enables parallel computation when possible, otherwise, it performs the equivalent on the CPU. It calculates the displacement of each individual on the displacement grid for each frame.
// Auteur : Mathurin Champémont
// Date : 02/08/2023
*******************************************************************************/

#include "Kernel.hpp"

#ifdef GRAPHICS_CARD_PRESENT
// CONSTRUCTOR GPU
Kernel::Kernel()
{
}

// FUNCTIONS GPU
void Kernel::initKerel(Map map) {
}

void Kernel::computeNextFrame() {
}

// GETTER AND SETTER GPU

const std::vector<individu> &Kernel::getPopulation() const {
    return population;
}

void Kernel::setPopulation(const std::vector<individu> &population) {
    Kernel::population = population;
}

const std::vector<int> &Kernel::getCostMap() const {
    return costMap;
}

void Kernel::setCostMap(const std::vector<int> &costMap) {
    Kernel::costMap = costMap;
}

// DESTRUCTOR GPU
Kernel::~Kernel()
{
}

#endif //KERNEL_CPU
