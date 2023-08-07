/*******************************************************************************
// Fichier : Kernel.cpp
// Description : This file enables parallel computation when possible, otherwise, it performs the equivalent on the CPU. It calculates the displacement of each individual on the displacement grid for each frame.
// Auteur : Mathurin Champémont
// Date : 02/08/2023
*******************************************************************************/

#include "Kernel.hpp"

#ifndef GRAPHICS_CARD_PRESENT

// CONSTRUCTOR CPU
Kernel::Kernel()
{
}

// FUNCTIONS CPU
void Kernel::initKerel(const Map& map) {
    for (int i = 0; i< map.getPopulations().size(); i++) {
        //Création du vecteur population
        for(auto &&individu : map.getPopulations()[i].getStates()){
            this->population.push_back({individu, i});
        }
        // copy des cartes de couts
        this->costMap.push_back(map.getPopulations()[i].getMapCost());
        // cpoy des déplacements
        this->pMovement.push_back(map.getPopulations()[i].getPMovement());
    }
}
void Kernel::computeNextFrame() {
    // Créez un générateur de nombres aléatoires
    std::random_device rd;
    std::default_random_engine rng(rd());

    // Utilisez std::shuffle pour mélanger le vecteur avec le générateur de nombres aléatoires
    std::shuffle(population.begin(), population.end(), rng);

    // Maintenant, le vecteur populations est mélangé
    // Vous pouvez itérer à travers lui pour voir l'ordre mélangé des éléments
    for (const auto &individu: population) {

    }
}

// GETTER AND SETTER CPU
const std::vector<individu> &Kernel::getPopulation() const {
    return population;
}
void Kernel::setPopulation(const std::vector<individu> &population) {
    Kernel::population = population;
}
const std::vector<std::vector<uint>> &Kernel::getCostMap() const {
    return costMap;
}
void Kernel::setCostMap(const std::vector<std::vector<uint>> &costMap) {
    Kernel::costMap = costMap;
}

// DESTRUCTOR CPU
Kernel::~Kernel()
{
}
#endif //KERNEL_CPU
