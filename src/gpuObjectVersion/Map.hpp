/*******************************************************************************
// Fichier : Map.hpp
// Description : Fichier d'en-tête de la classe Map qui représente la topologie de la simulation de foule.
// Auteur : Mathurin Champémont
// Date : 06/07/2023
*******************************************************************************/

#ifndef MAP_HPP
#define MAP_HPP

// Déclaration de la classe Map

#include "utils/utils.hpp"
#include "Population.hpp"

class Map
{
private:
    std::vector<Population> populations;
    std::vector<uint2> wallPositions;
    std::vector<int> map;
    uint2 dimentions;

    void initMapFromWallPositions();
    void addPopulationToMap(int index);

public:
    Map();

    void initRandomPopulation(uint nbPopulations, int populationSize);

    void initRandomWallPositions(uint nbWallPositions);
    void initRandomWallPositions(float purcenteOccupation);

    void initWithFile(std::string filePath);
    /**
     * @brief Créer la liste des mure uniquement à pratire du fichier PNG obligatoire
     * le vide c'est noir pure
     * les mure c'est blanc pure
     * 
     * @return void
     */
    
    void setWallPositions(std::vector<uint2> wallPositions);

    void addPopulation(Population population);

    std::vector<int>        getMap() const;
    std::vector<Population> getPopulations() const;
    std::vector<uint2>      getWallPositions() const;

    void printPopulations();
    void printWallPositions();
    void printMap(int index);
    void print();

    ~Map();
};

#endif // MAP_HPP