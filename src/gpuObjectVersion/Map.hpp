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
    Map(uint2 dimension);
    Map(uint2 dimension, int nbWall);
    Map(uint2 dimension, int nbWall, uint nbPopulations);
    Map(uint2 dimension, int nbWall, uint nbPopulations, int populationSize);

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
    const std::vector<Population> &getPopulations() const;
    void setPopulations(const std::vector<Population> &populations);
    const std::vector<uint2> &getWallPositions() const;
    const std::vector<int> &getMap() const;
    void setMap(const std::vector<int> &map);
    const uint2 &getDimensions() const;
    void setDimensions(const uint2 &dimensions);

    void addPopulation(Population population);

    void printPopulations();
    void printWallPositions();
    void printMap(int index);
    void print();

    ~Map();
};

#endif // MAP_HPP