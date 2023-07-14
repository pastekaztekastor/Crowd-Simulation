/*******************************************************************************
 * Fichier : Map.cpp
 * Description : Fichier de définition de la classe Map qui représente la topologie de la simulation de foule.
 * Auteur : Mathurin Champémont
 * Date : 06/07/2023
*******************************************************************************/

#include "Map.hpp"

// Private

void Map::initMapFromWallPositions(){
    this->map.clear();
    for (size_t i = 0; i < this->dimentions.x * this->dimentions.y; i++)
    {
        this->map.push_back(__MAP_EMPTY__);
    }
    for (auto && wall : this->wallPositions)
    {
        this->map[wall.y * this->dimentions.x + wall.x] = __MAP_WALL__;
    }
}
void Map::addPopulationToMap(int index) {
    for (auto && etat : this->populations[index].getEtats()) {
        this->map[etat.y * this->dimentions.x + etat.x] = 0;
    }
    for (auto && exit : this->populations[index].getExits()) {
        this->map[exit.y *this->dimentions.x + exit.x] = __MAP_EXIT__;
    }
}

// Public

Map::Map(){
    this->dimentions = {__MAP_NOMINALE_X_DIM__, __MAP_NOMINALE_Y_DIM__};
    Map::initRandomWallPositions((uint)__MAP_NOMINALE_WALL__);
    Map::initMapFromWallPositions();
}
Map::Map(uint2 dimension) {
    this->dimentions = dimension;
    Map::initRandomWallPositions((uint)__MAP_NOMINALE_WALL__);
    Map::initMapFromWallPositions();
}
Map::Map(uint2 dimension, int nbWall) {
    this->dimentions = dimension;
    Map::initRandomWallPositions((uint) nbWall);
    Map::initMapFromWallPositions();
}
Map::Map(uint2 dimension, int nbWall, uint nbPopulations) {
    this->dimentions = dimension;
    Map::initRandomWallPositions((uint) nbWall);
    Map::initMapFromWallPositions();
    Map::initRandomPopulation(nbPopulations, __MAP_NOMINALE_POPULATION_SIZE__);
}
Map::Map(uint2 dimension, int nbWall, uint nbPopulations, int populationSize) {
    this->dimentions = dimension;
    Map::initRandomWallPositions((uint) nbWall);
    Map::initMapFromWallPositions();
    Map::initRandomPopulation(nbPopulations, populationSize);
}

void Map::initRandomPopulation(uint nbPopulations, int populationSize){
    for (size_t i = 0; i < nbPopulations ; i++)
    {
        this->populations.push_back(Population(std::to_string(i), populationSize, __POPULATION_NOMINAL_NB_EXIT__, this->dimentions, this->map));
    }
}
void Map::initRandomWallPositions(uint nbWallPositions){
    for (size_t i = 0; i < this->dimentions.x * this->dimentions.y; i++)
    {
        this->map.push_back(__MAP_EMPTY__);
    }
    for (size_t i = 0; i < nbWallPositions; i++)
    {
        uint2 coord;
        do 
        {
            coord = {rand()%this->dimentions.x, rand()%this->dimentions.y};
            
        } while ( (this->map[coord.y * this->dimentions.x + coord.y] != __MAP_EMPTY__));

        this->wallPositions.push_back(coord);
    }
}
void Map::initRandomWallPositions(float purcenteOccupation){
    Map::initRandomWallPositions(float(this->dimentions.x)*float(this->dimentions.y)*purcenteOccupation);
}
void Map::initWithFile(std::string filePath){
    //TO DO
}

const std::vector<Population> &Map::getPopulations() const {
    return populations;
}
void Map::setPopulations(const std::vector<Population> &populations) {
    Map::populations = populations;
}
const std::vector<uint2> &Map::getWallPositions() const {
    return wallPositions;
}
const std::vector<int> &Map::getMap() const {
    return map;
}
void Map::setMap(const std::vector<int> &map) {
    Map::map = map;
}
const uint2 &Map::getDimensions() const {
    return dimentions;
}
void Map::setDimensions(const uint2 &dimensions) {
    Map::dimentions = dimensions;
}

void Map::addPopulation(Population population){
    this->populations.push_back(population);
}

void Map::printPopulations(){
    std::cout << " Affichage des populations de la carte : " << std::endl;
    for (auto && population : this->populations)
    {
        population.print(this->dimentions);
    }
    
}
void Map::printWallPositions(){
    std::cout << " Positions de tous les mures : " << std::endl;
    for (auto && wall : wallPositions)
    {
        std::cout << "[" << wall.x << "," << wall.y << "] ";
    }
    std::cout << std::endl;
}
void Map::printMap(int index){
    std::cout << " Carte de la population : " << std::endl;
    Map::initMapFromWallPositions();
    Map::addPopulationToMap(index);
    std::cout <<"  ";
        for (int x = 0; x < this->dimentions.x; x++)
        {
            printf(" %2d",x);
        }
        std::cout<<"  "<<std::endl;

    for (int y = 0; y < this->dimentions.y; y++)
    {
        printf("%2d ",y); 
        for (int x = 0; x < this->dimentions.x; x++)
        {
            if (this->map[y * this->dimentions.x + x] == __MAP_EMPTY__) std::cout << "   ";
            if (this->map[y * this->dimentions.x + x] == __MAP_WALL__) std::cout << "///";
            if (this->map[y * this->dimentions.x + x] == __MAP_EXIT__) std::cout << " X ";
            if (this->map[y * this->dimentions.x + x] >= 0) std::cout << "[P]";
        }
        std::cout << std::endl;
    }
}
void Map::print(){
    Map::printPopulations();
    Map::printWallPositions();
    for (int i = 0; i < this->populations.size(); i++){
        Map::printMap(i);
    }
}

Map::~Map(){
    //TO DO
}