/*******************************************************************************
 * Fichier : Map.cpp
 * Description : Fichier de définition de la classe Map qui représente la topologie de la simulation de foule.
 * Auteur : Mathurin Champémont
 * Date : 06/07/2023
*******************************************************************************/

#include "Map.hpp"

void Map::createMapFromPositions(){
    for (size_t i = 0; i < this->dimentions.x * this->dimentions.y; i++)
    {
        this->map.push_back(__MAP_EMPTY__);
    }
    for (auto && wall : this->wallPositions)
    {
        this->map[wall.y * this->dimentions.x + wall.x] = __MAP_WALL__;
    }
}

Map::Map(){
    this->dimentions = {5,5};
    Map::initRandomWallPositions((uint)10);
    Map::initRandomPopulation(1, 10);
    Map::createMapFromPositions();
}

void Map::initRandomPopulation(uint nbPopulations, int populationSize){
    for (size_t i = 0; i < nbPopulations ; i++)
    {
        this->populations.push_back(Population(std::to_string(i), populationSize, 1, this->dimentions, this->map));
    }
}

void Map::initRandomWallPositions(uint nbWallPositions){
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
    initRandomWallPositions(float(this->dimentions.x)*float(this->dimentions.y)*purcenteOccupation);
}


void Map::initWithFile(std::string filePath){
    //TO DO
}

void Map::setWallPositions(std::vector<uint2> wallPositions){
    this->wallPositions = wallPositions;
}

void Map::addPopulation(Population population){
    this->populations.push_back(population);
}

std::vector<int>        Map::getMap() const{
    return this->map;
}
std::vector<Population> Map::getPopulations() const{
    return this->populations;
}
std::vector<uint2>      Map::getWallPositions() const{
    return this->wallPositions;    
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
        std::cout << " x : " << wall.x << " - y : " << wall.y << std::endl;
    }
    
}
void Map::printMapWall(){
    std::cout << " Carte des mures : " << std::endl;
    std::cout <<"  ";
        for (int x = 0; x < this->dimentions.x; x++)
        {
            printf("%2d  ",x); 
        }
        std::cout<<"  "<<std::endl;

    for (int y = 0; y < this->dimentions.y; y++)
    {
        printf("%2d ",y); 
        for (int x = 0; x < this->dimentions.x; x++)
        {
            printf(" %2d ", this->map[y * this->dimentions.x + x]);
        }
        std::cout << std::endl;
    }
}
void Map::print(){
    //TO DO
}

Map::~Map(){
    //TO DO
}
