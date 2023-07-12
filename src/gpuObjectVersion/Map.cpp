#include "Map.hpp"

Map(){
    //TO DO
}
Map(std::vector<Population> new_populations, std::vector<uint2> new_wallPositions, std::vector<int> new_map, uint2 dimentions){
    //TO DO
}

void Map::initRandomPopulation(uint nbPopulations, int populationSize){
    for (size_t i = 0; i < nbPopulations ; i++)
    {
        this.populations.add(new Population(to_string(i), populationSize, 1, this.dimentions, this.map));
    }
}

void Map::initRandomWallPositions(uint nbWallPositions){
    for (size_t i = 0; i < nbWallPositions; i++)
    {
        uint2 coord;
        do 
        {
            coord = uint2(rand()%this.dimentions.x, rand()%this.dimentions.y);
            
        } while ( (this.map[coord.y * this.dimentions.x + coord.y] != __MAP_EMPTY__));

        this.wallPosition.push_back(coord);
    }
}
void Map::initRandomWallPositions(float purcenteOccupation){
    initRandomWallPositions(float(this.dimension.x)*float(this.dimension.y)*purcenteOccupation);
}


void Map::initWithFile(std::string filePath){
    //TO DO
}

void Map::setWallPositions(std::vector<uint2> wallPositions){
    this.wallPositions = wallPositions;
}

void Map::addPopulation(Population population){
    this.populations.push_back(population)
}

std::vector<int>        Map::getMap() const{
    //TO DO
}
std::vector<Population> Map::getPopulations() const{
    //TO DO
}
std::vector<uint2>      Map::getWallPositions() const{
    //TO DO
}

void Map::printPopulations(){
    //TO DO
}
void Map::printWallPositions(){
    //TO DO
}
void Map::printMap(){
    //TO DO
}
void Map::print(){
    //TO DO
}

~Map(){
    //TO DO
}
