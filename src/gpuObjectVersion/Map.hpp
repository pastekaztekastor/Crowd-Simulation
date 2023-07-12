#include "utils/utils.hpp"
#include "Population.hpp"

class Map

{
private:
    std::vector<Population> populations;
    std::vector<uint2> wallPositions;
    std::vector<int> map;
    uint2 dimentions;

public:
    Map();
    Map(std::vector<Population> new_populations, std::vector<uint2> new_wallPositions, std::vector<int> new_map, uint2 dimentions);

    void initRandomPopulation(uint nbPopulations);
    void initRandomPopulation(float purcentPopulations);

    void initRandomWallPositions(uint nbWallPositions);
    void initRandomWallPositions(float purcentWallPositions);

    void initWithFile(std::string filePath);
    /**
     * @brief Créer la liste des mure uniquement à pratire du fichier PNG obligatoire
     * le vide c'est noir pure
     * les mure c'est blanc pure
     * 
     * @return void
     */
    
    void setWallPosition(std::vector<uint2> new_wallPositions);

    void addPopulation(Population populations);

    std::vector<int>        getMap() const;
    std::vector<Population> getPopulations() const;
    std::vector<uint2>      getWallPositions() const;

    void printPopulations();
    void printWallPositions();
    void printMap();
    void print();

    ~Map();
};