/*******************************************************************************
 * Fichier : Map.cpp
 * Description : Fichier de définition de la classe Map qui représente la topologie de la simulation de foule.
 * Auteur : Mathurin Champémont
 * Date : 06/07/2023
*******************************************************************************/

#include "Map.hpp"

// Private

// Initializes the map based on wall positions.
void Map::initMapFromWallPositions()
{
    // Clear the existing map
    this->map.clear();

    // Fill the map with empty cells initially
    for (size_t i = 0; i < this->dimensions.x * this->dimensions.y; i++)
    {
        this->map.push_back(__MAP_EMPTY__);
    }

    // Mark wall positions on the map
    for (auto &&wall : this->wallPositions)
    {
        this->map[wall.y * this->dimensions.x + wall.x] = __MAP_WALL__;
    }
}
// Adds a population to the map at the specified index.
void Map::addPopulationToMap(int index)
{
    // Clear the existing positions of individuals from the map and mark them as picked cells
    for (auto &&etat : this->populations[index].getStates())
    {
        this->map[etat.y * this->dimensions.x + etat.x] = 0;
    }

    // Mark the exit positions of the population on the map
    for (auto &&exit : this->populations[index].getExits())
    {
        this->map[exit.y * this->dimensions.x + exit.x] = __MAP_EXIT__;
    }
}

// Public
// Constructor that initializes a Map object using an image file.
Map::Map(std::string filePath) {
    // Initialize the map from the provided image file
    if (initWithFile(filePath)) {
        // Initialize the map data based on the wall positions
        initMapFromWallPositions();
    }
}

Map::Map()
{
    // Set the default dimensions of the map
    this->dimensions = {__MAP_NOMINALE_X_DIM__, __MAP_NOMINALE_Y_DIM__};

    // Initialize random wall positions with a default number of walls
    Map::initRandomWallPositions((uint)__MAP_NOMINALE_WALL__);

    // Initialize the map based on the wall positions
    Map::initMapFromWallPositions();
}

// Constructor for Map class with a specified dimension.
Map::Map(uint2 dimension)
{
    // Set the dimensions of the map based on the specified 'dimension'.
    this->dimensions = dimension;

    // Initialize random wall positions with a default number of walls
    Map::initRandomWallPositions((uint)__MAP_NOMINALE_WALL__);

    // Initialize the map based on the wall positions
    Map::initMapFromWallPositions();
}

// Constructor for Map class with specified dimensions and a number of walls.
Map::Map(uint2 dimension, int nbWall)
{
    // Set the dimensions of the map based on the specified 'dimension'.
    this->dimensions = dimension;

    // Initialize random wall positions with the given number of walls 'nbWall'.
    Map::initRandomWallPositions((uint)nbWall);

    // Initialize the map based on the wall positions.
    Map::initMapFromWallPositions();
}

// Constructor for Map class with specified dimensions, a number of walls, and the number of populations.
Map::Map(uint2 dimension, int nbWall, uint nbPopulations)
{
    // Set the dimensions of the map based on the specified 'dimension'.
    this->dimensions = dimension;

    // Initialize random wall positions with the given number of walls 'nbWall'.
    Map::initRandomWallPositions((uint)nbWall);

    // Initialize the map based on the wall positions.
    Map::initMapFromWallPositions();

    // Initialize random populations with the given number of populations 'nbPopulations' and a default population size.
    Map::initRandomPopulation(nbPopulations, __MAP_NOMINALE_POPULATION_SIZE__);
}

// Constructor for Map class with specified dimensions, a number of walls, the number of populations, and the population size.
Map::Map(uint2 dimension, int nbWall, uint nbPopulations, int populationSize)
{
    // Set the dimensions of the map based on the specified 'dimension'.
    this->dimensions = dimension;

    // Initialize random wall positions with the given number of walls 'nbWall'.
    Map::initRandomWallPositions((uint)nbWall);

    // Initialize the map based on the wall positions.
    Map::initMapFromWallPositions();

    // Initialize random populations with the given number of populations 'nbPopulations' and the specified 'populationSize'.
    Map::initRandomPopulation(nbPopulations, populationSize);
}

// Initializes random populations on the map.
void Map::initRandomPopulation(uint nbPopulations, int populationSize)
{
    for (size_t i = 0; i < nbPopulations; i++)
    {
        // Create a new population with a unique identifier 'i', the specified 'populationSize',
        // a default number of exits, and the map dimensions and elements.
        // TODO fix
        // this->populations.push_back(Population(std::to_string(i), populationSize, __POPULATION_NOMINAL_NB_EXIT__, this->dimensions, this->map));
    }
}

// Initializes random wall positions on the map.
void Map::initRandomWallPositions(uint nbWallPositions)
{
    // Clear the existing map and fill it with empty cells.
    this->map.clear();
    for (size_t i = 0; i < this->dimensions.x * this->dimensions.y; i++)
    {
        this->map.push_back(__MAP_EMPTY__);
    }

    // Randomly place 'nbWallPositions' walls on the map.
    for (size_t i = 0; i < nbWallPositions; i++)
    {
        uint2 coord;
        do
        {
            // Generate a random coordinate within the map dimensions (0 to dimensions.x-1 and 0 to dimensions.y-1).
            coord = {rand() % this->dimensions.x, rand() % this->dimensions.y};

            // Repeat until a valid random coordinate is found that corresponds to an empty cell on the map.
        } while (this->map[coord.y * this->dimensions.x + coord.x] != __MAP_EMPTY__);

        // Add the random coordinate to the wallPositions vector, representing a wall's position.
        this->wallPositions.push_back(coord);
    }
}

// Initializes random wall positions on the map based on the specified percentage of wall occupation.
void Map::initRandomWallPositions(float percentageOccupation)
{
    // Calculate the number of walls to be placed based on the specified percentage of wall occupation.
    uint nbWallPositions = static_cast<uint>(percentageOccupation * static_cast<float>(this->dimensions.x) * static_cast<float>(this->dimensions.y));

    // Call the 'initRandomWallPositions' function with the calculated 'nbWallPositions'.
    Map::initRandomWallPositions(nbWallPositions);
}

void Map::initCostMap() {
    // Création de la carte de cout temporaire.
    std::vector<int> mapCostTmp;
    // pour chaque population
    for (Population& population : populations) {
        // Initialisation de la carte de coût
        mapCostTmp.resize(dimensions.x * dimensions.y, dimensions.x * dimensions.y);

        // Initialisation des sorties avec une valeur de coût de 0.
        for (const int2& exit : population.getExits()) {
            uint index = exit.x + exit.y * dimensions.x;
            mapCostTmp[index] = 0;
        }

        // Initialisation des obstacles avec une valeur de coût de -1
        for (const uint2& wall : wallPositions) {
            uint index = wall.x + wall.y * dimensions.x;
            mapCostTmp[index] = -1;
        }

        bool needsUpdate = true;
        int nbPasses = 0;

        while (needsUpdate) {
            needsUpdate = false;

            // De gauche à droite et de haut en bas
            for (int i = 0; i < dimensions.x; i++) {
                for (int j = 0; j < dimensions.y; j++) {
                    int index = i + j * dimensions.x;
                    if (i > 0 && mapCostTmp[index - 1] != -1 && mapCostTmp[index] > mapCostTmp[index - 1] + 1) {
                        mapCostTmp[index] = mapCostTmp[index - 1] + 1;
                        needsUpdate = true;
                    }
                    if (j > 0 && mapCostTmp[index - dimensions.x] != -1 && mapCostTmp[index] > mapCostTmp[index - dimensions.x] + 1) {
                        mapCostTmp[index] = mapCostTmp[index - dimensions.x] + 1;
                        needsUpdate = true;
                    }
                }
            }

            // De droite à gauche et de bas en haut
            for (int i = (int) dimensions.x - 1; i >= 0; i--) {
                for (int j = (int) dimensions.y - 1; j >= 0; j--) {
                    int index = i + j * dimensions.x;

                    if (i < dimensions.x - 1 && mapCostTmp[index + 1] != -1 & mapCostTmp[index] > mapCostTmp[index + 1] + 1) {
                        mapCostTmp[index] = mapCostTmp[index + 1] + 1;
                        needsUpdate = true;
                    }
                    if (j < dimensions.y - 1 && mapCostTmp[index + dimensions.x] != -1 & mapCostTmp[index] > mapCostTmp[index + dimensions.x] + 1) {
                        mapCostTmp[index] = mapCostTmp[index + dimensions.x] + 1;
                        needsUpdate = true;
                    }
                }
            }
            nbPasses ++;
        }
        std::vector<uint> mapCostFinal;
        for (auto & cost : mapCostTmp) {
            mapCostFinal.push_back((uint)cost);
        }
        population.setMapCost(mapCostFinal);
        std::cout << "NOMBRE DE PASSE CARTE COUT : " << nbPasses << std::endl;
    }
}
/*

// Initializes the cost map for navigation based on exits and possible movements.
void Map::initCostMap() {
    // Fill the cost map with high values
    for (Population &population : populations) {
        std::vector<int> mapCostTmp;
        mapCostTmp.resize(dimensions.x * dimensions.y, dimensions.x * dimensions.y + 1);

        // Set exit positions
        for (const auto &exit : population.getExits()) {
            mapCostTmp[exit.x + exit.y * dimensions.x] = 0;
        }

        // Define possible movements (up, down, left, right)
        std::vector<std::pair<int, int>> directions = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

        // Perform flood fill until the cost map is filled
        bool updated;
        do {
            updated = false;

            // Iterate over each cell on the map
            for (uint y = 0; y < dimensions.y; ++y) {
                for (uint x = 0; x < dimensions.x; ++x) {
                    // Check if the current cell is not a wall
                    if (map[x + y * dimensions.x] != __MAP_WALL__) {
                        int currentCost = mapCostTmp[x + y * dimensions.x];

                        // Explore possible movements from the current cell
                        for (const auto &direction : directions) {
                            int newX = x + direction.first;
                            int newY = y + direction.second;

                            // Check if the new coordinates are valid and not a wall
                            if (newX >= 0 &&
                                newY >= 0 &&
                                newX < static_cast<int>(dimensions.x) &&
                                newY < static_cast<int>(dimensions.y) &&
                                map[newX + newY * dimensions.x] != __MAP_WALL__) {

                                int newCost = currentCost + 1;

                                // Update the cost if necessary
                                if (newCost < mapCostTmp[newX + newY * dimensions.x]) {
                                    mapCostTmp[newX + newY * dimensions.x] = newCost;
                                    updated = true;
                                }
                            }
                        }
                    }
                }
            }
        } while (updated);

        // Convert the temporary cost map to the final format and set it for the population
        std::vector<uint> mapCostFinal;
        for (const auto &cost : mapCostTmp) {
            mapCostFinal.push_back(static_cast<uint>(cost));
        }
        population.setMapCost(mapCostFinal);
    }
}

*/
// Initializes the map using an image file and returns a status code.
int Map::initWithFile(std::string filePath) {
    // Load the image
    cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Error importing image: " << filePath << std::endl;
        return 0;  // Return 0 to indicate failure
    }

    // Assign values to the Map structure based on the loaded image
    this->dimensions = {static_cast<uint>(image.cols), static_cast<uint>(image.rows)};

    // Iterate through the image to populate wallPositions and map
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            cv::Scalar color(pixel[0], pixel[1], pixel[2]);

            if (color == __COLOR_BLACK__) {
                this->wallPositions.push_back({static_cast<uint>(x), static_cast<uint>(y)});
            }
        }
    }

    // Fill the map
    map.resize(dimensions.x * dimensions.y, __MAP_EMPTY__); // Initialize to __MAP_EMPTY__

    // Place the walls
    for (const auto &wallPos : wallPositions) {
        map[wallPos.x + wallPos.y * dimensions.x] = __MAP_WALL__;
    }

    return 1;  // Return 1 to indicate success
}

// Get the reference to the vector of populations.
const std::vector<Population> &Map::getPopulations() const
{
    // Return a constant reference to the vector of populations 'populations'.
    // This allows access to the populations without modifying them.
    return populations;
}

// Set the vector of populations with the provided vector.
void Map::setPopulations(const std::vector<Population> &populations)
{
    // Assign the input vector 'populations' to the 'Map::populations'.
    // This sets the populations of the map to the provided vector without making a copy.
    Map::populations = populations;
}

// Get the reference to the vector of wall positions.
const std::vector<uint2> &Map::getWallPositions() const
{
    // Return a constant reference to the vector of wall positions 'wallPositions'.
    // This allows access to the wall positions without modifying them.
    return wallPositions;
}

// Get the reference to the vector representing the map.
const std::vector<int> &Map::getMap() const
{
    // Return a constant reference to the vector representing the map 'map'.
    // This allows access to the map without modifying its elements.
    return map;
}

// Set the vector representing the map with the provided vector.
void Map::setMap(const std::vector<int> &map)
{
    // Assign the input vector 'map' to the 'Map::map'.
    // This sets the map of the object to the provided vector without making a copy.
    Map::map = map;
}

// Get the reference to the dimensions of the map.
const uint2 &Map::getDimensions() const
{
    // Return a constant reference to the dimensions of the map 'dimensions'.
    // This allows access to the dimensions without modifying them.
    return dimensions;
}

// Set the dimensions of the map with the provided dimensions.
void Map::setDimensions(const uint2 &dimensions)
{
    // Assign the input 'dimensions' to the 'Map::dimensions'.
    // This sets the dimensions of the map to the provided dimensions without making a copy.
    Map::dimensions = dimensions;
}

// Add a new population to the map.
void Map::addPopulation(Population population)
{
    // Push the provided 'population' object to the vector of populations.
    // This adds the 'population' to the list of populations on the map.
    for (auto & a : population.getStates()){
        for (auto & b : this->wallPositions){
            if (a.x == b.x && a.y == b.y) {
                population.removeStat(a);
            }
        }
    }
    this->populations.push_back(population);

}
// Print the populations of the map.
void Map::printPopulations()
{
    std::cout << "Displaying populations on the map:" << std::endl;
    for (auto &&population : this->populations)
    {
        population.print(this->dimensions);
    }
}

// Print the positions of all the walls.
void Map::printWallPositions()
{
    std::cout << "Positions of all the walls:" << std::endl;
    for (auto &&wall : wallPositions)
    {
        std::cout << "[" << wall.x << "," << wall.y << "] ";
    }
    std::cout << std::endl;
}

// Print the map for a specific population.
void Map::printMap(int index)
{
    std::cout << "Map of population " << index << ":" << std::endl;

    // Reinitialize the map based on the wall positions.
    Map::initMapFromWallPositions();

    // Add the specific population to the map.
    Map::addPopulationToMap(index);

    // Print the column indices at the top.
    std::cout << "  ";
    for (int x = 0; x < this->dimensions.x; x++)
    {
        printf(" %2d", x);
    }
    std::cout << "  " << std::endl;

    // Print the map with appropriate symbols for each cell.
    for (int y = 0; y < this->dimensions.y; y++)
    {
        printf("%2d ", y);
        for (int x = 0; x < this->dimensions.x; x++)
        {
            if (this->map[y * this->dimensions.x + x] == __MAP_EMPTY__)
                std::cout << "   ";
            else if (this->map[y * this->dimensions.x + x] == __MAP_WALL__)
                std::cout << "///";
            else if (this->map[y * this->dimensions.x + x] == __MAP_EXIT__)
                std::cout << " X ";
            else if (this->map[y * this->dimensions.x + x] >= 0)
                std::cout << " o ";
        }
        std::cout << std::endl;
    }
}

// Print the entire map with all populations and wall positions.
void Map::print()
{
    Map::printPopulations();
    Map::printWallPositions();
    for (int i = 0; i < this->populations.size(); i++)
    {
        Map::printMap(i);
    }
}

Map::~Map(){
    //TO DO
}

std::vector<bool> Map::getMapWall() const {
    std::vector<bool> out;
    for (auto &i : map) {
        if (i == __MAP_WALL__)  { out.push_back(false); }
        else                    { out.push_back(true);  }
    }
    return out;
}
