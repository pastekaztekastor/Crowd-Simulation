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
    uint2 dimensions;

    /**
     * Initializes the map based on wall positions.
     * Clears the existing map and fills it with empty cells.
     * Then, marks the wall positions on the map.
     */
    void initMapFromWallPositions();
    /**
     * Adds a population to the map at the specified index.
     * Clears the existing positions of individuals from the map and marks them as empty cells.
     * Then, marks the exit positions of the population on the map.
     *
     * @param index Index of the population to be added to the map.
     */
    void addPopulationToMap(int index);

public:
    /**
     * Constructor that initializes a Map object using an image file.
     * The image is used to populate wallPositions and map.
     *
     * @param filePath Path to the image file.
     */
    Map(std::string filePath);
    /**
     * Constructor for the Map class without any parameters.
     * Sets the default dimensions of the map and initializes random wall positions with a default number of walls.
     * Then, initializes the map based on the wall positions.
     */
    Map();
    /**
     * Constructor for the Map class with a specified dimension.
     * Sets the dimensions of the map based on the 'dimension' parameter.
     * Initializes random wall positions with a default number of walls.
     * Then, initializes the map based on the wall positions.
     *
     * @param dimension Dimensions of the map to be created (width, height).
     */
    Map(uint2 dimension);
    /**
     * Constructor for the Map class with specified dimensions and a number of walls.
     * Sets the dimensions of the map based on the 'dimension' parameter.
     * Initializes random wall positions with the given number of walls 'nbWall'.
     * Then, initializes the map based on the wall positions.
     *
     * @param dimension Dimensions of the map to be created (width, height).
     * @param nbWall Number of walls to be randomly positioned on the map.
     */
    Map(uint2 dimension, int nbWall);
    /**
      * Constructor for the Map class with specified dimensions, a number of walls, and the number of populations.
      * Sets the dimensions of the map based on the 'dimension' parameter.
      * Initializes random wall positions with the given number of walls 'nbWall'.
      * Then, initializes the map based on the wall positions.
      * Initializes random populations with the given number of populations 'nbPopulations' and a default population size.
      *
      * @param dimension Dimensions of the map to be created (width, height).
      * @param nbWall Number of walls to be randomly positioned on the map.
      * @param nbPopulations Number of populations to be randomly initialized on the map.
      */
    Map(uint2 dimension, int nbWall, uint nbPopulations);
    /**
     * Constructor for the Map class with specified dimensions, a number of walls, the number of populations, and the population size.
     * Sets the dimensions of the map based on the 'dimension' parameter.
     * Initializes random wall positions with the given number of walls 'nbWall'.
     * Then, initializes the map based on the wall positions.
     * Initializes random populations with the given number of populations 'nbPopulations' and the specified 'populationSize'.
     *
     * @param dimension Dimensions of the map to be created (width, height).
     * @param nbWall Number of walls to be randomly positioned on the map.
     * @param nbPopulations Number of populations to be randomly initialized on the map.
     * @param populationSize Size of each population (number of individuals).
    */
    Map(uint2 dimension, int nbWall, uint nbPopulations, int populationSize);
    /**
     * Initializes random populations on the map.
     * Creates 'nbPopulations' new populations with the specified 'populationSize', a default number of exits,
     * and assigns them unique identifiers (IDs) ranging from 0 to nbPopulations-1.
     * Each population is initialized with the provided 'populationSize' number of individuals.
     *
     * @param nbPopulations Number of populations to be randomly initialized on the map.
     * @param populationSize Size of each population (number of individuals).
     */
    void initRandomPopulation(uint nbPopulations, int populationSize);
    /**
     * Initializes random wall positions on the map.
     * Clears the existing map and fills it with empty cells.
     * Randomly places 'nbWallPositions' walls within the map dimensions.
     * The positions are picked from empty cells on the map.
     *
     * @param nbWallPositions Number of walls to be randomly positioned on the map.
     */
    void initRandomWallPositions(uint nbWallPositions);
    /**
     * Initializes random wall positions on the map based on the specified percentage of wall occupation.
     * Calculates the number of walls to be placed as 'percentageOccupation * dimensions.x * dimensions.y'
     * and then calls the 'initRandomWallPositions' function with the calculated number of walls.
     *
     * @param percentageOccupation Percentage of wall occupation (ranging from 0.0 to 1.0) for the map dimensions.
     */
    void initRandomWallPositions(float percentageOccupation);
    /**
     * Initializes the map using an image file and returns a status code.
     * The image is processed to populate wallPositions and map.
     *
     * @param filePath Path to the image file.
     * @return Status code: 1 if successful, 0 if there was an error.
     */
    int initWithFile(std::string filePath);

    /**
     * Initializes the cost map for navigation based on exits and possible movements.
     * The cost map represents the minimum cost (distance) required to reach each cell from exits,
     * taking into account possible movements in up, down, left, and right directions.
     * The cost values are computed using flood fill.
     */
    void initCostMap();
    /**
     * Get the reference to the vector of populations.
     * Returns a constant reference to the vector of populations 'populations'.
     * This allows access to the populations without modifying them.
     *
     * @return Constant reference to the vector of populations.
     */
    const std::vector<Population> &getPopulations() const;
    /**
     * Set the vector of populations with the provided vector.
     * Assigns the input vector 'populations' to the 'Map::populations'.
     * This sets the populations of the map to the provided vector without making a copy.
     *
     * @param populations Vector of populations to be set for the map.
     */
    void setPopulations(const std::vector<Population> &populations);
    /**
     * Get the reference to the vector of wall positions.
     * Returns a constant reference to the vector of wall positions 'wallPositions'.
     * This allows access to the wall positions without modifying them.
     *
     * @return Constant reference to the vector of wall positions.
     */
    const std::vector<uint2> &getWallPositions() const;
    /**
     * Get the reference to the vector representing the map.
     * Returns a constant reference to the vector representing the map 'map'.
     * This allows access to the map without modifying its elements.
     *
     * @return Constant reference to the vector representing the map.
     */
    const std::vector<int> &getMap() const;
    /**
     * Set the vector representing the map with the provided vector.
     * Assigns the input vector 'map' to the 'Map::map'.
     * This sets the map of the object to the provided vector without making a copy.
     *
     * @param map Vector representing the map to be set for the object.
     */
    void setMap(const std::vector<int> &map);
    /**
     * Get the reference to the dimensions of the map.
     * Returns a constant reference to the dimensions of the map 'dimensions'.
     * This allows access to the dimensions without modifying them.
     *
     * @return Constant reference to the dimensions of the map.
     */
    const uint2 &getDimensions() const;
    /**
     * Set the dimensions of the map with the provided dimensions.
     * Assigns the input 'dimensions' to the 'Map::dimensions'.
     * This sets the dimensions of the map to the provided dimensions without making a copy.
     *
     * @param dimensions Dimensions (width and height) of the map to be set.
     */
    void setDimensions(const uint2 &dimensions);
    /**
     * Add a new population to the map.
     * Appends the provided 'population' object to the vector of populations.
     * This adds the 'population' to the list of populations on the map.
     *
     * @param population Population object to be added to the map.
     */
    void addPopulation(Population population);
    /**
     * Print the populations of the map.
     * Displays information about each population present on the map, including their states and positions.
     */
    void printPopulations();
    /**
     * Print the positions of all the walls on the map.
     * Displays the coordinates of all the walls present on the map.
     */
    void printWallPositions();
    /**
     * Print the map for a specific population.
     * Initializes the map based on the wall positions, adds the specified population to the map,
     * and then prints the map with appropriate symbols for each cell.
     *
     * @param index Index of the population to be displayed on the map.
     */
    void printMap(int index);
    /**
     * Print the entire map with all populations and wall positions.
     * Calls the 'printPopulations', 'printWallPositions', and 'printMap' functions
     * to display information about populations and the map layout.
     */
    void print();

    ~Map();
};

#endif // MAP_HPP