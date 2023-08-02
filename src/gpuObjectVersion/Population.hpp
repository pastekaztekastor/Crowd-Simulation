/*******************************************************************************
// File: Population.hpp
// Description: Header file for the Population class representing a population in the crowd simulation.
// Author: Mathurin Champémont
// Date: 06/07/2023
*******************************************************************************/

#ifndef POPULATION_HPP
#define POPULATION_HPP

#include "utils/utils.hpp"

class Population
{
private:
    std::string name;               // Name of the population
    std::vector<int3> states;       // Current states of individuals in the population
    std::vector<int2> exits;        // Positions of exits in the simulation
    std::vector<uint> mapCost;      // Cost map for navigation
    std::vector<float> pDeplacement;// Probability of displacement for each individual
    color col;                      // Color representing the population in the visualization

public:
    // Default constructor
    Population();

    // Constructor with a specified name
    Population(std::string name);

    // Constructor with parameters to initialize the population
    Population(std::string name, int nbPopulations, int nbExit, uint2 simulationDim, std::vector<int> mapElements);

    // Initialize states randomly for the population
    void initRandomStates(int nbPopulations, uint2 simulationDim, std::vector<int> mapElements);
    /**
     * Initializes states randomly for the population.
     * Randomly assigns positions to 'nbPopulations' individuals within the simulation dimensions.
     * The individuals' positions are picked from empty cells on the map, and no two individuals
     * are assigned the same position.
     *
     * @param nbPopulations Number of individuals in the population.
     * @param simulationDim Dimensions of the simulation grid (width, height).
     * @param mapElements Map elements representing the state of each cell in the simulation grid.
    */

    // Initialize exit positions randomly for the population
    /**
     * Initializes exits randomly for the population.
     * Randomly places 'nbExit' exits within the simulation dimensions.
     * The positions are picked from empty cells on the map, and no two exits
     * are placed at the same position.
     *
     * @param nbExit Number of exits in the simulation.
     * @param simulationDim Dimensions of the simulation grid (width, height).
     * @param mapElements Map elements representing the state of each cell in the simulation grid.
     */
    void initRandomExits(int nbExit, uint2 simulationDim, std::vector<int> mapElements);

    // Initialize both states and exits randomly for the population
    /**
     * Initializes the population randomly.
     * Randomly assigns positions to 'nbExit' exits and 'nbPopulations' individuals within the simulation dimensions.
     * The individuals' positions are picked from empty cells on the map, and no two individuals
     * are assigned the same position. The exits are also placed on empty cells of the map.
     *
     * @param nbPopulations Number of individuals in the population.
     * @param nbExit Number of exits in the simulation.
     * @param simulationDim Dimensions of the simulation grid (width, height).
     * @param mapElements Map elements representing the state of each cell in the simulation grid.
     */
    void initRandom(int nbPopulations, int nbExit, uint2 simulationDim, std::vector<int> mapElements);

    // Get the name of the population
    /**
     * Gets the name of the population.
     *
     * @return const std::string& A constant reference to the name of the population.
     */
    const std::string &getName() const;

    // Set the name of the population
    /**
     * Sets the name of the population.
     *
     * @param name A constant reference to a string representing the new name of the population.
     */
    void setName(const std::string &name);

    // Get the current states of individuals in the population
    /**
     * Get the current states of individuals in the population.
     *
     * @return const std::vector<int3>& A constant reference to the vector containing the current states of individuals.
     */
    const std::vector<int3> &getStates() const;

    // Set the current states of individuals in the population
    /**
     * Set the states of individuals in the population.
     *
     * @param states A constant reference to a vector containing the new states of individuals to be set.
     */
    void setStates(const std::vector<int3> &states);

    // Get the positions of exits in the simulation
    /**
     * Get the positions of exits in the simulation.
     *
     * @return const std::vector<int2>& A constant reference to the vector containing the positions of exits.
     */
    const std::vector<int2> &getExits() const;

    // Set the positions of exits in the simulation
    /**
     * Set the positions of exits in the simulation.
     *
     * @param exits A constant reference to a vector containing the new positions of exits to be set.
     */
    void setExits(const std::vector<int2> &exits);

    // Get the cost map for navigation
    /**
     * Get the cost map for navigation in the simulation.
     *
     * @return const std::vector<unsigned int>& A constant reference to the vector containing the cost map.
     */
    const std::vector<unsigned int> &getMapCost() const;

    // Set the cost map for navigation
    /**
     * Set the cost map for navigation in the simulation.
     *
     * @param mapCost A constant reference to a vector containing the new cost map to be set.
     */
    void setMapCost(const std::vector<unsigned int> &mapCost);

    // Get the color representing the population in the visualization
    /**
     * Get the color representing the population in the visualization.
     *
     * @return const color& A constant reference to the color representing the population.
     */
    const color &getColor() const;

    // Set the color representing the population in the visualization
    /**
     * Set the color representing the population in the visualization.
     *
     * @param col A constant reference to a color representing the new color to be set.
     */
    void setColor(const color &col);

    // Set the color representing the population in the visualization using RGB values
    /**
     * Set the color representing the population in the visualization using RGB components.
     *
     * @param r The red component of the color (0-255).
     * @param g The green component of the color (0-255).
     * @param b The blue component of the color (0-255).
     */
    void setColor(uint r, uint g, uint b);

    // Print the current states of individuals in the population
    /**
     * Print the list of positions of individuals in the population.
     */
    void printStates() const;

    // Print the positions of exits in the simulation
    /**
     * Print the list of positions of exits in the population.
     */
    void printExits() const;

    // Print the cost map for navigation
    /**
     * Print the cost map of the population.
     *
     * @param dimension The dimensions (width and height) of the cost map.
     */
    void printMapCost(uint2 dimension) const;

    // Print both states and exits of the population
    /**
     * Print information about the population, including states, exits, and the cost map (if available).
     *
     * @param dimension The dimensions (width and height) of the cost map.
     */
    void print(uint2 dimension);

    // Destructor
    ~Population();
};

#endif // POPULATION_HPP
