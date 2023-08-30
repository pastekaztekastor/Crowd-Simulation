/*******************************************************************************
// Fichier : Population.cpp
// Description : Fichier de définition de la classe Population qui représente une population dans la simulation de foule.
// Auteur : Mathurin Champémont
// Date : 06/07/2023
*******************************************************************************/

#include "Population.hpp"

// Default constructor for the Population class
Population::Population()
{
    // Set the probability of displacement for each individual to the default values
    // The values below indicate equal probabilities in all directions (up, down, left, right, etc.)
    this->pMovement = getMatrixMove();
    initDirections();

    // Set the name of the population to a default value "NO_NAME"
    this->name = "NO_NAME";

    // Generate a random color to represent the population in the visualization
    // The color is represented using RGB values, with each component ranging from 0 to 255
    this->pcolor = {(uint)(rand() % 255), (uint)(rand() % 255), (uint)(rand() % 255)};
}

// Constructor that initializes a Population object based on density and exits images.
// The provided image files are loaded to populate the wallPositions and map.
Population::Population(std::string filePathDensity, std::string filePathExits) {
    // Load density image
    srand(time(NULL));
    this->pMovement = getMatrixMove();
    initDirections();
    cv::Mat imageDensity = cv::imread(filePathDensity, cv::IMREAD_COLOR);
    if (imageDensity.empty()) {
        std::cout << "Error importing image: " << filePathDensity << std::endl;
    }

    // Load exits image
    cv::Mat imageExits = cv::imread(filePathExits, cv::IMREAD_COLOR);
    if (imageExits.empty()) {
        std::cout << "Error importing image: " << filePathExits << std::endl;
    }

    // Iterate through the images to populate wallPositions and map
    for (int y = 0; y < imageDensity.rows; y++) {
        for (int x = 0; x < imageDensity.cols; x++) {
            cv::Vec3b pixelDensity = imageDensity.at<cv::Vec3b>(y, x);
            cv::Vec3b pixelExit = imageExits.at<cv::Vec3b>(y, x);

            cv::Scalar color(pixelExit[0], pixelExit[1], pixelExit[2]);
            float p = (float)pixelDensity[1]/(float)255;

            if (color == __COLOR_GREEN__) {
                this->exits.push_back({x, y});
            }
            int randomValue = rand();
            float r = (float )randomValue / RAND_MAX;

            if (r < p) {
                this->states.push_back({x, y, 0});
            }
        }
    }
    pcolor = {(uint)rand()%150, (uint)rand()%150, (uint)rand()/150};
}

// Constructor for the Population class with a specified name
Population::Population(std::string name)
{
    // Set the probability of displacement for each individual to the default values
    // The values below indicate equal probabilities in all directions (up, down, left, right, etc.)
    this->pMovement = getMatrixMove();
    initDirections();

    // Set the name of the population to the specified 'name'
    this->name = name;

    // Generate a random color to represent the population in the visualization
    // The color is represented using RGB values, with each component ranging from 0 to 255
    this->pcolor = {(uint)(rand() % 255), (uint)(rand() % 255), (uint)(rand() % 255)};
}

// Constructor for the Population class with specified parameters to initialize the population
Population::Population(std::string name, int nbPopulations, int nbExit, uint2 simulationDim, std::vector<int> mapElements)
{
    // Set the probability of displacement for each individual to the default values
    // The values below indicate equal probabilities in all directions (up, down, left, right, etc.)
    this->pMovement = getMatrixMove();
    initDirections();

    // Set the name of the population to the specified 'name'
    this->name = name;

    // Generate a random color to represent the population in the visualization
    // The color is represented using RGB values, with each component ranging from 0 to 255
    this->pcolor = {(uint)(rand() % 255), (uint)(rand() % 255), (uint)(rand() % 255)};

    // Initialize the population randomly based on the given parameters
    Population::initRandom(nbPopulations, nbExit, simulationDim, mapElements);
}

// Initialize states randomly for the population
void Population::initRandomStates(int nbPopulations, uint2 simulationDim, std::vector<int> mapElements)
{
    std::cout << "init Random States" << std::endl;

    // Iterate 'nbPopulations' times to assign random states to individuals in the population
    for (size_t i = 0; i < nbPopulations; i++)
    {
        int3 coord;
        bool isPicked = false;

        // Repeat until a valid random coordinate is found that corresponds to an empty cell on the map and is not already picked
        do
        {
            isPicked = false;
            // Generate a random coordinate within the simulation dimensions (0 to simulationDim.x-1 and 0 to simulationDim.y-1)
            coord = {(int)(rand() % simulationDim.x), (int)(rand() % simulationDim.y), 0};

            // Check if the generated coordinate is already picked by another individual
            for (auto &&etat : this->states)
            {
                if (etat.x == coord.x && etat.y == coord.y)
                {
                    // If the coordinate is already picked, set 'isPicked' to true and break the loop
                    isPicked = true;
                    break;
                }
            }

            // Check if the generated coordinate corresponds to an empty cell on the map
        } while ((mapElements[coord.y * simulationDim.x + coord.x] != __MAP_EMPTY__) || isPicked);

        // Add the random coordinate to the states vector, representing an individual's position
        this->states.push_back(coord);
    }

    std::cout << "end" << std::endl;
}

// Initializes exits randomly for the population.
void Population::initRandomExits(int nbExit, uint2 simulationDim, std::vector<int> mapElements)
{
    for (size_t i = 0; i < nbExit; i++)
    {
        int2 coord;
        bool isPicked = false;

        // Repeat until a valid random coordinate is found that corresponds to an empty cell on the map and is not already picked
        do
        {
            // Generate a random coordinate within the simulation dimensions (0 to simulationDim.x-1 and 0 to simulationDim.y-1)
            coord = {(int)(rand() % simulationDim.x), (int)(rand() % simulationDim.y)};

            // Check if the generated coordinate is already picked by another exit
            for (auto &&exit : this->exits)
            {
                if (exit.x == coord.x && exit.y == coord.y)
                {
                    // If the coordinate is already picked, set 'isPicked' to true and break the loop
                    isPicked = true;
                    break;
                }
            }

            // Check if the generated coordinate corresponds to an empty cell on the map
        } while ((mapElements[coord.y * simulationDim.x + coord.y] != __MAP_EMPTY__) || isPicked);

        // Add the random coordinate to the exits vector, representing an exit's position
        this->exits.push_back(coord);
    }
}


// Initializes the population randomly.
void Population::initRandom(int nbPopulations, int nbExit, uint2 simulationDim, std::vector<int> mapElements)
{
    // Initialize exits randomly
    Population::initRandomExits(nbExit, simulationDim, mapElements);

    // Initialize states randomly for the population
    Population::initRandomStates(nbPopulations, simulationDim, mapElements);
}

// Gets the name of the population.
const std::string &Population::getName() const {
    return name;
}
// Sets the name of the population.
void Population::setName(const std::string &name)
{
    Population::name = name;
}
// Get the current states of individuals in the population
const std::vector<int3> &Population::getStates() const {
    return states;
}

// Set the states of individuals in the population.
void Population::setStates(const std::vector<int3> &states) {
    Population::states = states;
}

// Get the positions of exits in the simulation.
const std::vector<int2> &Population::getExits() const {
    return exits;
}

// Set the positions of exits in the simulation.
void Population::setExits(const std::vector<int2> &exits) {
    Population::exits = exits;
}

// Get the cost map for navigation in the simulation.
const std::vector<unsigned int> Population::getMapCost() const {
    return mapCost;
}

// Set the cost map for navigation in the simulation.
void Population::setMapCost(const std::vector<unsigned int> &mapCost) {
    Population::mapCost = mapCost;
}

// Get the color representing the population in the visualization.
const color &Population::getColor() const {
    return pcolor;
}

// Returns a constant reference to the vector of possible movement directions for the population.
const std::vector<std::pair<int, int>>& Population::getDirections() const {
    return directions;
}

// Sets the vector of possible movement directions for the population.
void Population::setDirections(const std::vector<std::pair<int, int>>& directions) {
    Population::directions = directions;
}


// Set the color representing the population in the visualization.
void Population::setColor(const color &col) {
    Population::pcolor = col;
}

// Set the color representing the population in the visualization using RGB components.
void Population::setColor(uint r, uint g, uint b) {
    this->pcolor = {r, g, b};
}

// Returns a constant reference to the probability of displacement for each individual in the population.
std::vector<float> Population::getPMovement() const {
    return pMovement;
}

// Sets the probability of displacement for each individual in the population.
void Population::setPMovement(const std::vector<float> &pMovement) {
    Population::pMovement = pMovement;
    initDirections();
}

// Print the list of positions of individuals in the population.
void Population::printStates() const {
    // Print the header with the name of the population
    std::cout << "List of positions of individuals in the population: " << this->name << std::endl;

    // Iterate through each coordinate in the 'states' vector
    for (auto && coord : this->states)
    {
        // Print the x, y, and z components of the coordinate in a formatted manner
        std::cout << "  [" << coord.x << "," << coord.y << "] " << coord.z << std::endl;
    }
}


// Print the list of positions of exits in the population.
void Population::printExits() const {
    // Print the header with the name of the population
    std::cout << "List of positions of exits in the population: " << this->name << std::endl;

    // Iterate through each coordinate in the 'exits' vector
    for (auto && coord : this->exits)
    {
        // Print the x and y components of the coordinate in a formatted manner
        std::cout << "  [" << coord.x << "," << coord.y << "]" << std::endl;
    }
}

// Print the cost map of the population.
void Population::printMapCost(uint2 dimension) const {
    // Print the header with the name of the population
    std::cout << "Cost map of the population: " << this->name << std::endl;

    // Print the top row of x-axis labels
    std::cout << "   ";
    for (int x = 0; x < dimension.x; x++)
    {
        printf("%3d ", x);
    }
    std::cout << "  " << std::endl;

    // Iterate through each row (y-axis) in the cost map
    for (int y = 0; y < dimension.y; y++)
    {
        // Print the y-axis label for the current row
        printf("%3d ", y);

        // Iterate through each column (x-axis) in the cost map
        for (int x = 0; x < dimension.x; x++)
        {
            // Print the cost value at the current position (x, y) in a formatted manner
            printf("%3d ", this->mapCost[y * dimension.x + x]);
        }

        // Move to the next line after printing all values in the row
        std::cout << std::endl;
    }
}

// Print information about the population, including states, exits, and the cost map (if available).
void Population::print(uint2 dimension) {
    // Print the header with the name of the population
    std::cout << "POPULATION: " << this->name << std::endl;

    // Print the list of positions of individuals in the population
    Population::printStates();

    // Print the list of positions of exits in the population
    Population::printExits();

    // Check if the cost map is available and print it
    if (this->mapCost.size() > 0)
        Population::printMapCost(dimension);

    // Print an empty line for better readability
    std::cout << std::endl;
}

// Removes a state from the population's states vector based on the provided parameters.
void Population::removeStat(const int3 &param) {
    for (size_t i = 0; i < states.size(); ++i) {
        if (states[i].x == param.x && states[i].y == param.y) {
            states.erase(states.begin() + i);
            break;  // Once the element is found and erased, exit the loop
        }
    }
}


Population::~Population()
{
}

// Initializes the vector of possible movement directions based on the pMovement values.
void Population::initDirections() {
    int sizeMatrice = static_cast<int>(std::sqrt(pMovement.size()));
    int midleMat = (sizeMatrice - 1) / 2;
    directions.clear();

    for (size_t i = 0; i < pMovement.size(); i++) {
        if (pMovement[i] > 0) {
            directions.push_back({static_cast<int>(i % sizeMatrice) - midleMat, static_cast<int>(i / sizeMatrice) - midleMat});
        }
    }
}

cv::Scalar Population::getColorScalar() const {
    return cv::Scalar(pcolor.b, pcolor.g, pcolor.r);;
}

