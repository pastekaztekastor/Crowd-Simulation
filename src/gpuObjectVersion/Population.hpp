#include "utils/utils.hpp"

class Population

{
private:
    std::string name;
    std::vector<int3> etats;
    std::vector<int2> exits;
    std::vector<uint> mapCost;
    cv::Scalar(255, 255, 255) color;

public:
    Population(std::string new_name);
    
    void initRandomEtats(int nbPopulations, uint2 simulationDim, vector<int> mapElements);
    void initRandomExits(int nbExit, uint2 simulationDim, vector<int> mapElements);

    void initRandom(int nbPopulations, int nbExit, uint2 simulationDim, vector<int> mapElements);

    void setName(std::string name) const;

    int2 getPosOf(int index) const;
    int  getWaitOf(int index) const;
    int  getNbPopulations() const;

    void printEtats();
    void printExits();
    void printMapCost();
    void print();

    ~Population();
};