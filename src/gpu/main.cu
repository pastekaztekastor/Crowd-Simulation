/*******************************************************************************
* File Name: main.cpp
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: main file of the crowd simulation with parallelization on GPU. Contains only the main program.
*******************************************************************************/

// Include necessary libraries here
#include "kernel.hpp"

int main(int argc, char const *argv[])
{
    uint2 *         populationPosition       = nullptr;             // Position table of all the individuals in the simulation [[x,y],[x,y],...]
    uint  *         cost                     = nullptr;             //
    int   *         map                      = nullptr;             // 2D map composed of the uint 0: empty, 1 humain, 2 wall, 3 exit on 1 line
    uint2           simExit                  = make_uint2(0, 0) ;   // [x,y] coordinate of simulation output
    uint  *         populationIndex          = nullptr;             // List of individual indexes so as not to disturb the order of the individuals
    uint2           simDim                   = make_uint2(0, 0);    // Simulation x-dimension
    uint            simDimG                  = 10;                  // Number of generation that the program will do before stopping the simulation
    uint            simDimP                  = 10;                  // Number of individuals who must evolve during the simulation
    uint            simPIn                   = simDimP;             //
    uint            simIsFinish              = 0;                   //
    uint            settings_print           = 2;                   // For display the debug [lvl to print]
    uint            settings_debugMap        = 0;                   // For display map
    uint            settings_model           = 0;                   //
    uint            settings_exportType      = 0;                   //
    uint            settings_exportFormat    = 0;                   //
    uint            settings_finishCondition = 0;                   //
    std::string     settings_dir             = "bin/";              // For chose the directory to exporte bin files
    std::string     settings_dirName         = "";                  //
    std::string     settings_fileName        = "";                  //

    initSimParam(
        argc,
        argv,
        &simDim,
        &simDimP,
        &simPIn,
        &simDimG,
        &settings_print,
        &settings_debugMap,
        &settings_model,
        &settings_exportType,
        &settings_exportFormat,
        &settings_finishCondition,
        &settings_dir,
        &settings_dirName,
        &settings_fileName
    );

    if( settings_print > 2 )std::cout  << " ### Init simulation ###" << std::endl;
    srand(time(NULL));
    initSimExit( &simExit, simDim, settings_print );
    initPopulationPositionMap( &populationPosition, &map, simExit, simDim, simDimP, settings_print );
    initPopulationIndex( &populationIndex, simDimP, settings_print );
    initCost( &cost, map, simExit, simDim, settings_print );

    printMap(map, simDim, settings_print);

    if( settings_print > 2 )std::cout << std::endl << " ### Init kernel params ###" << std::endl;
    if( settings_print > 2 )std::cout  << " \t> dev_ variable";
    uint **     dev_populationPosition  = nullptr;
    uint **     dev_map                 = nullptr;
    uint *      dev_simPIn              = nullptr;
    //uint *      dev_outMove             = nullptr;
    //uint        outMove                 = 0;
    if( settings_print > 2 )std::cout  << " OK " << std::endl;

    if( settings_print > 2 )std::cout << " \t> Maloc";
    cudaMalloc((void**) &dev_populationPosition , 2 * sizeof(uint) * simDimP); 
    cudaMalloc((void**) &dev_map                , simDim.x * simDim.y * sizeof(uint ));
    cudaMalloc((void**) &dev_simPIn             , sizeof(uint));
    //cudaMalloc((void**) dev_outMove            , sizeof(uint));
    cudaMemcpy(dev_populationPosition, populationPosition, (2 * sizeof(uint) * simDimP)         , cudaMemcpyHostToDevice);
    cudaMemcpy(dev_map               , map               , (simDim.x * simDim.y * sizeof(uint ))  , cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_simPIn            , simPIn            , sizeof(uint)                         , cudaMemcpyHostToDevice);
    if( settings_print > 2 )std::cout  << " OK " << std::endl;

    if( settings_print > 2 )std::cout << " \t> Threads & blocks" ;
    uint nb_threads = 32;
    dim3 blocks((simDimP + (nb_threads-1))/nb_threads);
    dim3 threads(nb_threads);
    if( settings_print > 2 )std::cout  << " OK " << std::endl;
    
    if( settings_print > 2 )std::cout << std::endl << " ### Start simulation ###" << std::endl;
    while (simIsFinish == 0){
        if (simPIn == 0) simIsFinish = 1; 
        
        //progressBar(simDimP - simPIn, simDimP, 100, 0);
        shuffleIndex(&populationIndex, simDimP, 0);
        
        // MODEL
        switch (settings_model){
            case 0: // MODEL : sage ignorant
                // TO DO
                simPIn--;

                // kernel_model1_GPU<<<blocks,threads>>>(dev_populationPosition, dev_map, dev_simPIn, cost, simExit, simDim, simDimP);
                
                for (size_t tid = 0; tid < simDimP; tid++)
                {
                    // position de l'individue tid
                    uint2 pos    = make_uint2((populationPosition)[tid].x, (populationPosition)[tid].y);
                    uint2 delta  = make_uint2(simExit.x-pos.x, simExit.y-pos.y);
                    uint  maxDim = max(abs(delta.x), abs(delta.y));
                    uint2 move   = make_uint2(delta.x / maxDim, delta.y / maxDim);
                    std::cout <<"c "<<pos.x<<" "<<pos.y<<"\td "<<delta.x<<" "<<delta.y<<"\tm "<<move.x<<" "<<move.y;

                    // on regarde si la case est disponible 
                    if((map[ simDim.x * (pos.y+move.y) + (pos.x + move.x)]) == -1){ // if is EMPTY
                        std::cout <<"-> moove" << std::endl;
                        (populationPosition)[tid] = make_uint2(pos.x + move.x, pos.y + move.y);
                    
    
                        // Temporaire
                        (map)[simDim.x * (pos.y+move.y) + (pos.x + move.x)]    = tid;
                        (map)[simDim.x * pos.y + pos.x]                        = -1;
                        //atomicExch( (map)[y+deltaY][x+deltaX]  , 1); // HUMAN
                        //atomicExch( (map)[y][x]                , 0); // EMPTY
                    }
                    else std::cout << std::endl;
                }
                break;

            case 1: // MDOEL : Impatient ignorant
            case 2: // MDOEL : Forcée
            case 3: // MDOEL : Conne de vision
            case 4: // MDOEL : Meilleur coût
            case 5: // MDOEL : Meilleur déplacement 
            default:
                simPIn--;
                break;
        }
        //printMap(map, simDim, settings_print);
        // EXPORT 
        switch (settings_model){
            case 1:
                // TO DO  
                break;
            
            default:
                break;
        }
    }
    
    if( settings_print > 2 )std::cout << " \t> Cuda Copy ";
    //cudaMemcpy(outMove           , dev_outMove           , sizeof(uint)                      , cudaMemcpyDeviceToHost);
    cudaMemcpy(populationPosition, dev_populationPosition, 2 * sizeof(uint) * simDimP        , cudaMemcpyDeviceToHost);
    cudaMemcpy(map               , dev_map               , simDim.x * simDim.y * sizeof(uint ) , cudaMemcpyDeviceToHost);
    if( settings_print > 2 )std::cout  << " OK " << std::endl;

    std::cout << std::endl;
    printMap(map, simDim, settings_print);

    if( settings_print > 2 )std::cout  << std::endl  << std::endl << "### Memory free ###" << std::endl;
    cudaFree(dev_populationPosition);
    cudaFree(dev_map);
    cudaFree(dev_simPIn);
    // freeTab(&populationPosition, settings_print);
    // freeTab(&populationIndex, settings_print); // passer a des uint2
    // freeCost (&cost, simDim.y, settings_print);
    // freeTab(&map, settings_print);

    return 0;
}
