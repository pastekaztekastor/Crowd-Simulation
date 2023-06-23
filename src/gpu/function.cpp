/*******************************************************************************
* File Name: function.cpp 
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: This file contains all the functions
*******************************************************************************/

// Include necessary libraries here
#include "main.hpp"

// Declare functions and classes here

/*
  _____       _ _ 
 |_   _|     (_) |  
   | |  _ __  _| |_ 
   | | | '_ \| | __|
  _| |_| | | | | |_ 
 |_____|_| |_|_|\__|

*/
void initSimSettings( int argc, char const *argv[], simParam * _simParam, settings * _settings){
    _simParam->populationPosition = nullptr;            
    _simParam->cost               = nullptr;            
    _simParam->map                = nullptr;            
    _simParam->exit               = make_uint2(0, 0) ;  
    _simParam->populationIndex    = nullptr;            
    _simParam->dimension          = make_uint2(0, 0);   
    _simParam->nbIndividual       = 10;                 
    _simParam->pInSim             = _simParam->nbIndividual;            
    _simParam->isFinish           = 0;     
    _simParam->nbFrame            = 0;             
    _settings->print              = 2;                  
    _settings->debugMap           = 0;                  
    _settings->model              = 0;                  
    _settings->exportType         = 0;                  
    _settings->exportFormat       = 0;                  
    _settings->finishCondition    = 0;                  
    _settings->dir                = "bin/";             
    _settings->dirName            = "";                 
    _settings->fileName           = "";    

    if (argc > 1){
        for (size_t i = 1; i < argc; i += 2) {
            if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0) {
                // print help
                printf(" +++++ HELP GPU VERSION +++++\n");

                printf("  -x        : sets the dimension in x of the simulation\n");
                printf("  -y        : same but for y dimension\n");
                printf("  -p        : number of individuals in the simulation\n");
                printf("  -debug    : _settings.print           param [val] default:'normal'\n");
                printf("              - 'off' : print nothing\n");
                printf("              - 'time' : print only time execution\n");
                printf("              - 'normal' : print time execution and simlulation progression\n");
                printf("              - 'all' : print all\n");
                printf("  -debugMap : settings_debugMap        param ['on'/'off'] default:'off'\n");
                printf("  -model    : settings_model           param [num] default:0\n");
                printf("              - 0 : Acctuel\n");
                printf("              - 1 : Rng\n");
                printf("              - 2 : Sage ignorant\n");
                printf("              - 3 : Impatient ingorant\n");
                printf("              - 4 : Forcée\n");
                printf("              - 5 : Cone de vision\n");
                printf("              - 6 : Meilleur cout\n");
                printf("              - 7 : Meilleur cout & déplacement forcé\n");
                printf("  -exptT    : settings_exportType      param ['type'] default:'txt'\n");
                printf("              - 'txt', 'jpeg', 'bin', ...\n");
                printf("  -exptF    : settings_exportFormat    param ['type'] default:'m'\n");
                printf("              - 'm' : export map\n");
                printf("              - 'p' : export position [ONLY 'txt' OR 'bin' exptT]\n");
                printf("              - 'c' : export congestion\n");
                printf("  -fCon     : settings_finishCondition param ['fix'/'empty'] default:empty\n");
                printf("  -dir      : settings_dir             param ['dir'] default:'bin/'\n");
                printf("              - Chose a custom directory path to export\n");
                printf("  -dirName  : settings_dirName         param ['name'] default:'X-Y-P'\n");
                printf("              - Chose a custom directory name to export\n");
                printf("  -fileName : settings_fileName        param ['std::string'] default:''\n");
                printf("              - Chose a custom file name to export\n");
                printf("  -help -h : help (if so...)\n");
                exit(0);
            } 
            else if (strcmp(argv[i], "-x") == 0) {
                _simParam->dimension.x = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-y") == 0) {
                _simParam->dimension.y = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-p") == 0) {
                _simParam->nbIndividual = atoi(argv[i + 1]);
                _simParam->pInSim  = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-debug") == 0) {
                if (strcmp(argv[i+1], "off") == 0) {
                    _settings->print = 0;
                }
                else if (strcmp(argv[i+1], "time") == 0) {
                    _settings->print = 10;
                }
                else if (strcmp(argv[i+1], "normal") == 0) {
                    _settings->print = 20;
                }
                else if (strcmp(argv[i+1], "all") == 0) {
                    _settings->print = 30;
                }
                else {
                    printf("Unrecognized argument for -debug param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-debugMap") == 0) {
                if (strcmp(argv[i+1], "off") == 0) {
                    _settings->debugMap = 0;
                }
                else if (strcmp(argv[i+1], "on") == 0) {
                    _settings->debugMap = 1;
                }
                else {
                    printf("Unrecognized argument for -debugMap param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-model") == 0) {
                _settings->model = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-exptT") == 0) {
                if (strcmp(argv[i+1], "txt") == 0) {
                    _settings->exportType = 0;
                }
                else if (strcmp(argv[i+1], "bin") == 0) {
                    _settings->exportType = 1;
                }
                else if (strcmp(argv[i+1], "jpeg") == 0) {
                    _settings->exportType = 2;
                }
                else {
                    printf("Unrecognized argument for -exptT param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-exptF") == 0) {
                if (strcmp(argv[i+1], "m") == 0) {
                    _settings->exportFormat = 0;
                }
                else if (strcmp(argv[i+1], "p") == 0) {
                    _settings->exportFormat = 1;
                }
                else if (strcmp(argv[i+1], "c") == 0) {
                    _settings->exportFormat = 2;
                }
                else {
                    printf("Unrecognized argument for -exptF param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-fCon") == 0) {
                if (strcmp(argv[i+1], "fix") == 0) {
                    _settings->finishCondition = 0;
                }
                else if (strcmp(argv[i+1], "empty") == 0) {
                    _settings->finishCondition = 1;
                }
                else {
                    printf("Unrecognized argument for -fCon param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-dir") == 0) {
                _settings->dir = argv[i+1];
            }
            else if (strcmp(argv[i], "-dirName") == 0) {
                _settings->dirName = argv[i+1];
            }
            else if (strcmp(argv[i], "-fileName") == 0) {
                _settings->fileName = argv[i+1];
            }
            else{
                printf("Unrecognized argument, try -h or -help\n");
                exit(0);
            }
        }  
    }
    if(_simParam->nbIndividual > _simParam->dimension.x * _simParam->dimension.y * 0.8){
        std::cout << "The number of individuals exceeds 80\% of the simulation space. Terrain generation will not be compleor efficient. Please decrease the individual quantity or increase the dimension.";
        exit(0);
    }

    // Calloc
    _simParam->populationPosition = ( int2 * ) calloc(_simParam->nbIndividual, sizeof( int2 ));
    _simParam->map = ( int * ) calloc(_simParam->dimension.x * _simParam->dimension.y , sizeof( int ));
    for (size_t i = 0; i < _simParam->dimension.x * _simParam->dimension.y; i++){
        _simParam->map[i] = -1; // -1 for empty 
    }
    _simParam->populationIndex = ( uint * ) calloc(_simParam->nbIndividual * 2, sizeof( uint));
    _simParam->exit = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));
    
    
    if(_settings->print > 1) std::cout << " *** WELCOME TO CROWD SIMULATION ON CUDA *** " << std::endl 
        << "\t # simDimX = " << _simParam->dimension.x <<std::endl
        << "\t # simDimY = " << _simParam->dimension.y <<std::endl
        << "\t # simDimP = " << _simParam->nbIndividual <<std::endl
        << "\t # _settings.print = " << _settings->print <<std::endl
        << "\t # settings_debugMap = " << _settings->debugMap <<std::endl
        << "\t # settings_model = " << _settings->model <<std::endl
        << "\t # settings_exportType = " << _settings->exportType <<std::endl
        << "\t # settings_exportFormat = " << _settings->exportFormat <<std::endl
        << "\t # settings_finishCondition = " << _settings->finishCondition <<std::endl
        << "\t # settings_dir = " << _settings->dir <<std::endl
        << "\t # settings_dirName = " << _settings->dirName <<std::endl
        << "\t # settings_fileName = " << _settings->fileName <<std::endl <<std::endl;
}
void initPopulationPositionMap(simParam * _simParam, settings _settings){
    if(_settings.print >2) std::cout << "\t# initPopulationPositionMap  " << std::endl;
    uint2 coord = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));
    
    // -2) Placing the walls
        // if(_settings.print >2)std::cout << "   --> Placing the walls  ";
        // TO DO - Currently we do not put
        // if(_settings.print >2)std::cout << "-> DONE " << std::endl;

    // -3) Placing exit
    if(_settings.print >2) std::cout << "\t -> Placing element"<< std::endl;
    if(_settings.print >2) std::cout << "\t --> Wall " ;
    _simParam->map[valueOfxy(_simParam->exit.x,_simParam->exit.y,_simParam->dimension.x, _simParam->dimension.y)] = -2;
    if(_settings.print >2) std::cout << "\tOK " << std::endl ;
    
    // -4) Place individuals only if it is free.
    if(_settings.print >2) std::cout << "\t --> People " ;
    for (size_t i = 0; i < _simParam->nbIndividual; i++){
        coord = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));

        while (_simParam->map[valueOfxy(coord.x,coord.y,_simParam->dimension.x,_simParam->dimension.y)] != -1){
            coord = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));
        }
        // ---- population
        _simParam->populationPosition[i] = make_int2(coord.x, coord.y) ;
        // ---- map
        _simParam->map[valueOfxy(coord.x,coord.y,_simParam->dimension.x,_simParam->dimension.y)] = i;
    }
    // printPopulationPosition((*_simParam), _settings);

    if(_settings.print >2)std::cout << "\tOK " << std::endl ;
}
void initKernelParam(kernelParam * _kernelParam, simParam _simParam, settings _settings){
    if( _settings.print > 2 )std::cout << std::endl << " ### Init kernel params ###" << std::endl;
    if( _settings.print > 2 )std::cout  << " \t> dev_ variable";
    _kernelParam->populationPosition  = nullptr;
    _kernelParam->map                 = nullptr;
    _kernelParam->simPIn              = nullptr;
    if( _settings.print > 2 )std::cout  << " OK " << std::endl;

    if( _settings.print > 2 )std::cout << " \t> Maloc";
    cudaMalloc((void**) &_kernelParam->populationPosition , 2 * sizeof(uint) * _simParam.nbIndividual); 
    cudaMalloc((void**) &_kernelParam->map                , _simParam.dimension.x * _simParam.dimension.y * sizeof(uint ));
    cudaMalloc((void**) &_kernelParam->simPIn             , sizeof(uint));

    cudaMemcpy(_kernelParam->populationPosition, _simParam.populationPosition, (2 * sizeof(uint) * _simParam.nbIndividual)         , cudaMemcpyHostToDevice);
    cudaMemcpy(_kernelParam->map               , _simParam.map               , (_simParam.dimension.x * _simParam.dimension.y * sizeof(uint))  , cudaMemcpyHostToDevice);
    if( _settings.print > 2 )std::cout  << " OK " << std::endl;

    if( _settings.print > 2 )std::cout << " \t> Threads & blocks" ;
    _kernelParam->nb_threads = 32;
    _kernelParam->blocks = ((_simParam.nbIndividual + (_kernelParam->nb_threads-1))/_kernelParam->nb_threads);
    _kernelParam->threads = (_kernelParam->nb_threads);
    if( _settings.print > 2 )std::cout  << " OK " << std::endl;
}

/*
   _____      _   _            
  / ____|    | | | |           
 | (___   ___| |_| |_ ___ _ __ 
  \___ \ / _ \ __| __/ _ \ '__|
  ____) |  __/ |_| ||  __/ |   
 |_____/ \___|\__|\__\___|_|   
                               
*/
void setSimExit(simParam * _simParam, settings _settings){
    // TO DO
}
void setPopulationPositionMap(simParam * _simParam, settings _settings){
    // TO DO
}

/*
   _____ _                 _       _   _             
  / ____(_)               | |     | | (_)            
 | (___  _ _ __ ___  _   _| | __ _| |_ _  ___  _ __  
  \___ \| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \ 
  ____) | | | | | | | |_| | | (_| | |_| | (_) | | | |
 |_____/|_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_|
                                                     
*/
void progressBar(uint progress, uint total, uint width, uint iteration) {
    // From ChatGPT
    float percentage = (float)progress / total;
    uint filledWidth = ( uint)(percentage * width);
    printf("[");
    for ( uint i = 0; i < width; i++) {
        if (i < filledWidth) {
            printf("=");
        } else {
            printf(" ");
        }
    }
    printf("] %.2f%% : %d frames\r", percentage * 100, iteration);
    fflush(stdout);
}
void shuffleIndex(simParam * _simParam, settings _settings){
    for ( uint i = _simParam->nbIndividual - 1; i > 0; i--) {
       uint j = rand() % (i + 1);
       uint temp = _simParam->populationIndex[i];
       _simParam->populationIndex[i] = _simParam->populationIndex[j];
       _simParam->populationIndex[j] = temp;
    }
}

/*
  ______             
 |  ____|            
 | |__ _ __ ___  ___ 
 |  __| '__/ _ \/ _ \
 | |  | | |  __/  __/
 |_|  |_|  \___|\___|
                     
*/
void freeSimParam (simParam * _simParam, settings _settings){
    if(_settings.print >2)std::cout << "\t# freeTab  " ;
    free(_simParam->populationPosition);
    free(_simParam->populationIndex);
    free(_simParam->cost);
    free(_simParam->map);
    if(_settings.print >2)std::cout << "\tOK " << std::endl;
}

/*
  _    _ _   _ _     
 | |  | | | (_) |    
 | |  | | |_ _| |___ 
 | |  | | __| | / __|
 | |__| | |_| | \__ \
  \____/ \__|_|_|___/
                     
*/
void printMap(simParam _simParam, settings _settings){
    if(_settings.print > 2)std::cout << " # - Display map --- "<<std::endl;

    // Display column numbers
    std::cout<<"  ";
        for (int x = 0; x < _simParam.dimension.x; x++)
        {
            printf(" %2d",x); 
        }
        std::cout<<"  "<<std::endl;

    // We browse the map and we display according to what the box contains
    for (int y = 0; y < _simParam.dimension.y; y++)
    {
        printf("%2d ",y); 
        for (int x = 0; x < _simParam.dimension.x; x++)
        {
            switch (_simParam.map[valueOfxy(x,y,_simParam.dimension.x,_simParam.dimension.y)])
            {
            case -3:
                std::cout<<"///";
                break;
            case -2:
                std::cout<<"(s)";
                break;
            case -1:
                std::cout<<" . ";
                break;
            default:
                std::cout<<"[H]";
                break;
            }
        }
        std::cout<<std::endl;
    }
    if(_settings.print < 2)std::cout << "                         --- DONE " << std::endl;
}
void printPopulationPosition(simParam _simParam, settings _settings){
    std::cout << std::endl << "\t# printPopulationPosition"<< std::endl << "\t   Id \t x \t y " <<std::endl;
    for (size_t i = 0; i < _simParam.nbIndividual; i++)
    {
        std::cout << "\t  - " << i << " )\t" << _simParam.populationPosition[i].x << "\t" << _simParam.populationPosition[i].y <<std::endl;
    }
    std::cout << "\t OK" <<std::endl; 
}
uint xPosof(uint value, uint dimX, uint dimY) {
    return value % dimX;
}
uint yPosof(uint value, uint dimX, uint dimY) {
    return value / dimX;
}
uint valueOfxy(uint xPos, uint yPos, uint dimX, uint dimY) {
    return yPos * dimX + xPos;
}