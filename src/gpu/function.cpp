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
void initSimParam( int argc, char const *argv[], unsigned int * simDimX, unsigned int * simDimY, unsigned int * simDimP, unsigned int * simPIn, unsigned int * simDimG, unsigned int * settings_print, unsigned int * settings_debugMap, unsigned int * settings_model, unsigned int * settings_exportType, unsigned int * settings_exportFormat, unsigned int * settings_finishCondition,  std::string * settings_dir, std::string * settings_dirName, std::string * settings_fileName){
    if (argc > 1){
        for (size_t i = 1; i < argc; i += 2) {
            if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0) {
                // print help
                printf(" +++++ HELP GPU VERSION +++++\n");

                printf("  -x        : sets the dimension in x of the simulation\n");
                printf("  -y        : same but for y dimension\n");
                printf("  -p        : number of individuals in the simulation\n");
                printf("  -g        : generation number/frame\n");
                printf("  -debug    : settings_print           param [val] default:'normal'\n");
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
                (* simDimX) = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-y") == 0) {
                (* simDimY) = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-p") == 0) {
                (* simDimP) = atoi(argv[i + 1]);
                (* simPIn) = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-g") == 0) {
                (* simDimG) = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-debug") == 0) {
                if (strcmp(argv[i+1], "off") == 0) {
                    (* settings_print) = 0;
                }
                else if (strcmp(argv[i+1], "time") == 0) {
                    (* settings_print) = 10;
                }
                else if (strcmp(argv[i+1], "normal") == 0) {
                    (* settings_print) = 20;
                }
                else if (strcmp(argv[i+1], "all") == 0) {
                    (* settings_print) = 30;
                }
                else {
                    printf("Unrecognized argument for -debug param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-debugMap") == 0) {
                if (strcmp(argv[i+1], "off") == 0) {
                    (* settings_debugMap) = 0;
                }
                else if (strcmp(argv[i+1], "on") == 0) {
                    (* settings_debugMap) = 1;
                }
                else {
                    printf("Unrecognized argument for -debugMap param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-model") == 0) {
                (* settings_model) = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-exptT") == 0) {
                if (strcmp(argv[i+1], "txt") == 0) {
                    (* settings_exportType) = 0;
                }
                else if (strcmp(argv[i+1], "bin") == 0) {
                    (* settings_exportType) = 1;
                }
                else if (strcmp(argv[i+1], "jpeg") == 0) {
                    (* settings_exportType) = 2;
                }
                else {
                    printf("Unrecognized argument for -exptT param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-exptF") == 0) {
                if (strcmp(argv[i+1], "m") == 0) {
                    (* settings_exportFormat) = 0;
                }
                else if (strcmp(argv[i+1], "p") == 0) {
                    (* settings_exportFormat) = 1;
                }
                else if (strcmp(argv[i+1], "c") == 0) {
                    (* settings_exportFormat) = 2;
                }
                else {
                    printf("Unrecognized argument for -exptF param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-fCon") == 0) {
                if (strcmp(argv[i+1], "fix") == 0) {
                    (* settings_finishCondition) = 0;
                }
                else if (strcmp(argv[i+1], "empty") == 0) {
                    (* settings_finishCondition) = 1;
                }
                else {
                    printf("Unrecognized argument for -fCon param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-dir") == 0) {
                (* settings_dir) = argv[i+1];
            }
            else if (strcmp(argv[i], "-dirName") == 0) {
                (* settings_dirName) = argv[i+1];
            }
            else if (strcmp(argv[i], "-fileName") == 0) {
                (* settings_fileName) = argv[i+1];
            }
            else{
                printf("Unrecognized argument, try -h or -help\n");
                exit(0);
            }
        }  
    }
    if((* simDimP) > ((* simDimX)*(* simDimY))*0.8){
        std::cout << "The number of individuals exceeds 80\% of the simulation space. Terrain generation will not be compleor efficient. Please decrease the individual quantity or increase the dimensions.";
        exit(0);
    }
    
    if((*settings_print) > 1) std::cout << " *** WELCOME TO CROWD SIMULATION ON CUDA *** " << std::endl 
        << "\t # simDimX = " << (*simDimX) <<std::endl
        << "\t # simDimY = " << (*simDimY) <<std::endl
        << "\t # simDimG = " << (*simDimG) <<std::endl
        << "\t # simDimP = " << (*simDimP) <<std::endl
        << "\t # settings_print = " << (*settings_print) <<std::endl
        << "\t # settings_debugMap = " << (*settings_debugMap) <<std::endl
        << "\t # settings_model = " << (*settings_model) <<std::endl
        << "\t # settings_exportType = " << (*settings_exportType) <<std::endl
        << "\t # settings_exportFormat = " << (*settings_exportFormat) <<std::endl
        << "\t # settings_finishCondition = " << (*settings_finishCondition) <<std::endl
        << "\t # settings_dir = " << (*settings_dir) <<std::endl
        << "\t # settings_dirName = " << (*settings_dirName) <<std::endl
        << "\t # settings_fileName = " << (*settings_fileName) <<std::endl <<std::endl;
}
void initPopulationPositionMap( unsigned int ** populationPosition, unsigned int ** map, unsigned int * simExit, unsigned int simDimX, unsigned int simDimY, unsigned int simDimP, unsigned int settings_print){
    if(settings_print >2) std::cout << "\t# initPopulationPositionMap  " << std::endl;
    unsigned int x = (rand() % simDimX);
    unsigned int y = (rand() % simDimY);
    
    // -1) Make memory allocations
    if(settings_print >2) std::cout << "\t -> Make memory allocations ";

    // ---- population
    (*populationPosition) = ( unsigned int * ) calloc(simDimP * 2, sizeof( unsigned int));

    // ---- map
    (*map) = (unsigned int * ) calloc(simDimY * simDimX , sizeof(unsigned int * ));

    if(settings_print >2) std::cout << "\tOK " << std::endl ;
    
    // -2) Placing the walls
        // if(settings_print >2)std::cout << "   --> Placing the walls  ";
        // TO DO - Currently we do not put
        // if(settings_print >2)std::cout << "-> DONE " << std::endl;

    // -3) Placing exit
    if(settings_print >2) std::cout << "\t -> Placing element"<< std::endl;
    if(settings_print >2) std::cout << "\t --> Wall " ;
    (*map)[valueOfxy(simExit[1],simExit[0],simDimX, simDimY)] = 3;
    if(settings_print >2) std::cout << "\tOK " << std::endl ;
    
    // -4) Place individuals only if it is free.
    if(settings_print >2) std::cout << "\t --> People " ;
    for (size_t i = 0; i < simDimP; i++)
    {
        x = (rand() % simDimX);
        y = (rand() % simDimY);

        while ((*map)[valueOfxy(x,y,simDimX,simDimY)] != 0){
            x = (rand() % simDimX);
            y = (rand() % simDimY);
        }
        // ---- population
        (*populationPosition)[valueOfxy(i,0,simDimP,2)] = x;
        (*populationPosition)[valueOfxy(i,1,simDimP,2)] = y;
        // ---- map
        (*map)[valueOfxy(x,y,simDimX,simDimY)] = 1;
    }
    if(settings_print >2)std::cout << "\tOK " << std::endl ;
}
void initPopulationIndex( unsigned int ** populationIndex, unsigned int simDimP, unsigned int settings_print){
    if(settings_print >2)std::cout << "\t# initPopulationIndex  " ;
    (* populationIndex) = ( unsigned int * ) calloc(simDimP * 2, sizeof( unsigned int));
    if(settings_print >2)std::cout << "\tOK " << std::endl;
}
void initSimExit( unsigned int ** simExit, unsigned int simDimX, unsigned int simDimY, unsigned int settings_print ){
    if(settings_print >2 ) std::cout << "\t# initSimExit  " ;
    (*simExit) = ( unsigned int *) malloc( 2 * sizeof( unsigned int));
    (*simExit)[0] = (rand() % simDimX);
    (*simExit)[1] = (rand() % simDimY);
    if(settings_print >2) std::cout << "\tOK "  << std::endl;
}
void initCost( unsigned int ** cost, unsigned int * map, unsigned int * simExit, unsigned int simDimX, unsigned int simDimY, unsigned int settings_print){
    // TO DO
}

/*
   _____      _   _            
  / ____|    | | | |           
 | (___   ___| |_| |_ ___ _ __ 
  \___ \ / _ \ __| __/ _ \ '__|
  ____) |  __/ |_| ||  __/ |   
 |_____/ \___|\__|\__\___|_|   
                               
*/

void setSimExit( unsigned int ** simExit, unsigned int posX, unsigned int posY, unsigned int settings_print){
    // TO DO
}
void setPopulationPositionMap( unsigned int *** populationPosition, unsigned int *** map, unsigned int * simExit, unsigned int simDimX, unsigned int simDimY, unsigned int settings_print){
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

void progressBar( unsigned int progress, unsigned int total, unsigned int width, unsigned int settings_print) {
    // From ChatGPT
    if(settings_print >2)std::cout << "\t# progressBar  " ;
    float percentage = (float)progress / total;
   unsigned int filledWidth = ( unsigned int)(percentage * width);
    printf("[");
    for ( unsigned int i = 0; i < width; i++) {
        if (i < filledWidth) {
            printf("=");
        } else {
            printf(" ");
        }
    }
    printf("] %.2f%%\r", percentage * 100);
    fflush(stdout);
    if(settings_print >2)std::cout << "\tOK " << std::endl;
}
void shuffleIndex( unsigned int ** PopulationIndex, unsigned int simDimP, unsigned int settings_print){
    if(settings_print >2)std::cout << "\t# shuffleIndex  " ;
    for ( unsigned int i = simDimP - 1; i > 0; i--) {
       unsigned int j = rand() % (i + 1);
       unsigned int *temp = PopulationIndex[i];
        PopulationIndex[i] = PopulationIndex[j];
        PopulationIndex[j] = temp;
    }
    if(settings_print >2)std::cout << "\tOK " << std::endl;
}

/*
  ______             
 |  ____|            
 | |__ _ __ ___  ___ 
 |  __| '__/ _ \/ _ \
 | |  | | |  __/  __/
 |_|  |_|  \___|\___|
                     
*/
void freeTab ( unsigned int ** populationPosition, unsigned int settings_print){
    if(settings_print >2)std::cout << "\t# freeTab  " ;
    free(*populationPosition);
    *populationPosition = NULL;
    if(settings_print >2)std::cout << "\tOK " << std::endl;
}

/*
  _    _ _   _ _     
 | |  | | | (_) |    
 | |  | | |_ _| |___ 
 | |  | | __| | / __|
 | |__| | |_| | \__ \
  \____/ \__|_|_|___/
                     
*/
void printMap(unsigned int * map, unsigned int simDimX, unsigned int simDimY, unsigned int settings_print){
    if(settings_print > 2)std::cout << " # - Display map --- "<<std::endl;

    // Display column numbers
    std::cout<<"  ";
        for (int x = 0; x < simDimX; x++)
        {
            printf(" %2d",x); 
        }
        std::cout<<"  "<<std::endl;

    // We browse the map and we display according to what the box contains
    for (int y = 0; y < simDimY; y++)
    {
        printf("%2d ",y); 
        for (int x = 0; x < simDimX; x++)
        {
            switch (map[valueOfxy(x,y,simDimX,simDimY)])
            {
            case 1:
                std::cout<<"[H]";
                break;
            case 2:
                std::cout<<"[ ]";
                break;
            case 3:
                std::cout<<"(s)";
                break;

            default:
            case 0:
                std::cout<<" . ";
                break;
            }
        }
        std::cout<<std::endl;
    }
    if(settings_print < 2)std::cout << "                         --- DONE " << std::endl;
}

unsigned int xPosof(unsigned int value, unsigned int dimX, unsigned int dimY) {
    return value % dimX;
}
unsigned int yPosof(unsigned int value, unsigned int dimX, unsigned int dimY) {
    return value / dimX;
}
unsigned int valueOfxy(unsigned int xPos, unsigned int yPos, unsigned int dimX, unsigned int dimY) {
    return yPos * dimX + xPos;
}

