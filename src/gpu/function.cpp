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

// Start implementing the code here

void simParamInit(int argc, 
        char const *argv[],
        int * simDimX, // Because change
        int * simDimY, // Because change
        int * simDimP, // Because change
        int * simDimG, // Because change
        int * settings_print, // Because change
        int * settings_debugMap, // Because change
        int * settings_model, // Because change
        int * settings_exportType, // Because change
        int * settings_exportFormat, // Because change
        int * settings_finishCondition, // Because change
        string * settings_dir, // Because change
        string * settings_dirName, // Because change
        string * settings_fileName // Because change
    ){
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
                printf("              - 'all' : print all\n");
                printf("              - 'time' : print only time execution\n");
                printf("              - 'normal' : print time execution and simlulation progression\n");
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
                printf("  -fileName : settings_fileName        param ['string'] default:''\n");
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
        cout << "The number of individuals exceeds 80\% of the simulation space. Terrain generation will not be compleor efficient. Please decrease the individual quantity or increase the dimensions.";
        exit(0);
    }
}

void  _CPU_shuffleIndex(int **index, int _nbIndividual){
    if(_debug == 1)cout << " # - Mix of indexes --- ";

    for (size_t i = 0; i < _nbIndividual; i++)
    {
        int b = rand() % (_nbIndividual-1);
        // cout<<"invert "<<i<<" "<<b<<endl;
        int tmp = (*index)[i];
        (*index)[i] = (*index)[b];
        (*index)[b] = tmp;  
    }

    if(_debug == 1)cout << " DONE " << endl;
}
int * _CPU_shifting(_Element *** map, float*** population, int individue, int * exitSimulation){
    if((*population)[individue][0] == -1.f && (*population)[individue][1] == -1.f){
        return nullptr;
    }
    if(_debug == 1)cout << " # - Population displacement --- ";

    // - step 1) determine what is the displacement vector
    float posX = (*population)[individue][0];
    float posY = (*population)[individue][1];

    float deltaX = (exitSimulation[0] - posX);
    float deltaY = (exitSimulation[1] - posY);

    // - step 2) find if the neighbor which is known the trajectory of the moving vector is free
    float moveX = deltaX / max(abs(deltaX), abs(deltaY));
    float moveY = deltaY / max(abs(deltaX), abs(deltaY));
    if(_debug == 1)cout << "[" << (*population)[individue][0] << "," << (*population)[individue][1] << "] + [" << moveX << "," << moveY << "]";

    int otherSideX = (int) (rand()% 3 )-1;
    int otherSideY = (int) (rand()% 3 )-1;
    // - step 3) Displacement according to the different scenarios
    switch ((*map)[(int)(posY+moveY)][(int)(posX+moveX)])
    {
    case (HUMAN):
        // For the moment we don't deal with this scenario.
        break;

    case (WALL):
        // When we encounter a wall the individual begins by looking at another box around him and if it is free he goes there.

        if ((*map)[(int)(posY+otherSideY)][(int)(posX+otherSideX)] == EMPTY){
            // Moving the individual in the list of people
            (*population)[individue][0] = posX+otherSideX;
            (*population)[individue][1] = posY+otherSideY;
            //Change on the map. We set the old position to empty and we pass the new one to occupied
            (*map)[(int) posY][(int) posX] = EMPTY;
            (*map)[(int)(posY+otherSideY)][(int)(posX+otherSideX)] = HUMAN;
        } 
        break;

    case (EXIT):
        // We remove the individual from the table of people and we stop displaying it

        /* We have 2 possibilities:
         *  -1) either we consider that the individual has for ID the index of the array and therefore we are obliged to keep all the individuals out. (we put their population at -1,-1)
         *  -2) either we add an ID in addition to the x and y dimensions to the table that contains the population, and in this case we can get rid of people who have left the simulation.
        break;
        */  
       
        // -1)
        // Moving the individual in the list of people
        (*population)[individue][0] = -1.f;
        (*population)[individue][1] = -1.f;
        //Change on the map. We set the old position to empty
        (*map)[(int) posY][(int) posX] = EMPTY;
        break;

    case (EMPTY):
        // Moving the individual in the list of people
        (*population)[individue][0] = posX+moveX;
        (*population)[individue][1] = posY+moveY;
        //Change on the map. We set the old position to empty and we pass the new one to occupied
        (*map)[(int) posY][(int) posX] = EMPTY;
        (*map)[(int)(posY+moveY)][(int)(posX+moveX)] = HUMAN;
        break;
    
    default:
        // For the moment we don't deal with this scenario. gozmyg-3suthy-tywmAj
        break;
    }

    if(_debug == 1)cout << " DONE " << endl;
    return nullptr;
}
void  binFrame(float** population, int * exitSimulation, string dir, int _xParam, int _yParam, int _nbIndividual, int generationAcc){
    if(_debug == 1)cout << " # - Saving data to a Json file --- ";
    FILE* F; 
    char fileName[100]; // X000-Y000-P000(000000).bin
    char directory[100];
    sprintf( directory, "%sX%03d-Y%03d-P%03d",dir.c_str(), _xParam, _yParam, _nbIndividual);
    sprintf(fileName, "%s/%06d.bin",directory, generationAcc);

    mkdir( directory, 0755 );
    F = fopen(fileName,"wb");

    for (size_t i = 0; i < _nbIndividual; i++){
        fwrite(&population[i][0],sizeof(int),1,F);
        fwrite(&population[i][1],sizeof(int),1,F);
    }
    fclose(F);
    if(_debug == 1)cout << " DONE " << endl;
}
void  printMap(_Element ** map, int _xParam, int _yParam){
    if(_debug == 1)cout << " # - Display map --- "<<endl;

    // Display column numbers
    cout<<"  ";
        for (int x = 0; x < _xParam; x++)
        {
            printf(" %2d",x); 
        }
        cout<<"  "<<endl;

    // We browse the map and we display according to what the box contains
    for (int y = 0; y < _yParam; y++)
    {
        printf("%2d ",y); 
        for (int x = 0; x < _xParam; x++)
        {
            switch (map[y][x])
            {
            case HUMAN:
                cout<<"[H]";
                break;
            case WALL:
                cout<<"[ ]";
                break;
            case EXIT:
                cout<<"(s)";
                break;

            default:
            case EMPTY:
                cout<<" . ";
                break;
            }
        }
        cout<<endl;
    }
    if(_debug == 1)cout << "                         --- DONE " << endl;
}
void  printPopulation(float** population, int _nbIndividual){
    if(_debug == 1)cout << " # - Population display --- " << endl;
    for (size_t i = 0; i < _nbIndividual; i++){
        cout<<"Creation of individual "<< i <<" on "<< _nbIndividual <<" - position: ["<<population[i][0]<<","<<population[i][1]<<"]"<<endl; // For debuging 
    }
    if(_debug == 1)cout << "                        --- DONE " << endl;
}
int   signeOf(float a){
    if (a<0){
        return -1;
    }
    else{
        return 1;
    }
}
void  _CPU_generateSimulation(_Element *** map, float*** population, int _nbIndividual, int _xParam, int _yParam, int ** exitSimulation){

    if(_debug == 1)cout << " # - Generate simulation  " << endl;
    int x = (rand() % _xParam);
    int y = (rand() % _yParam);
    
    // -1) Make memory allocations
    if(_debug == 1)cout << "   --> Make memory allocations ";
    // ---- population
    (*population) = (float ** ) malloc(_nbIndividual * sizeof(float*));
    for (size_t i = 0; i < _nbIndividual; i++) {
        (*population)[i] = (float * ) malloc(2 * sizeof(float));
    }
    // ---- map
    (*map) = (_Element ** ) malloc(_yParam * sizeof(_Element * ));
    for (size_t y = 0; y < _yParam; y++) {
        (*map)[y] = (_Element * ) malloc(_xParam * sizeof(_Element));
        for (size_t x = 0; x < _xParam; x++){
            (*map)[y][x] = EMPTY;
        }
    }
    if(_debug == 1)cout << "-> DONE " << endl;
    
    // -2) Placing the walls
        // if(_debug == 1)cout << "   --> Placing the walls  ";
        // TO DO - Currently we do not put
        // if(_debug == 1)cout << "-> DONE " << endl;
    
    // -3) Place the output
    if(_debug == 1)cout << "   --> Place the output  " ;

    while ((*map)[y][x] != EMPTY){
        if(_debug == 1) cout << (*map)[y][x] << " / ";
        x = (rand() % _xParam);
        y = (rand() % _yParam);
    }
    // ---- exitSimulation
    (*exitSimulation)[0] = x;
    (*exitSimulation)[1] = y;
    // ---- map
    (*map)[y][x] = EXIT;
    if(_debug == 1)cout << "-> DONE " << endl;
    
    // -4) Place individuals only if it is free.
    if(_debug == 1)cout << "   --> Place individuals only if it is free  " ;
    for (size_t i = 0; i < _nbIndividual; i++)
    {
        x = (rand() % _xParam);
        y = (rand() % _yParam);

        while ((*map)[y][x] != EMPTY){
            if(_debug == 1) cout << (*map)[y][x] << " / ";
            x = (rand() % _xParam);
            y = (rand() % _yParam);
        }
        // ---- population
        (*population)[i][0] = x;
        (*population)[i][1] = y;
        // ---- map
        (*map)[y][x] = HUMAN;
    }
    if(_debug == 1)cout << "-> DONE " << endl;

    if(_debug == 1)cout << "   --> DONE  " << endl;
}