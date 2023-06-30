/*******************************************************************************
* File Name: function.cpp 
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: This file contains all the functions
*******************************************************************************/

// Include necessary libraries here
#include "simulation.hpp"

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
    _simParam->dimension          = make_uint2(5, 5);   
    _simParam->nbIndividual       = 10;                 
    _simParam->pInSim             = _simParam->nbIndividual;            
    _simParam->isFinish           = 0;     
    _simParam->nbFrame            = 0;   

    _settings->print              = __DEBUG_PRINT_ALL__;  
    _settings->model              = 0;                  
    _settings->exportDataType     = __EXPORT_TYPE_GIF__;
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
                printf("  -debug    : _settings.print           param [val] default:'step'\n");
                printf("              - 'none' : print nothing\n");
                printf("              - 'step' : print only time execution\n");
                printf("              - 'all' : print time execution and simlulation progression\n");
                printf("              - 'debug' : print all\n");
                printf("  -model    : settings_model           param [num] default:0\n");
                printf("              - 0 : Acctuel\n");
                printf("              - 1 : Rng\n");
                printf("              - 2 : Sage ignorant\n");
                printf("              - 3 : Impatient ingorant\n");
                printf("              - 4 : Forcée\n");
                printf("              - 5 : Cone de vision\n");
                printf("              - 6 : Meilleur cout\n");
                printf("              - 7 : Meilleur cout & déplacement forcé\n");
                printf("  -exptT    : settings_exportDataType      param ['type'] default:'gif'\n");
                printf("              - gif\n");
                printf("              - value\n");
                printf("              - all\n");
                printf("  -dir      : settings_dir             param ['dir'] default:'bin/'\n");
                printf("              - Chose a custom directory path to exportData\n");
                printf("  -dirName  : settings_dirName         param ['name'] default:'X-Y-P'\n");
                printf("              - Chose a custom directory name to exportData\n");
                printf("  -fileName : settings_fileName        param ['string'] default:''\n");
                printf("              - Chose a custom file name to exportData\n");
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
                if (strcmp(argv[i+1], "none") == 0) {
                    _settings->print = __DEBUG_PRINT_NONE__;
                }
                else if (strcmp(argv[i+1], "step") == 0) {
                    _settings->print = __DEBUG_PRINT_STEP__;
                }
                else if (strcmp(argv[i+1], "all") == 0) {
                    _settings->print = __DEBUG_PRINT_ALL__;
                }
                else if (strcmp(argv[i+1], "debug") == 0) {
                    _settings->print = __DEBUG_PRINT_DEBUG__;
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
                if (strcmp(argv[i+1], "gif") == 0) {
                    _settings->exportDataType = __EXPORT_TYPE_GIF__;
                }
                else if (strcmp(argv[i+1], "value") == 0) {
                    _settings->exportDataType = __EXPORT_TYPE_VALUE__;
                }
                else if (strcmp(argv[i+1], "all") == 0) {
                    _settings->exportDataType = __EXPORT_TYPE_ALL__;
                }
                else {
                    printf("Unrecognized argument for -exptT param\n");
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
        cout << "The number of individuals exceeds 80\% of the simulation space. Terrain generation will not be compleor efficient. Please decrease the individual quantity or increase the dimension.";
        exit(0);
    }

    // Regarde si la population n'entraine pas trop de frame
    // TO DO

    // Concaténer les valeurs des variables en chaînes de caractères
    string dimensionXStr = to_string(_simParam->dimension.x);
    string dimensionYStr = to_string(_simParam->dimension.y);
    string nbIndividualStr = to_string(_simParam->nbIndividual);

    // Concaténer les chaînes pour former le nom du dossier
    _settings->dirName = "X" + dimensionXStr + "_Y" + dimensionYStr + "_P" + nbIndividualStr;

    // Calloc
    _simParam->populationPosition = ( float3 * ) calloc(_simParam->nbIndividual, sizeof( float3 ));
    _simParam->map = ( int * ) calloc(_simParam->dimension.x * _simParam->dimension.y , sizeof( int ));
    for (size_t i = 0; i < _simParam->dimension.x * _simParam->dimension.y; i++){
        _simParam->map[i] = __MAP_EMPTY__ ; // -1 for empty 
    }
    _simParam->populationIndex = ( uint * ) calloc(_simParam->nbIndividual * 2, sizeof( uint));
    _simParam->exit = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));
    
    
    if(_settings->print >= __DEBUG_PRINT_ALL__) cout << " *** WELCOME TO CROWD SIMULATION ON CUDA *** " << endl 
        << "\t # simDimX = " << _simParam->dimension.x <<endl
        << "\t # simDimY = " << _simParam->dimension.y <<endl
        << "\t # simDimP = " << _simParam->nbIndividual <<endl
        << "\t # _settings.print = " << _settings->print <<endl
        << "\t # settings_model = " << _settings->model <<endl
        << "\t # settings_exportDataType = " << _settings->exportDataType <<endl
        << "\t # settings_dir = " << _settings->dir <<endl
        << "\t # settings_dirName = " << _settings->dirName <<endl
        << "\t # settings_fileName = " << _settings->fileName <<endl <<endl;
    if(_settings->print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}
void initPopulationPositionMap(simParam * _simParam, settings _settings){
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - initPopulationPositionMap ---"<<endl;
    uint2 coord = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));
    
    // -2) Placing the walls
        // if(_settings.print >2)cout << "   --> Placing the walls  ";
        // TO DO - Currently we do not put
        // if(_settings.print >2)cout << "-> DONE " << endl;

    // -3) Placing exit
    if(_settings.print >2) cout << "\t -> Placing element"<< endl;
    if(_settings.print >2) cout << "\t --> Wall " ;
    _simParam->map[valueOfxy(_simParam->exit.x,_simParam->exit.y,_simParam->dimension.x, _simParam->dimension.y)] = -2;
    if(_settings.print >2) cout << "\tOK " << endl ;
    
    // -4) Place individuals only if it is free.
    if(_settings.print >2) cout << "\t --> People " ;
    for (size_t i = 0; i < _simParam->nbIndividual; i++){
        coord = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));

        while (_simParam->map[valueOfxy(coord.x,coord.y,_simParam->dimension.x,_simParam->dimension.y)] != -1){
            coord = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));
        }
        // ---- population
        _simParam->populationPosition[i] = make_float3(coord.x, coord.y,0.f) ;
        // ---- map
        _simParam->map[valueOfxy(coord.x,coord.y,_simParam->dimension.x,_simParam->dimension.y)] = i;
    }
    // printPopulationPosition((*_simParam), _settings);

    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}
void initExportData(simParam _simParam, exportData * _exportData, settings _settings){
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - initExportData ---"<<endl;
    // gifFrames
    _exportData->gifOutputFilename  = "animation" + _settings.dirName + ".gif";
    _exportData->gifSizeFactor      = 1;
    if (_simParam.dimension.x < __MAX_X_DIM_JPEG__ && _simParam.dimension.y <__MAX_Y_DIM_JPEG__){ 
        _exportData->gifSizeFactor = min((__MAX_X_DIM_JPEG__ / _simParam.dimension.x), (__MAX_Y_DIM_JPEG__ / _simParam.dimension.y));
    }
    _exportData->gifPath       = _settings.dir + _settings.dirName + "/";
    _exportData->gifRatioFrame = 1 ;// mettre un parametre en géniie log
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
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
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - setSimExit ---"<<endl;
    // TO DO
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}
void setPopulationPositionMap(simParam * _simParam, settings _settings){
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - setPopulationPositionMap ---"<<endl;
    // TO DO
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
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
    printf("] %.2f%% : %d Frames\r", percentage * 100, iteration);
    fflush(stdout);
}
void shuffleIndex(simParam * _simParam, settings _settings){
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - Shuffle index --- "<<endl;
    for ( uint i = _simParam->nbIndividual - 1; i > 0; i--) {
       uint j = rand() % (i + 1);
       uint temp = _simParam->populationIndex[i];
       _simParam->populationIndex[i] = _simParam->populationIndex[j];
       _simParam->populationIndex[j] = temp;
    }
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}

void exportDataFrame(simParam _simParam, exportData * _exportData, settings _settings){
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - exportDataFrame --- "<<endl;
    if(_settings.exportDataFormat == __EXPORT_TYPE_GIF__ || _settings.exportDataFormat == __EXPORT_TYPE_ALL__){
        if(_simParam.nbFrame%_exportData->gifRatioFrame == 0){
            cv::Mat frame(_simParam.dimension.x * _exportData->gifSizeFactor, _simParam.dimension.y * _exportData->gifSizeFactor, CV_8UC3, __COLOR_BLACK__);

            // Dessiner le pixel de sortie en vert
            cv::Point TL(_simParam.exit.x * _exportData->gifSizeFactor, _simParam.exit.y * _exportData->gifSizeFactor);
            cv::Point BR(TL.x + _exportData->gifSizeFactor , TL.y + _exportData->gifSizeFactor);
            cv::Rect rectangle(TL, BR);
            cv::rectangle(frame, rectangle, __COLOR_GREEN__, -1);

            // Dessiner les pixels de la population en blanc
            for (size_t i = 0; i < _simParam.nbIndividual; ++i) {
                cv::Point TL(_simParam.populationPosition[i].x * _exportData->gifSizeFactor, _simParam.populationPosition[i].y * _exportData->gifSizeFactor);
                cv::Point BR(TL.x + _exportData->gifSizeFactor, TL.y + _exportData->gifSizeFactor);
                cv::Rect rectangle(TL, BR);
                cv::rectangle(frame, rectangle, __COLOR_WHITE__, -1);
            }
            _exportData->gifFrames.push_back(frame);
        }
    }
    if(_settings.exportDataFormat == __EXPORT_TYPE_VALUE__ || _settings.exportDataFormat == __EXPORT_TYPE_ALL__){
        // TO DO
    }
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}


void saveExportData(simParam _simParam, exportData _exportData, settings _settings){
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - saveExportData --- "<<endl;
    if(_settings.exportDataFormat == __EXPORT_TYPE_GIF__ || _settings.exportDataFormat == __EXPORT_TYPE_ALL__){
        // Créer un objet VideoWriter pour écrire le fichier GIF
        cv::VideoWriter writer(_exportData.gifOutputFilename, cv::VideoWriter::fourcc('G', 'I', 'F', 'S'), __GIF_FPS__, _exportData.gifFrames[0].size());

        // Vérifier si le VideoWriter a été correctement initialisé
        if (!writer.isOpened()) {
            cout << "Erreur lors de l'ouverture du fichier GIF de sortie" << endl;
        }

        // Écrire chaque image dans le fichier GIF
        for (const cv::Mat& frame : _exportData.gifFrames) {
            writer.write(frame);
        }

        // Fermer le fichier GIF
        writer.release();
        cout << "Le fichier GIF a été créé avec succès." << endl;
    }
    if(_settings.exportDataFormat == __EXPORT_TYPE_VALUE__ || _settings.exportDataFormat == __EXPORT_TYPE_ALL__){
        // TO DO
    }
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
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
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - freeSimParam --- "<<endl;
    free(_simParam->populationPosition);
    free(_simParam->populationIndex);
    free(_simParam->cost);
    free(_simParam->map);
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
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
    if(_settings.print > __DEBUG_PRINT_DEBUG__)cout << " # - Display map --- "<<endl;

    // Display column numbers
    cout<<"  ";
        for (int x = 0; x < _simParam.dimension.x; x++)
        {
            printf(" %2d",x); 
        }
        cout<<"  "<<endl;

    // We browse the map and we display according to what the box contains
    for (int y = 0; y < _simParam.dimension.y; y++)
    {
        printf("%2d ",y); 
        for (int x = 0; x < _simParam.dimension.x; x++)
        {
            switch (_simParam.map[valueOfxy(x,y,_simParam.dimension.x,_simParam.dimension.y)])
            {
            case __MAP_WALL__:
                cout<<"///";
                break;
            case __MAP_EXIT__:
                cout<<"(s)";
                break;
            case __MAP_EMPTY__:
                cout<<" . ";
                break;
            default:
                cout<<"[H]";
                break;
            }
        }
        cout<<endl;
    }
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}
void printPopulationPosition(simParam _simParam, settings _settings){
    if(_settings.print > __DEBUG_PRINT_DEBUG__)cout << " # - printPopulationPosition --- "<<endl;
    for (size_t i = 0; i < _simParam.nbIndividual; i++)
    {
        cout << "\t  - " << i << " )\t" << _simParam.populationPosition[i].x << "\t" << _simParam.populationPosition[i].y <<endl;
    }
    if(_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
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

