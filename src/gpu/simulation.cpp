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
    _settings->print              = 2;                  
    _settings->debugMap           = 0;                  
    _settings->model              = 0;                  
    _settings->exportType         = 0;                  
    _settings->exportFormat       = 0;                  
    _settings->finishCondition    = 0;                  
    _settings->dir                = "exe/bin";             
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
                printf("  -fileName : settings_fileName        param ['string'] default:''\n");
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
        cout << "The number of individuals exceeds 80\% of the simulation space. Terrain generation will not be compleor efficient. Please decrease the individual quantity or increase the dimension.";
        exit(0);
    }

    // Concaténer les valeurs des variables en chaînes de caractères
    string dimensionXStr = to_string(_simParam->dimension.x);
    string dimensionYStr = to_string(_simParam->dimension.y);
    string nbIndividualStr = to_string(_simParam->nbIndividual);

    // Concaténer les chaînes pour former le nom du dossier
    _settings->dirName = "X" + dimensionXStr + "_Y" + dimensionYStr + "_P" + nbIndividualStr;

    // Calloc
    _simParam->populationPosition = ( int2 * ) calloc(_simParam->nbIndividual, sizeof( int2 ));
    _simParam->map = ( int * ) calloc(_simParam->dimension.x * _simParam->dimension.y , sizeof( int ));
    for (size_t i = 0; i < _simParam->dimension.x * _simParam->dimension.y; i++){
        _simParam->map[i] = __MAP_EMPTY__ ; // -1 for empty 
    }
    _simParam->populationIndex = ( uint * ) calloc(_simParam->nbIndividual * 2, sizeof( uint));
    _simParam->exit = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));
    
    
    if(_settings->print > 1) cout << " *** WELCOME TO CROWD SIMULATION ON CUDA *** " << endl 
        << "\t # simDimX                    = " << _simParam->dimension.x <<endl
        << "\t # simDimY                    = " << _simParam->dimension.y <<endl
        << "\t # simDimP                    = " << _simParam->nbIndividual <<endl
        << "\t # _settings.print            = " << _settings->print <<endl
        << "\t # settings_debugMap          = " << _settings->debugMap <<endl
        << "\t # settings_model             = " << _settings->model <<endl
        << "\t # settings_exportType        = " << _settings->exportType <<endl
        << "\t # settings_exportFormat      = " << _settings->exportFormat <<endl
        << "\t # settings_finishCondition   = " << _settings->finishCondition <<endl
        << "\t # settings_dir               = " << _settings->dir <<endl
        << "\t # settings_dirName           = " << _settings->dirName <<endl
        << "\t # settings_fileName          = " << _settings->fileName <<endl <<endl;
}
void initPopulationPositionMap(simParam * _simParam, settings _settings){
    if(_settings.print >2) cout << "\t# initPopulationPositionMap  " << endl;
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
        _simParam->populationPosition[i] = make_int2(coord.x, coord.y) ;
        // ---- map
        _simParam->map[valueOfxy(coord.x,coord.y,_simParam->dimension.x,_simParam->dimension.y)] = i;
    }
    // printPopulationPosition((*_simParam), _settings);

    if(_settings.print >2)cout << "\tOK " << endl ;
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
void exportSimParam2Json(simParam _simParam, settings _settings) {
    // Création du chemin complet du fichier JSON
    string filePath = _settings.dir + "/" + _settings.dirName + "/start.json";

    // Création d'un objet JSON
    json jsonData;

    // Ajout des valeurs de la structure _simParam à l'objet JSON
    jsonData["dimension"]["x"] = _simParam.dimension.x;
    jsonData["dimension"]["y"] = _simParam.dimension.y;
    jsonData["nbIndividual"] = _simParam.nbIndividual;
    jsonData["pInSim"] = _simParam.pInSim;
    jsonData["isFinish"] = _simParam.isFinish;
    jsonData["nbFrame"] = _simParam.nbFrame;

    // Conversion de l'objet JSON en une chaîne de caractères
    string jsonString = jsonData.dump();

    // Création du fichier JSON et écriture des données
    ofstream file(filePath);
    if (file.is_open()) {
        file << jsonString;
        file.close();
        cout << "Exportation des paramètres de simulation au format JSON terminée avec succès." << endl;
    } else {
        cerr << "Erreur lors de la création du fichier JSON." << endl;
    }
}
*/ 

void exportPopulationPosition2HDF5(simParam _simParam, settings _settings) {
    // Création du nom du fichier
    string filePath = _settings.dir + "/" + _settings.dirName + "/";
    string fileName = filePath + to_string(_simParam.nbFrame) + ".h5";
    // Vérifier si le répertoire existe et le créer si nécessaire
    if (mkdir(_settings.dir.c_str(), 0777) != 0) {
        if (mkdir(filePath.c_str(), 0777) != 0) {
            // Création du fichier HDF5
            hid_t file = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            if (file < 0) {
                cerr << "Erreur lors de la création du fichier HDF5." << endl;
                return;
            }

            // Création de l'espace de données pour les positions
            hsize_t dims[2] = {_simParam.nbIndividual, 2}; // Dimensions du tableau de positions
            hid_t dataspace = H5Screate_simple(2, dims, NULL);

            // Création du dataset pour les positions
            hid_t dataset = H5Dcreate(file, "populationPosition", H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (dataset < 0) {
                cerr << "Erreur lors de la création du dataset HDF5." << endl;
                H5Sclose(dataspace);
                H5Fclose(file);
                return;
            }

            // Écriture des positions dans le dataset
            herr_t status = H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, _simParam.populationPosition);
            if (status < 0) {
                cerr << "Erreur lors de l'écriture des positions dans le dataset HDF5." << endl;
            }

            // Fermeture des ressources HDF5
            H5Dclose(dataset);
            H5Sclose(dataspace);
            H5Fclose(file);
        }
    }
    
    if(_settings.print >2) cout << "Exportation des positions terminée avec succès." << endl;
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
    if(_settings.print >2)cout << "\t# freeTab  " ;
    free(_simParam->populationPosition);
    free(_simParam->populationIndex);
    free(_simParam->cost);
    free(_simParam->map);
    if(_settings.print >2)cout << "\tOK " << endl;
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
    if(_settings.print > 2)cout << " # - Display map --- "<<endl;

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
    if(_settings.print < 2)cout << "                         --- DONE " << endl;
}
void printPopulationPosition(simParam _simParam, settings _settings){
    cout << endl << "\t# printPopulationPosition"<< endl << "\t   Id \t x \t y " <<endl;
    for (size_t i = 0; i < _simParam.nbIndividual; i++)
    {
        cout << "\t  - " << i << " )\t" << _simParam.populationPosition[i].x << "\t" << _simParam.populationPosition[i].y <<endl;
    }
    cout << "\t OK" <<endl; 
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

