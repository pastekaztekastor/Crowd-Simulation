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
    _settings->model              = 1;                  
    _settings->exportDataType     = __EXPORT_TYPE_VIDEO__;
    _settings->dir                = "video/";
    _settings->inputMapPath       = "";

    if (argc > 1){
        for (size_t i = 1; i < argc; i += 2) {
            if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0) {
                // print help
                printf(" +++++ HELP GPU VERSION +++++\n");

                printf("  -x        : sets the dimension in x of the simulation\n");
                printf("  -y        : same but for y dimension\n");
                printf("  -p        : number of individuals in the simulation\n");
                printf("  -print    : _settings.print           param [val] default:'step'\n");
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
                printf("  -exptT    : settings_exportDataType      param ['type'] default:'video'\n");
                printf("              - video\n");
                printf("              - value\n");
                printf("              - all\n");
                printf("  -dir      : settings_dir             param ['dir'] default:'video/'\n");
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
            else if (strcmp(argv[i], "-w") == 0) {
                _simParam->nbWall = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-p") == 0) {
                _simParam->nbIndividual = atoi(argv[i + 1]);
                _simParam->pInSim  = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-print") == 0) {
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
                if (strcmp(argv[i+1], "video") == 0) {
                    _settings->exportDataType = __EXPORT_TYPE_VIDEO__;
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
            else if (strcmp(argv[i], "-input") == 0) {
                _settings->inputMapPath = string(argv[i+1]);
            }
            else{
                printf("Unrecognized argument, try -h or -help\n");
                exit(0);
            }
        }  
    }
    if (_simParam->nbIndividual > _simParam->dimension.x * _simParam->dimension.y * 0.8){
        cout << "The number of individuals exceeds 80\% of the simulation space. Terrain generation will not be compleor efficient. Please decrease the individual quantity or increase the dimension.";
        exit(0);
    }

    // Regarde si la population n'entraine pas trop de frame
    // TO DO

    // Concaténer les valeurs des variables en chaînes de caractères
    string dimensionXStr = to_string(_simParam->dimension.x);
    string dimensionYStr = to_string(_simParam->dimension.y);
    string nbIndividualStr = to_string(_simParam->nbIndividual);

    if (_settings->print >= __DEBUG_PRINT_ALL__) cout << " *** WELCOME TO CROWD SIMULATION ON CUDA *** " << endl 
        << "\t # simDimX = " << _simParam->dimension.x <<endl
        << "\t # simDimY = " << _simParam->dimension.y <<endl
        << "\t # simDimP = " << _simParam->nbIndividual <<endl
        << "\t # settings.print = " << _settings->print <<endl
        << "\t # settings_model = " << _settings->model <<endl
        << "\t # settings_exportDataType = " << _settings->exportDataType <<endl
        << "\t # settings_dir = " << _settings->dir <<endl
        << "\t # settings_inputMapPath = " << _settings->inputMapPath <<endl <<endl;
    if (_settings->print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}
void initPopulationPositionMap(simParam * _simParam, settings _settings){
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - initPopulationPositionMap ---"<<endl;
    
    // Calloc
    _simParam->populationPosition = ( float3 * ) calloc(_simParam->nbIndividual, sizeof( float3 ));
    _simParam->wallPosition = ( uint2 * ) calloc(_simParam->nbIndividual, sizeof( uint2 ));
    _simParam->map = ( int * ) calloc(_simParam->dimension.x * _simParam->dimension.y , sizeof( int ));
    for (size_t i = 0; i < _simParam->dimension.x * _simParam->dimension.y; i++){
        _simParam->map[i] = __MAP_EMPTY__ ; // -1 for empty 
    }
    _simParam->cost = ( uint * ) calloc(_simParam->dimension.x * _simParam->dimension.y , sizeof( uint ));
    _simParam->populationIndex = ( uint * ) calloc(_simParam->nbIndividual * 2, sizeof( uint));
    _simParam->exit = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));
    
    uint2 coord = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));
    
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\t -> Placing element"<< endl;
    // -2) Placing the walls
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\t --> Wall " ;
    for (size_t i = 0; i < _simParam->nbWall; i++)
    {
        coord = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));
        while (_simParam->map[valueOfxy(coord.x,coord.y,_simParam->dimension.x,_simParam->dimension.y)] != __MAP_EMPTY__){
            coord = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));
        }
        _simParam->wallPosition[i] = make_uint2(coord.x, coord.y) ;
        _simParam->map[valueOfxy(coord.x,coord.y,_simParam->dimension.x,_simParam->dimension.y)] = __MAP_WALL__;
    }
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\tOK " << endl ;

    // -3) Placing exit
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\t --> Exit " ;
    _simParam->map[valueOfxy(_simParam->exit.x,_simParam->exit.y,_simParam->dimension.x, _simParam->dimension.y)] = __MAP_EXIT__;
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\tOK " << endl ;
    
    // -4) Place individuals only if it is free.
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\t --> People " ;
    for (size_t i = 0; i < _simParam->nbIndividual; i++){
        coord = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));

        while (_simParam->map[valueOfxy(coord.x,coord.y,_simParam->dimension.x,_simParam->dimension.y)] != __MAP_EMPTY__){
            coord = make_uint2((rand() % _simParam->dimension.x),(rand() % _simParam->dimension.y));
        }
        // ---- population
        _simParam->populationPosition[i] = make_float3(coord.x, coord.y,0.f) ;
        // ---- map
        _simParam->map[valueOfxy(coord.x,coord.y,_simParam->dimension.x,_simParam->dimension.y)] = i;
    }
    // printPopulationPosition((*_simParam), _settings);

    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}
void initExportData(simParam _simParam, exportData * _exportData, settings _settings){
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - initExportData ---"<<endl;
    // videoFrames
    _exportData->videoFilename = "anim_M" + to_string(_settings.model) + "-X" + to_string(_simParam.dimension.x) + "-Y" + to_string(_simParam.dimension.y) + "-P" + to_string(_simParam.nbIndividual) + ".mp4";
    _exportData->videoPath = _settings.dir + _exportData->videoFilename;
    _exportData->videoNbFrame = 0;
    _exportData->videoSizeFactor = 1;
    if (_simParam.dimension.x < __MAX_X_DIM_JPEG__ && _simParam.dimension.y <__MAX_Y_DIM_JPEG__){ 
        _exportData->videoSizeFactor = min((__MAX_X_DIM_JPEG__ / _simParam.dimension.x), (__MAX_Y_DIM_JPEG__ / _simParam.dimension.y));
    }
    _exportData->videoRatioFrame = __VIDEO_RATIO_FRAME__ ;// mettre un parametre en génie log
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;

    // création de l'arboraissance et du nom du fichier 
    struct stat info;
    if (stat(_settings.dir.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
        if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "Le dossier existe déjà." << endl;
    } else {
        // Le dossier n'existe pas, on le crée
        int status = mkdir(_settings.dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (status == 0) {
            if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "Le dossier a été créé avec succès." << endl;
        } else {
            if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "Erreur lors de la création du dossier." << endl;
        }
    }
    if ( max(_simParam.dimension.x, _simParam.dimension.y) > 50 ){
        _exportData->videoCalcCostPlot = __VIDEO_CALC_COST_PLOT_OFF__;
    }
    else{
        _exportData->videoCalcCostPlot = __VIDEO_CALC_COST_PLOT_ON__;
    }
    
    _exportData->videoCalcCostPlot = __VIDEO_CALC_COST_PLOT_OFF__;
    if ( _exportData->videoCalcCostPlot = __VIDEO_CALC_COST_PLOT_ON__ ){
        _exportData->videoCalcCost = cv::Mat(_simParam.dimension.y * _exportData->videoSizeFactor, _simParam.dimension.x * _exportData->videoSizeFactor, CV_8UC3, __COLOR_ALPHA__);
        for (size_t i = 0; i < _simParam.dimension.x * _simParam.dimension.y; i++){
            // Paramètres du texte
            string texte = to_string(_simParam.cost[i]);
            cv::Point position((xPosof(i, _simParam.dimension.x, _simParam.dimension.y ) * _exportData->videoSizeFactor) + (_exportData->videoSizeFactor * 0,9), (yPosof(i, _simParam.dimension.x, _simParam.dimension.y) * _exportData->videoSizeFactor) + (_exportData->videoSizeFactor * 0.9));
            int epaisseur = 1;
            float taillePolice = 0.8;
            int ligneType = cv::LINE_AA;
        
            // Écrire le texte sur l'image
            cv::putText(_exportData->videoCalcCost, texte, position, cv::FONT_HERSHEY_SIMPLEX, taillePolice, __COLOR_GREY__, epaisseur, ligneType);
        }
    }
}

void initCostMap (simParam * _simParam, settings _settings) {
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - initCostMap ---"<<endl;
    // Remplire la carte de coût avec des valeur élevé.
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   - Remplire la carte de coût avec des valeur élevé. "<<endl;
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "     - Taille de la carte de couts "<<  endl;
    for (size_t i = 0; i < _simParam->dimension.x * _simParam->dimension.y; i++){
        _simParam->cost[i] = _simParam->dimension.x*_simParam->dimension.x;
    }
    // if (_settings.print >= __DEBUG_PRINT_DEBUG__)printCostMap((*_simParam), _settings);
    
    // Définir la case de sortie avec un coût de 0
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   - Définir la case de sortie avec un coût de 0 "<<endl;
    _simParam->cost[valueOfxy(_simParam->exit.x, _simParam->exit.y, _simParam->dimension.x, _simParam->dimension.y)] = 0;

    // Définir les déplacements possibles (haut, bas, gauche, droite)
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   - Définir les déplacements possibles (haut, bas, gauche, droite) "<<endl;
    vector<pair<int, int>> directions = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

    // Effectuer l'inondation jusqu'à ce que la carte de coût soit remplie
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   - Effectuer l'inondation jusqu'à ce que la carte de coût soit remplie "<<endl;
    bool updated;
    do {
        updated = false;

        // Parcourir chaque case de la carte
        for (uint y = 0; y < _simParam->dimension.y; ++y) {
            for (uint x = 0; x < _simParam->dimension.x; ++x) { // Cols
                // Vérifier si la case actuelle n'est pas un mur
                if (_simParam->map[valueOfxy(x, y, _simParam->dimension.x, _simParam->dimension.y)] != __MAP_WALL__) {
                    int currentCost = _simParam->cost[valueOfxy(x, y, _simParam->dimension.x, _simParam->dimension.y)];

                    // Explorer les déplacements possibles à partir du point actuel
                    for (const auto& direction : directions) {
                        uint2 newPos = make_uint2(x + direction.first, y + direction.second);

                        // Vérifier si les nouvelles coordonnées sont valides et si ce n'est pas un mure
                        if (newPos.x >= 0 && newPos.x < _simParam->dimension.x && newPos.y >= 0 && newPos.y < _simParam->dimension.y && _simParam->map[valueOfxy(newPos.x, newPos.y, _simParam->dimension.x, _simParam->dimension.y)] != __MAP_WALL__) {
                            int newCost = currentCost + 1;

                            // Mettre à jour le coût si nécessaire
                            if (newCost < _simParam->cost[valueOfxy(newPos.x, newPos.y, _simParam->dimension.x, _simParam->dimension.y)]) {
                                _simParam->cost[valueOfxy(newPos.x, newPos.y, _simParam->dimension.x, _simParam->dimension.y)] = newCost;
                                updated = true;
                            }
                        }
                    }
                }
            }
        }
    } while (updated);
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
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
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - setSimExit ---"<<endl;
    // TO DO
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}
void setPopulationPositionMap(simParam * _simParam, settings _settings){
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - setPopulationPositionMap ---"<<endl;
    // TO DO
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}
void importMap (simParam * _simParam, settings _settings){
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - importMap ---"<<endl;

    // on Réinitialise tout les paramètre qui touche a la simulaiton ou l'export mais qui en dépendent
    _simParam->populationPosition = nullptr;
    _simParam->wallPosition       = nullptr;
    _simParam->cost               = nullptr;
    _simParam->map                = nullptr;
    _simParam->exit               = make_uint2(0,0);
    _simParam->populationIndex    = nullptr;
    _simParam->dimension          = make_uint2(0,0);
    _simParam->nbIndividual       = 0;
    _simParam->nbWall             = 0;
    _simParam->pInSim             = 0;
    _simParam->isFinish           = 0;
    _simParam->nbFrame            = 0;

    // On charge l'image
    cv::Mat image = cv::imread(_settings.inputMapPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        cout << "Erreur lors de l'importation de l'image :" << _settings.inputMapPath << endl;
        return;
    }
    // On la parcour une première fois pour en connaitre la topologie
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            cv::Scalar color(pixel[0], pixel[1], pixel[2]);
            if ( color == __COLOR_BLUE__ ){
                _simParam->nbWall ++;
            }
            if ( color == __COLOR_WHITE__ ){
                _simParam->nbIndividual ++;
            }
        }
    }

    // on attribut les valeur de _simParam en fonction de l'image chargé
    _simParam->dimension = make_uint2(image.cols, image.rows);
    _simParam->populationPosition = ( float3 * ) calloc(_simParam->nbIndividual, sizeof( float3 ));
    _simParam->wallPosition = ( uint2 * ) calloc(_simParam->nbIndividual, sizeof( uint2 ));
    _simParam->cost = ( uint * ) calloc(_simParam->dimension.x * _simParam->dimension.y , sizeof( uint ));
    _simParam->map = ( int * ) calloc(_simParam->dimension.x * _simParam->dimension.y , sizeof( int ));
    for (size_t i = 0; i < _simParam->dimension.x * _simParam->dimension.y; i++){
        _simParam->map[i] = __MAP_EMPTY__ ; // -1 for empty 
    }
    _simParam->populationIndex = ( uint * ) calloc(_simParam->nbIndividual * 2, sizeof( uint));
    _simParam->pInSim = _simParam->nbIndividual;
    //printMap((*_simParam), _settings);

    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   - Nb Wall : "<< _simParam->nbWall << " - Nb Individual : " << _simParam->nbIndividual <<endl;

    // On la parcour une 2iem fois pour remplire nos elements WALL et POPULATION
    int acctualWall = 0;
    int acctualIndividual = 0;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            cv::Scalar color(pixel[0], pixel[1], pixel[2]);

            if ( color == __COLOR_GREEN__ ){
                _simParam->exit = make_uint2(x,y);
            } 
            else if ( color == __COLOR_BLUE__ ){
                _simParam->wallPosition[acctualWall] = (make_uint2(x,y));
                acctualWall ++;
            }
            else if ( color == __COLOR_WHITE__ ){
                _simParam->populationPosition[acctualIndividual] = (make_float3((float)x,(float)y, 0.f));
                acctualIndividual ++;
            }
        }
    }

    // On Remplie la carte 
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\t -> Placing element"<< endl;
    // -4) Place individuals
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\t --> People " ;
    for (size_t i = 0; i < _simParam->nbIndividual; i++){
        _simParam->map[valueOfxy(_simParam->populationPosition[i].x,_simParam->populationPosition[i].y,_simParam->dimension.x,_simParam->dimension.y)] = i;
    }
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\tOK " << endl ;
    // -2) Placing the walls
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\t --> Wall " ;
    for (size_t i = 0; i < _simParam->nbWall; i++){
        _simParam->map[valueOfxy(_simParam->wallPosition[i].x,_simParam->wallPosition[i].y,_simParam->dimension.x,_simParam->dimension.y)] = __MAP_WALL__;
    }
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\tOK " << endl ;
    // -3) Placing exit
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\t --> Exit " ;
    _simParam->map[valueOfxy(_simParam->exit.x,_simParam->exit.y,_simParam->dimension.x, _simParam->dimension.y)] = __MAP_EXIT__;
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "\tOK " << endl ;
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
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
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - Shuffle index --- "<<endl;
    for ( uint i = _simParam->nbIndividual - 1; i > 0; i--) {
       uint j = rand() % (i + 1);
       uint temp = _simParam->populationIndex[i];
       _simParam->populationIndex[i] = _simParam->populationIndex[j];
       _simParam->populationIndex[j] = temp;
    }
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}

void exportDataFrameVideo(simParam _simParam, exportData * _exportData, settings _settings){
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - exportDataFrameVideo --- "<<endl;
    cv::Mat frame(_simParam.dimension.y * _exportData->videoSizeFactor, _simParam.dimension.x * _exportData->videoSizeFactor, CV_8UC3, __COLOR_BLACK__);

    // Dessiner les cercle de la population en blanc / rouge 
    for (size_t i = 0; i < _simParam.nbIndividual; ++i) {
        cv::Point center((_simParam.populationPosition[i].x * _exportData->videoSizeFactor) + (_exportData->videoSizeFactor / 2), (_simParam.populationPosition[i].y * _exportData->videoSizeFactor) + (_exportData->videoSizeFactor / 2));
        float radius = (_exportData->videoSizeFactor / 2) * 0.9;
        cv::Scalar color = colorInterpol(__COLOR_WHITE__, __COLOR_RED__, _simParam.populationPosition[i].z / float(__SIM_MAX_WAITING__));
        int thickness = -1;  // Remplacer par un nombre positif pour un contour solide
        cv::circle(frame, center, radius, color, thickness);
    }
    // Dessiner les mure en bleu
    for (size_t i = 0; i < _simParam.nbWall; ++i) {
        cv::Point TL(_simParam.wallPosition[i].x * _exportData->videoSizeFactor, _simParam.wallPosition[i].y * _exportData->videoSizeFactor);
        cv::Point BR(TL.x + _exportData->videoSizeFactor, TL.y + _exportData->videoSizeFactor);
        cv::Rect rectangle(TL, BR);
        cv::rectangle(frame, rectangle, __COLOR_BLUE__, -1);
    }
    // Dessiner le pixel de sortie en vert
    cv::Point TL(_simParam.exit.x * _exportData->videoSizeFactor, _simParam.exit.y * _exportData->videoSizeFactor);
    cv::Point BR(TL.x + _exportData->videoSizeFactor , TL.y + _exportData->videoSizeFactor);
    cv::Rect rectangle(TL, BR);
    cv::rectangle(frame, rectangle, __COLOR_GREEN__, -1);

    // Superposition avec le calque de cout de chaque case
    if ( 0 ){
        // Paramètre pour la superposition
        double alpha = 0.5; // Facteur de pondération pour l'image 1
        double beta = 0.5;  // Facteur de pondération pour l'image 2
        double gamma = 0.0; // Paramètre d'ajout d'un scalaire

        // Superposer les images
        cv::addWeighted(_exportData->videoCalcCost, alpha, frame, beta, gamma, frame);
    }

    // Exporte la frame
    _exportData->videoFrames.push_back(frame);
    _exportData->videoNbFrame ++;
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}

void exportDataFrameValue(simParam _simParam, exportData * _exportData, settings _settings){
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - exportDataFrameValue --- "<<endl;
    // TO DO
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}


void saveExportDataVideo(simParam _simParam, exportData _exportData, settings _settings){
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - saveExportDataVideo --- "<<endl;
    // Créer un objet VideoWriter pour écrire le fichier MP4
    if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "Création d'un vidéo : " << _exportData.videoPath << ", fps :" << __VIDEO_FPS__ << ", taille : " << _exportData.videoFrames[0].size() << endl;
    _exportData.videoWriter.open(_exportData.videoPath, cv::VideoWriter::fourcc('a','v','c','1'), __VIDEO_FPS__, _exportData.videoFrames[0].size(), true);

    // Vérifier si le VideoWriter a été correctement initialisé
    if (!_exportData.videoWriter.isOpened()) {
        cout << "Erreur lors de l'ouverture du fichier : " << _exportData.videoFilename << " dans (" << _settings.dir << ")" << endl;
    }
    else
    {
        if (_settings.print >= __DEBUG_PRINT_STEP__)cout << " # - Vidéo : " << _exportData.videoPath <<endl;
        // Écrire chaque image dans le fichier MP4
        for (size_t i = 0; i < _exportData.videoNbFrame ; i++){
            _exportData.videoWriter.write(_exportData.videoFrames[i]);
            progressBar(i,_exportData.videoNbFrame-1, 100, i);
        }
        cout << endl;
        // Fermer le fichier MP4
        _exportData.videoWriter.release();
        if (_settings.print >= __DEBUG_PRINT_DEBUG__) cout << "Le fichier video a été créé avec succès." << endl;
    }
}

void saveExportDataValue(simParam _simParam, exportData _exportData, settings _settings){
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - saveExportDataValue --- "<<endl;
        // TO DO
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
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
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - freeSimParam --- "<<endl;
    free(_simParam->populationPosition);
    free(_simParam->populationIndex);
    free(_simParam->cost);
    free(_simParam->map);
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
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
    if (_settings.print > __DEBUG_PRINT_DEBUG__)cout << " # - Display map --- "<<endl;

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
                cout<<" "<< _simParam.map[valueOfxy(x,y,_simParam.dimension.x,_simParam.dimension.y)] <<" ";
                break;
            }
        }
        cout<<endl;
    }
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}

void printCostMap (simParam _simParam, settings _settings){
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << " # - printCostMap --- "<<endl;
    // Display column numbers
    cout<<"  ";
        for (int x = 0; x < _simParam.dimension.x; x++)
        {
            printf("%2d  ",x); 
        }
        cout<<"  "<<endl;

    // We browse the map and we display according to what the box contains
    for (int y = 0; y < _simParam.dimension.y; y++)
    {
        printf("%2d ",y); 
        for (int x = 0; x < _simParam.dimension.x; x++)
        {
            printf(" %2d ", _simParam.cost[valueOfxy(x,y,_simParam.dimension.x,_simParam.dimension.y)]);
        }
        cout<<endl;
    }
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
}
void printPopulationPosition(simParam _simParam, settings _settings){
    if (_settings.print > __DEBUG_PRINT_DEBUG__)cout << " # - printPopulationPosition --- "<<endl;
    for (size_t i = 0; i < _simParam.nbIndividual; i++)
    {
        cout << "\t  - " << i << " )\t" << _simParam.populationPosition[i].x << "\t" << _simParam.populationPosition[i].y <<endl;
    }
    if (_settings.print >= __DEBUG_PRINT_DEBUG__)cout << "   ->OK"<<endl;
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
cv::Scalar colorInterpol(cv::Scalar a, cv::Scalar b, float ratio) {
    cv::Scalar result;
    for (int i = 0; i < 4; ++i) {
        result[i] = a[i] + (b[i] - a[i]) * ratio;
    }
    return result;
}
