/*******************************************************************************
* File Name: main.cpp
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: main file of the crowd simulation with parallelization on GPU. Contains only the main program.
*******************************************************************************/


// Include necessary libraries here

#include "utils/utils.hpp"

#include "Kernel.hpp"
#include "Map.hpp"
#include "Population.hpp"
#include "Settings.hpp"
#include "Simulation.hpp"
#include "Export.hpp"

// Main function
int main(int argc, char const *argv[])
{
    srand(time(NULL));

    // Output the title for the simulation
    std::cout << "---------- Crowd Simulation ----------" << std::endl;

    clock_t start = clock();
    // Lecture du fichier de setup.txt
    std::string pathSetup = std::string("../../input Image/") +
                          std::string(__NAME_DIR_INPUT_FILES__) +
                          std::string("/setup.txt");

    std::ifstream configFile(pathSetup); // Ouvrir le fichier en lecture
    if (!configFile.is_open()) {
        if(__PRINT_DEBUG__)std::cerr << "Erreur lors de l'ouverture du fichier de configuration." << std::endl;
        return 1;
    }

    std::map<std::string, std::string> configVariables; // Utiliser un std::map pour stocker les variables

    std::string line;
    while (std::getline(configFile, line)) {
        size_t pos = line.find(" ");
        if (pos != std::string::npos) {
            std::string variable = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            configVariables[variable] = value;
        }
    }

    // Rapel dans le terminal des config
    std::cout << "CHARGEMENT DU FICHIER SETUP" << std::endl
            << "__NAME_VIDEO__ : " << configVariables["__NAME_VIDEO__"] << std::endl
            << "__PONDERATION_ABSOLUE__ : " << configVariables["__PONDERATION_ABSOLUE__"] << std::endl
            << "__VIDEO_FPS__ : " << configVariables["__VIDEO_FPS__"] << std::endl
            << "__NB_FRAMES_MAX__ : " << configVariables["__NB_FRAMES_MAX__"] << std::endl
            << "__MATRIX_POP_0__ : " << configVariables["__MATRIX_POP_0__"] << std::endl;

    configFile.close(); // Fermer le fichier

    // Pour savoire automatiquement le nombre de populations
    std::string pathDir = "../../input Image/" + std::string(__NAME_DIR_INPUT_FILES__);
    std::__fs::filesystem::path directoryPath(pathDir);
    int fileCount = 0;
    if (std::__fs::filesystem::is_directory(directoryPath)) {

        for (const auto& entry : std::__fs::filesystem::directory_iterator(directoryPath)) {
            if (std::__fs::filesystem::is_regular_file(entry)) {
                fileCount++;
            }
        }

        if(__PRINT_DEBUG__)std::cout << "Nombre de fichiers dans le dossier : " << fileCount << std::endl;
    } else {
        if(__PRINT_DEBUG__)std::cerr << "Le chemin n'est pas un dossier valide." << std::endl;
    }
    fileCount = (fileCount-2)/2;

    // Create a Map object
    // Note: The usage of 'new' is not recommended as it can cause memory leaks.
    // Instead, consider using smart pointers or stack-based objects.
    std::string pathMap = std::string("../../input Image/") +
                std::string(__NAME_DIR_INPUT_FILES__) +
                std::string("/map.png");
    Map map = *new Map(pathMap);

    // Create a Population object named "Test1"
    for (int i = 0; i < fileCount; i++) {
        std::string pathPopulation = std::string("../../input Image/") +
                                     std::string(__NAME_DIR_INPUT_FILES__) +
                                     std::string("/population") +
                                     std::to_string(i) +
                                     std::string(".png");
        std::string pathExit = std::string("../../input Image/") +
                               std::string(__NAME_DIR_INPUT_FILES__) +
                               std::string("/exit") +
                               std::to_string(i) +
                               std::string(".png");
        std::string nameVarMatrix = std::string("__MATRIX_POP_") +
                                    std::to_string(i) +
                                    std::string("__");
        std::string nameVarColor = std::string("__COLOR_POP_") +
                                   std::to_string(i) +
                                   std::string("__");
        Population population = *new Population(pathPopulation,pathExit);
        // Création de la matrice pour la population
        std::vector<float> floatVector;
        std::istringstream iss1(configVariables[nameVarMatrix]);
        std::string token;
        while (std::getline(iss1, token, ',')) {
            float floatValue = std::stof(token);
            floatVector.push_back(floatValue);
        }
        population.setPMovement(floatVector);

        // Création de la couleur pour la population
        std::vector<int> intVector;
        std::istringstream iss2(configVariables[nameVarColor]);
        while (std::getline(iss2, token, ',')) {
            int intValue = std::stof(token);
            intVector.push_back(intValue);
        }
        population.setColor(intVector[0],intVector[1],intVector[2]);

        map.addPopulation(population);
    }
    //calculat Map Cost
    int passeCarteCout = map.initCostMap();

    // Print the Map and its contents
    if(__PRINT_DEBUG__)map.print();

    // initialisaiton du kernel
    Kernel kernel(map);
    // initialisation de l'export
    Export extp(map);
    if (std::string(configVariables["__NAME_VIDEO__"]).size()>0){
        extp.setVideoFilename(configVariables["__NAME_VIDEO__"] + std::string(".mp4"));
    }
    if (std::string(configVariables["__VIDEO_FPS__"]).size()>0){
        extp.setFps(atoi(configVariables["__VIDEO_FPS__"].c_str()));
    }


    int minExit = INT_MAX;
    for(auto &i : map.getPopulations()){
        minExit = std::min(minExit, (int)i.getExits().size());
    }
    long int secu = std::sqrt(std::pow(map.getDimensions().x,2) * std::pow(map.getDimensions().y,2))*kernel.getPopulation().size();
    if (atoi(configVariables["__NB_FRAMES_MAX__"].c_str()) > 0) {
        secu = (secu < atoi(configVariables["__NB_FRAMES_MAX__"].c_str())) ? secu : atoi(configVariables["__NB_FRAMES_MAX__"].c_str());
    }

    std::cout << "INITIALISATION DES PARAMÈTRES DE LA SIMULATION" << std::endl
            << " - Nb mure : " << map.getWallPositions().size() << std::endl
            << " - Nb passe carte coût : " << passeCarteCout << std::endl
            << " - Nb population : " << map.getPopulations().size() << std::endl
            << " - Nb individu total : " << kernel.getPopulation().size() << std::endl
            << " - Frames max : " << secu << std::endl;
              //<< " - Nb sorti total" << kernel. << std::endl;

    if (minExit == 0) {
        std::cout << "#ERREUR : l'une des populations ne possède aucune sortie"<< std::endl;
        return 0;
    }

    int testFin = 1;
    int frame = 0;
    extp.creatFrame(kernel); // enregistrer la frame 0

    clock_t end = clock();
    double init_time = (double)(end - start) / CLOCKS_PER_SEC;

    std::cout << "DÉBUT DE LA SIMULATION "<< std::endl;

    start = clock();
    while (testFin > 0 && frame < secu){
        frame ++;
        if (configVariables["__PONDERATION_ABSOLUE__"] == "true")    testFin = kernel.computeNextFrame2();
        else                            testFin = kernel.computeNextFrame();
        extp.creatFrame(kernel);
        progressBar((int)kernel.getPopulation().size()-testFin, (int)kernel.getPopulation().size(), 200, frame);
    }
    end = clock();
    double simul_time = (double)(end - start) / CLOCKS_PER_SEC;

    std::cout << std::endl;
    std::cout << "CRÉATION DE LA VIDÉO "<< std::endl;
    start = clock();
    extp.compileFramesToVid(map);
    std::cout << std::endl;
    end = clock();
    double expt_time = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Vidéo enregistré dans : " << extp.getVideoPath() << extp.getVideoFilename() << std::endl;

    int tmpTotal = expt_time + simul_time + init_time;
    std::cout << "RÉSUMÉ " << std::endl
              << " - calculé en " << frame << " frames " << std::endl;
    if(init_time > 60) {
        init_time /= 60;
        std::cout << " - temps pour l'initialisation " << init_time << " min " << std::endl;
    } else {
        std::cout << " - temps pour l'initialisation " << init_time << " sec " << std::endl;
    }
    if(simul_time > 60) {
        simul_time /= 60;
        std::cout << " - temps pour la simulation    " << simul_time << " min " << std::endl;
    } else {
        std::cout << " - temps pour la simulation    " << simul_time << " sec " << std::endl;
    }
    if(expt_time > 60) {
        expt_time /= 60;
        std::cout << " - temps pour l'export vidéo   " << expt_time << " min " << std::endl;
    } else {
        std::cout << " - temps pour l'export vidéo   " << expt_time << " sec " << std::endl;
    }

    if(tmpTotal > 60) {
        std::cout << " - temps total                 " << (float)(tmpTotal /= 60) << " min " << std::endl;
    } else {
        std::cout << " - temps total                 " << tmpTotal << " sec " << std::endl;
    }

    // End the main function and return 0, indicating successful execution
    return 0;
}
