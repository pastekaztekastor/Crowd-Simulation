#include "Export.hpp"

/**
 * Constructor for the Export class, responsible for exporting simulation frames to video.
 * Initializes various parameters for video export, including file names, paths, and settings.
 *
 * @param map The map object used for the simulation.
 */
Export::Export(Map map)
        : videoFilename("anim_" +
                        std::to_string(map.getDimensions().x) + "X-" +
                        std::to_string(map.getDimensions().y) + "Y-" +
                        std::to_string(map.getPopulations().size()) + "P.mp4"),
          videoPath(__PATH_VIDEO__),
          videoSizeFactor(1),
          videoRatioFrame(__VIDEO_RATIO_FRAME__), // TODO: Implement dynamic value here
          videoNbFrame(0),
          tmpPath(__PATH_TEMP_FRAME__),
          frameCounter(0),
          videoCalcCostPlot(__VIDEO_CALC_COST_PLOT_OFF__),
          dimensionSimulation(map.getDimensions())
{
    if (map.getDimensions().x < __MAX_X_DIM_JPEG__ &&
        map.getDimensions().y < __MAX_Y_DIM_JPEG__) {
        this->videoSizeFactor = std::min((__MAX_X_DIM_JPEG__ / map.getDimensions().x), (__MAX_Y_DIM_JPEG__ / map.getDimensions().y));
    }

    // Check if the video folder exists, create if not
    struct stat info;
    if (stat(this->videoPath.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
        if (__PRINT_DEBUG__)std::cout << "The folder already exists." << std::endl;
    } else {
        int status = mkdir(this->videoPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (status == 0) {
            if (__PRINT_DEBUG__)std::cout << "The folder was created successfully." << std::endl;
        } else {
            if (__PRINT_DEBUG__)std::cout << "Error creating the folder." << std::endl;
        }
    }

    // Determine videoCalcCostPlot value based on map dimensions
    if (std::max(map.getDimensions().x, map.getDimensions().y) > 50) {
        this->videoCalcCostPlot = __VIDEO_CALC_COST_PLOT_OFF__;
    } else {
        this->videoCalcCostPlot = __VIDEO_CALC_COST_PLOT_ON__;
    }
}


Export::~Export()
{
}

const std::vector<cv::Mat> &Export::getVideoFrames() const {
    return videoFrames;
}

void Export::setVideoFrames(const std::vector<cv::Mat> &videoFrames) {
    Export::videoFrames = videoFrames;
}

const std::string &Export::getVideoFilename() const {
    return videoFilename;
}

void Export::setVideoFilename(const std::string &videoFilename) {
    Export::videoFilename = videoFilename;
}

const std::string &Export::getVideoPath() const {
    return videoPath;
}

void Export::setVideoPath(const std::string &videoPath) {
    Export::videoPath = videoPath;
}

int Export::getVideoSizeFactor() const {
    return videoSizeFactor;
}

void Export::setVideoSizeFactor(int videoSizeFactor) {
    Export::videoSizeFactor = videoSizeFactor;
}

int Export::getVideoRatioFrame() const {
    return videoRatioFrame;
}

void Export::setVideoRatioFrame(int videoRatioFrame) {
    Export::videoRatioFrame = videoRatioFrame;
}

int Export::getVideoNbFrame() const {
    return videoNbFrame;
}

void Export::setVideoNbFrame(int videoNbFrame) {
    Export::videoNbFrame = videoNbFrame;
}

const cv::VideoWriter &Export::getVideoWriter() const {
    return videoWriter;
}

void Export::setVideoWriter(const cv::VideoWriter &videoWriter) {
    Export::videoWriter = videoWriter;
}

const cv::Mat &Export::getVideoCalcCost() const {
    return videoCalcCost;
}

void Export::setVideoCalcCost(const cv::Mat &videoCalcCost) {
    Export::videoCalcCost = videoCalcCost;
}

int Export::getVideoCalcCostPlot() const {
    return videoCalcCostPlot;
}

void Export::setVideoCalcCostPlot(int videoCalcCostPlot) {
    Export::videoCalcCostPlot = videoCalcCostPlot;
}

const std::string &Export::getTmpPath() const {
    return tmpPath;
}

void Export::setTmpPath(const std::string &tmpPath) {
    Export::tmpPath = tmpPath;
}

int Export::getFrameCounter() const {
    return frameCounter;
}

void Export::setFrameCounter(int frameCounter) {
    Export::frameCounter = frameCounter;
}

void Export::creatFrame(Kernel kernel) {

    // Check if the video folder exists, create if not
    struct stat info;
    if (stat(this->tmpPath.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
        if (__PRINT_DEBUG__)std::cout << "The folder already exists." << std::endl;
    } else {
        int status = mkdir(this->tmpPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (status == 0) {
            if (__PRINT_DEBUG__)std::cout << "The folder was created successfully." << std::endl;
        } else {
            if (__PRINT_DEBUG__)std::cout << "Error creating the folder." << std::endl;
        }
    }

    /*
     * Créer un fichier txt.
     *  - nom du fichier : this->frameCounter;
     *  - chemin : this->tmpPath;
     *  - contenue du fichier :
     *    - une ligne par éléments dans kernel.getPopulation();
     *    - chaque ligne contient " kernel.getPopulation()[i].x kernel.getPopulation()[i].y kernel.getPopulation()[i].z "
     */

    // Create a txt file
    char filePath[256];
    snprintf(filePath, sizeof(filePath), "%s%d.txt", tmpPath.c_str(), frameCounter);

    std::vector<individu> population = kernel.getPopulation();
    FILE *file = fopen(filePath, "w");
    if (file) {
        for (auto & i : population) {
            fprintf(file, "%d %d %d %d %d\n",  i.from, i.id, i.position.x, i.position.y, i.position.z);
        }
        fclose(file);
    } else {
        printf("Error creating the file.\n");
    }
    this->frameCounter ++;
}

void Export::compileFramesToVid(Map map) {
    std::string fullVideoPath = videoPath + videoFilename;
    // créer la vidéo
    cv::Mat frameType(map.getDimensions().y * videoSizeFactor, map.getDimensions().x * videoSizeFactor, CV_8UC3,__COLOR_GREY__);
    videoWriter.open(fullVideoPath, cv::VideoWriter::fourcc( 'a', 'v', 'c', '1'), __VIDEO_FPS__, frameType.size(), true);

    // Vérifier si le VideoWriter a été correctement initialisé
    if (!videoWriter.isOpened()) {
        if (__PRINT_DEBUG__)std::cout << "Erreur lors de l'ouverture du fichier : " << std::endl;
    }
    else
    {
        // ouvrir toutes les frames :
        for (int i = 1; i <= frameCounter; i++) {
            cv::Mat frame(map.getDimensions().y * videoSizeFactor, map.getDimensions().x * videoSizeFactor, CV_8UC3,__COLOR_GREY__);

            char filePath[256];
            snprintf(filePath, sizeof(filePath), "%s%d.txt", tmpPath.c_str(), i);

            // on place les gens de couleur LOL
            FILE *file = fopen(filePath, "r");  // Ouvrir le fichier en mode lecture
            if (file != NULL) {
                // Traiter le contenu du fichier ici
                int x, y, z, id, from;
                while (fscanf(file, "%d %d %d %d %d", &from, &id, &x, &y, &z) == 5) {
                    cv::Point center((x * videoSizeFactor) + (videoSizeFactor / 2),
                                     (y * videoSizeFactor) + (videoSizeFactor / 2));
                    float radius = (videoSizeFactor / 2) * 0.9;
                    // TODO faire l'interpolation de couleur quand ils attende
                    int thickness = -1;  // Remplacer par un nombre positif pour un contour solide
                    cv::circle(frame, center, radius, map.getPopulations()[id].getColorScalar(), thickness);
                }
                // Fermer le fichier après traitement
                fclose(file);
            } else {
                printf("Erreur lors de l'ouverture du fichier %s\n", filePath);
            }

            // on place les mures
            for (auto &wall: map.getWallPositions()) {
                cv::Point TL(wall.x * videoSizeFactor, wall.y * videoSizeFactor);
                cv::Point BR(TL.x + videoSizeFactor, TL.y + videoSizeFactor);
                cv::Rect rectangle(TL, BR);
                cv::rectangle(frame, rectangle, __COLOR_WHITE__, -1);
            }

            // Placer les sorties
            for (Population p: map.getPopulations()) {
                for (auto &exit: p.getExits()) {
                    cv::Point TL(exit.x * videoSizeFactor, exit.y * videoSizeFactor);
                    cv::Point BR(TL.x + videoSizeFactor, TL.y + videoSizeFactor);
                    cv::Rect rectangle(TL, BR);
                    cv::rectangle(frame, rectangle, p.getColorScalar(), -1);
                }
            }
            // Superposition avec le calque de cout de chaque case
            /*
            if (videoCalcCostPlot == __VIDEO_CALC_COST_PLOT_OFF__) {
                // Paramètre pour la superposition
                double alpha = 0.5; // Facteur de pondération pour l'image 1
                double beta = 0.5;  // Facteur de pondération pour l'image 2
                double gamma = 0.0; // Paramètre d'ajout d'un scalaire

                // Superposer les images
                cv::addWeighted(videoCalcCost, alpha, frame, beta, gamma, frame);
            }
            */

            // Exporte la frame
            videoWriter.write(frame);
            progressBar(i, frameCounter, 200, frameCounter);
        }
        videoWriter.release();
    }
}

void Export::creatCalcCost(Map map){
    if ( videoCalcCostPlot == __VIDEO_CALC_COST_PLOT_ON__){
        if (map.getPopulations().size() == 1){
            videoCalcCost = cv::Mat(map.getDimensions().x * videoSizeFactor, map.getDimensions().y * videoSizeFactor, CV_8UC3, __COLOR_ALPHA__);
            for (size_t i = 0; i < map.getPopulations()[0].getMapCost().size(); i++){
                // Paramètres du texte
                std::string texte = std::to_string(map.getPopulations()[0].getMapCost()[i]);
                int x = (i % map.getDimensions().x) * videoSizeFactor + (videoSizeFactor * 0.9);
                int y = (i / map.getDimensions().y) * videoSizeFactor + (videoSizeFactor * 0.9);
                cv::Point position(x, y);
                int epaisseur = 1;
                float taillePolice = 0.8;
                int ligneType = cv::LINE_AA;

                // Écrire le texte sur l'image
                cv::putText(videoCalcCost, texte, position, cv::FONT_HERSHEY_SIMPLEX, taillePolice, __COLOR_GREY__, epaisseur, ligneType);
            }
        }
    }
}
