#include "Export.hpp"

Ex/**
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
          videoPath("video/"),
          videoSizeFactor(1),
          videoRatioFrame(__VIDEO_RATIO_FRAME__), // TODO: Implement dynamic value here
          videoNbFrame(0),
          tmpPath("tmp/frames/"),
          frameCounter(0),
          videoCalcCostPlot(__VIDEO_CALC_COST_PLOT_OFF__)
{
    if (map.getDimensions().x < __MAX_X_DIM_JPEG__ &&
        map.getDimensions().y < __MAX_Y_DIM_JPEG__) {
        this->videoSizeFactor = std::min((__MAX_X_DIM_JPEG__ / map.getDimensions().x), (__MAX_Y_DIM_JPEG__ / map.getDimensions().y));
    }

    // Check if the video folder exists, create if not
    struct stat info;
    if (stat(this->videoPath.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
        std::cout << "The folder already exists." << std::endl;
    } else {
        int status = mkdir(this->videoPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (status == 0) {
            std::cout << "The folder was created successfully." << std::endl;
        } else {
            std::cout << "Error creating the folder." << std::endl;
        }
    }

    // Determine videoCalcCostPlot value based on map dimensions
    if (max(map.getDimensions().x, map.getDimensions().y) > 50) {
        this->videoCalcCostPlot = __VIDEO_CALC_COST_PLOT_OFF__;
    } else {
        this->videoCalcCostPlot = __VIDEO_CALC_COST_PLOT_ON__;
    }
}


Export::~Export()
{
    this->frameCounter = 0;
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

}

void Export::compileFramesToVid() {
    // on créer le calce de cout

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
