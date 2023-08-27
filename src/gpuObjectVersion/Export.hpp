#ifndef EXPORT_HPP
#define EXPORT_HPP

#include "utils/utils.hpp"
#include "Simulation.hpp"
#include "Kernel.hpp"
#include "Map.hpp"

#define __PATH_TEMP_FRAME__     "../../tmp/frames/"
#define __PATH_VIDEO__          "../../video/"

class Export
{
private:
    // Pour la vidéo
    uint2 dimensionSimulation;
    std::vector<cv::Mat> videoFrames;
    std::string videoFilename;
    std::string videoPath;
    int videoSizeFactor;
    int videoRatioFrame;
    int videoNbFrame;
    cv::VideoWriter videoWriter;
    cv::Mat videoCalcCost;
    int videoCalcCostPlot;

    // Pour les fichié temporaire
    std::string tmpPath;
    int frameCounter;

public:
    /**
     * Constructor for the Export class, responsible for exporting simulation frames to video.
     * Initializes various parameters for video export, including file names, paths, and settings.
     *
     * @param map The map object used for the simulation.
     */
    Export(Map map);

    ~Export();

    const std::vector<cv::Mat> &getVideoFrames() const;
    void setVideoFrames(const std::vector<cv::Mat> &videoFrames);
    const std::string &getVideoFilename() const;
    void setVideoFilename(const std::string &videoFilename);
    const std::string &getVideoPath() const;
    void setVideoPath(const std::string &videoPath);
    int getVideoSizeFactor() const;
    void setVideoSizeFactor(int videoSizeFactor);
    int getVideoRatioFrame() const;
    void setVideoRatioFrame(int videoRatioFrame);
    int getVideoNbFrame() const;
    void setVideoNbFrame(int videoNbFrame);
    const cv::VideoWriter &getVideoWriter() const;
    void setVideoWriter(const cv::VideoWriter &videoWriter);
    const cv::Mat &getVideoCalcCost() const;
    void setVideoCalcCost(const cv::Mat &videoCalcCost);
    int getVideoCalcCostPlot() const;
    void setVideoCalcCostPlot(int videoCalcCostPlot);
    const std::string &getTmpPath() const;
    void setTmpPath(const std::string &tmpPath);
    int getFrameCounter() const;
    void setFrameCounter(int frameCounter);

    // Créer un frame à partire de la simulation.
    /**
     * TODO write comment
     * @param kernel
     */
    void creatFrame(Kernel kernel);

    // Une fois la simulaiton terminé utilise tous les fichiers temporaires pour générer la vidéo.
    /**
     * TODO write comment
     */
    void compileFramesToVid(Map map);

    // créer le calque avec les couts qui sont affichés
    /**
     * TODO comment
     * @param map
     */
    void creatCalcCost(Map map);
};

#endif //EXPORT_HPP
