//
// Created by Mathurin Cartron on 02/08/2023.
//
#include "utils.hpp"

// Change déclaration here
std::vector<float> mat3x3_perso = {
        1.f,1.f,1.f,
        5.f,0.f,5.f,
        0.f,9.f,0.f
};
std::vector<float> mat3x3_unitaire_full = {
        1.f,1.f,1.f,
        1.f,0.f,1.f,
        1.f,1.f,1.f
};
std::vector<float> mat3x3_unitaire_axes = {
        0.f,1.f,0.f,
        1.f,0.f,1.f,
        0.f,1.f,0.f
};
std::vector<float> mat3x3_unitaire_diagonal = {
        1.f,0.f,1.f,
        0.f,0.f,0.f,
        1.f,0.f,1.f
};
std::vector<float> mat5x5_unitaire_full = {
        1.f,1.f,1.f,1.f,1.f,
        1.f,1.f,1.f,1.f,1.f,
        1.f,1.f,0.f,1.f,1.f,
        1.f,1.f,1.f,1.f,1.f,
        1.f,1.f,1.f,1.f,1.f
};
std::vector<float> mat5x5_unitaire_axes = {
        0.f,0.f,1.f,0.f,0.f,
        0.f,0.f,1.f,0.f,0.f,
        1.f,1.f,0.f,1.f,1.f,
        0.f,0.f,1.f,0.f,0.f,
        0.f,0.f,1.f,0.f,0.f
};
std::vector<float> mat5x5_unitaire_diagonal = {
        1.f,1.f,0.f,1.f,1.f,
        1.f,1.f,0.f,1.f,1.f,
        0.f,0.f,0.f,0.f,0.f,
        1.f,1.f,0.f,1.f,1.f,
        1.f,1.f,0.f,1.f,1.f
};


std::vector<float> getMatrixMove(){
    return mat3x3_perso;
}
