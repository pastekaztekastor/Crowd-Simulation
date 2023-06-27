/*******************************************************************************
* File Name: main.hpp
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: Programme qui génère des image a partir des fichier HDF5
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <hdf5.h>

#pragma pack(push, 1)
typedef struct {
    uint32_t width;
    uint32_t height;
} ImageHeader;
#pragma pack(pop)

int main() {
    const char* filename = "chemin/vers/fichier.h5";
    const char* outputFilename = "output.ppm";
    uint32_t imgWidth = 50;
    uint32_t imgHeight = 50;
    
    // Charger les positions à partir du fichier HDF5
    int** positions = loadPositionsFromFile(filename, imgWidth, imgHeight);
    if (positions == NULL) {
        printf("Erreur lors du chargement des positions du fichier.\n");
        return 1;
    }
    
    // Créer une image PPM avec les positions
    createPPMImage(outputFilename, positions, imgWidth, imgHeight);
    
    // Libérer la mémoire
    freePositions(positions, imgHeight);
    
    printf("L'image a été créée avec succès.\n");
    
    return 0;
}

int** loadPositionsFromFile(const char* filename, uint32_t width, uint32_t height) {
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        printf("Erreur lors de l'ouverture du fichier HDF5.\n");
        return NULL;
    }
    
    hid_t dataset_id = H5Dopen(file_id, "populationPosition", H5P_DEFAULT);
    if (dataset_id < 0) {
        printf("Erreur lors de l'ouverture du dataset dans le fichier HDF5.\n");
        H5Fclose(file_id);
        return NULL;
    }
    
    hid_t dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        printf("Erreur lors de l'obtention de l'espace de données dans le fichier HDF5.\n");
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }
    
    hsize_t dims[2];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    if (dims[0] != height || dims[1] != width) {
        printf("Dimensions invalides dans le fichier HDF5.\n");
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }
    
    int** positions = (int**)malloc(height * sizeof(int*));
    if (positions == NULL) {
        printf("Erreur lors de l'allocation de mémoire pour les positions.\n");
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }
    
    for (uint32_t i = 0; i < height; i++) {
        positions[i] = (int*)malloc(width * sizeof(int));
        if (positions[i] == NULL) {
            printf("Erreur lors de l'allocation de mémoire pour les positions.\n");
            for (uint32_t j = 0; j < i; j++) {
                free(positions[j]);
            }
            free(positions);
            H5Sclose(dataspace_id);
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            return NULL;
        }
    }
    
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, positions[0]);
    if (status < 0) {
        printf("Erreur lors de la lecture des positions dans le fichier HDF5.\n");
        for (uint32_t i = 0; i < height; i++) {
            free(positions[i]);
        }
        free(positions);
    }
    
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    
    return positions;
}


void createPPMImage(const char* filename, int** positions, uint32_t width, uint32_t height) {
    // Créer un fichier PPM pour l'image
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Erreur lors de la création du fichier PPM.\n");
        return;
    }
    
    // Écrire l'en-tête de l'image PPM
    ImageHeader header;
    header.width = width;
    header.height = height;
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    
    // Écrire les pixels de l'image
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            if (positions[y][x] == 1) {
                fputc(0, file); // Composante rouge
                fputc(0, file); // Composante verte
                fputc(0, file); // Composante bleue
            } else {
                fputc(255, file); // Composante rouge
                fputc(255, file); // Composante verte
                fputc(255, file); // Composante bleue
            }
        }
    }
    
    // Fermer le fichier
    fclose(file);
}

void freePositions(int** positions, uint32_t height) {
    // Libérer la mémoire allouée pour les positions
    for (uint32_t y = 0; y < height; y++) {
        free(positions[y]);
    }
    free(positions);
}
