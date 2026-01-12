# UM5BIP12_Deep-Learning-KASSIOUI-LARDJANI-CAMARGO
Breast Cancer RNAseq analysis pipeline

This project is centered on the study of Lobular Breast Cancer, using deep learning approaches to better understand the pathologies and provide a way to predict pathologies using transcriptomic data. 

DOI: 10.1016/j.cell.2015.09.033

## Pipeline on Whole dataset

This part of the project aims to utilize MLP and AE approaches to predict the final pathology of our cohort, based on bulkRNAseq transcriptomic data.

### This pipeline is separated into 2 scripts: 

-Projet_DL_Data-Loading.ipynb: handles the loading of the cohort matrix obtained from the source article, cleans the data and then saves it into 2 dataframes: "test_matrix" and "data_cohort".
- test_matrix is a dataframe patiens x genes which is the result of bulkRNAseq. 
- data_cohort is a dataframe patiens x metadata which carries all the metadata that is used in this project. 

-Projet_DL.ipynb: uses test_matrix and data_cohort to predict different features using AE+MLP approach. 
    This script has a custom network class that constructs, trains, and employs the autoencoder and the MLP in order to do the predictions. Its fully modular and can be configured easily. 


