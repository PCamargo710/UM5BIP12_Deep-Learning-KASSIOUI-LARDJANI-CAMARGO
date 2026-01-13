# UM5BIP12_Deep-Learning-KASSIOUI-LARDJANI-CAMARGO
Breast Cancer RNAseq analysis pipeline

This project is centered on the study of Lobular Breast Cancer, using deep learning approaches to better understand the pathologies and provide a way to predict pathologies using transcriptomic data. 

DOI: 10.1016/j.cell.2015.09.033

## Pipeline on Whole dataset

This part of the project aims to utilize MLP and AE approaches to predict the molelucar subtype PAM50 of our cohort, based on bulkRNAseq transcriptomic data.

This pipeline uses the dataset after QC and data_cohort to predict different features using AE+MLP approach on the metadata. 

This script has a custom network class that constructs, trains, and employs the autoencoder and the MLP in order to do the predictions.

A tuning function is used to find the best combinations of parameters, and a reproductibility cell gives peace of mind and ensure proper RNG management. 


