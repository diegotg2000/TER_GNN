# TER: Graph Neural Networks for glassy dynamics modelling

## Members:
- Etienne Efoe Blavo
- Diego Andres Torres Guarin

## Description:
This repository contains all the code used throughout the TER, from the first explorations with Pytorch geometric and toy datasets to the model used to try to replicate the results of the DeepMind paper. However, we focus only on the latter part, since it is the most important one to be able to replicate. 
The actual computations were performed on Kaggle, uploading the notebooks and necessary utility scripts. We provide instructions on how to replicate our results in the same way.



## Instructions:
The implementation of the model architecture described in the paper is done in the script `mpn_model.py`, and the notebook with the training settings is `train-bapts-mpn.ipynb`.
1. Upload the `mpn_model.py` to Kaggle, set the editor type as a utility script and do a quick save so that the code can be imported from other notebooks.
2. Uplaod the `train-bapts-mpn.ipynb` notebook, and in the Data tab on the right, import the following datasets:
    - Pytorch geometric wheels: https://www.kaggle.com/datasets/lyomega/torch-geometric
    - Bapst dataset: https://www.kaggle.com/datasets/quentinletellier/bapst-data-static-structure-in-glasses
    


3. Import `mpn_model.py` as a utility script.
4. (Optional): Activate the GPU acceleration, available on the tab on the right (you need to verify your kaggle account)

Training for 400 from scratch should give you around 0.56 correlation. The saved output cells of our notebooks show higher correlation because they come from a model that was previously trained for 550 epochs (as described in the report).