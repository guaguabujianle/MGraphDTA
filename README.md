# MGraphDTA: Deep Multiscale Graph Neural Network for Explainable Drug-target binding affinity Prediction

## Dataset
All data used in this paper are publicly available can be accessed here:  
(1) Davis and KIBA: https://github.com/hkmztrk/DeepDTA/tree/master/data  
(2) Human and *C.elegans*: https://github.com/masashitsubaki/CPI_prediction  
(3) ToxCast: https://github.com/simonfqy/PADME  
Or you can download the datasets in https://drive.google.com/file/d/1K21HJI72fmhryjXka_ijCrSgrCdcq2Or/view?usp=sharing  

## Requirements  
matplotlib==3.2.2  
pandas==1.2.4  
torch_geometric==1.7.0  
CairoSVG==2.5.2  
torch==1.7.1  
tqdm==4.51.0  
opencv_python==4.5.1.48  
networkx==2.5.1  
numpy==1.20.1  
ipython==7.24.1  
rdkit==2009.Q1-1  
scikit_learn==0.24.2  

## Step-by-step running:  
### 0.Visulization using Grad-AAM
First, download the ToxCast dataset from https://drive.google.com/file/d/1K21HJI72fmhryjXka_ijCrSgrCdcq2Or/view?usp=sharing, copy full_toxcast folder into MGraphDTA/visualization/data.  
Second, cd MGraphDTA/visualization, and run preprocessing.py using  
`python preprocessing.py`  
Third, run visualization_mgnn.py using  
`python visualization_mgnn.py`  
and you will the visualization results in MGraphDTA/visualization/results folders  

### 1. Classification  
First, cd MGraphDTA/classification, and run preprocessing.py using  
`python preprocessing.py`  
Second, run train.py using 
`python train.py --dataset human --save_model` for Human dataset and `python train.py --dataset celegans --save_model` for *C.elegans* dataset

### 2. Regression
Similar to Classification.




