# MGraphDTA: Deep Multiscale Graph Neural Network for Explainable Drug-target binding affinity Prediction

## Note
The concordance index (CI) is an important metric for performance evaluation of DTA prediction. However, the implemention of CI for DeepDTA and GraphDTA are inefficient. Here, we implement it in a high-efficiency way. Please see regression/metrics.py for details. One may also try to incorporate CI into the loss function to optimize it. 

## Dataset

All data used in this paper are publicly available can be accessed here:  

- Davis and KIBA: https://github.com/hkmztrk/DeepDTA/tree/master/data  
- Filtered Davis: https://github.com/cansyl/MDeePred
- Human and *C.elegans*: https://github.com/masashitsubaki/CPI_prediction  
- ToxCast: https://github.com/simonfqy/PADME  
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

## Descriptions of folders and files in the MGraphDTA repository

* **classification** folder includes the source code of MGraphDTA for classification tasks in the Human and *C.elegans* datasets. Note that this folder already contains raw data and can be used directly.
  + **data** folder contains raw data of the Human and *C.elegans* datasets.
  + **log** folder includes the source codes to record the training process.
  + **dataset.py** file prepares the data for training.
  + **metrics.py** contains a series of metrics to evalute the model performances.
  + **model.py**, the implementation of MGraphDTA can be found in here.
  + **preprocessing.py**, a file that preprocesses the raw data into graph format and should be executed before model trianing.
  + **test.py**, test a trained model and print the results.
  + **train.py**, train MGraphDTA model.
  + **utils.py** file includes useful tools for model training.
* **filtered_davis** folder includes the source code of MGraphDTA for regression task in the filtered davis dataset. Note that this folder already contains raw data and can be used directly. The training, validation, and test sets are extactly the same as MDeePred.
* **regression** folder includes the source code of MGraphDTA for regression tasks in the davis and KIBA datasets. Note that this folder do not contain raw data. The raw data can be downloaded from https://drive.google.com/file/d/1K21HJI72fmhryjXka_ijCrSgrCdcq2Or/view?usp=sharing  
* **visualization** folder includes the source code for visualization of a trained model. Note that this folder incudes a pretrained model for visualization. The raw data can be downloaded from https://drive.google.com/file/d/1K21HJI72fmhryjXka_ijCrSgrCdcq2Or/view?usp=sharing.
  * **visualization_mgnn.py**, this file includes algorithms that can produce heatmaps to reveal how MGraphDTA makes decisions. The core of this file is GradAAM class that takes a model and a module (layer) that you want to visualize as input where the module is chosen as the last layer of MGNN in our experiments. You can also try to visualize other layers. 

## Step-by-step running:  

### 1. Train/test MGraphDTA

#### 1.1 filtered_davis folder

- First, cd MGraphDTA/filtered_davis, and run preprocessing.py using  
  `python preprocessing.py`  

  Running preprocessing.py convert the raw data into graph format.

- Second, run train.py using 
  `python train.py --fold 0 --save_model` 

  to train MGraphDTA. The training record can be found in save/ folder.

  Explanation of parameters

  - --fold: k-th fold, from 0 to 4
  - --save_model: whether save model or not
  - --lr: learning rate, default =  5e-4
  - --batch_size: default = 512

- To test a trained model please run test.py using

  `python test.py --model_path model_path`

  This will return the RMSE, CI, and  Spearm performance in the test set.

  For example, running

  `python test.py --model_path '/home/yang/project/MGraphDTA/filtered_davis/save/20211127_040602_filtered_davis/model/epoch-149, loss-0.1245, cindex-0.9016, test_loss-0.4863.pt'`

  will output results as follows

  `Reading fold_0 from data/filtered_davis
  test_rmse:0.6973, test_cindex:0.7438, test_spearm:0.6642`

* To train MGraphDTA in your own datasets, please organize your data as the format shown in data/filtered_davis/raw/data.csv and provide a data/filtered_davis/warm.kfold file to describle the train/val/test split index.

#### 1.2 regression folder

- First, cd MGraphDTA/regression, and run preprocessing.py using  
  `python preprocessing.py`  

  Running preprocessing.py convert the raw data into graph format.

- Second, run train.py using 
  `python train.py --dataset davis --save_model` 

  to train MGraphDTA.

  Explanation of parameters

  - --dataset: davis or kiba
  - --save_model: whether save model or not
  - --lr: learning rate, default =  5e-4
  - --batch_size: default = 512

- To test a trained model please run test.py using

  `python test.py --dataset dataset --model_path model_path`

  This will return the MSE, CI, and  R2 performance in the test set.

#### 1.3 classification folder

- First, cd MGraphDTA/classification, and run preprocessing.py using  
  `python preprocessing.py`  
- Second, run train.py using 
  `python train.py --dataset human --save_model` for Human dataset and `python train.py --dataset celegans --save_model` for *C.elegans* dataset

### 2. Visualization using Grad-AAM

We provide an example of how to visualize MGNN using Grad-AAM.

- First, download the ToxCast dataset from https://drive.google.com/file/d/1K21HJI72fmhryjXka_ijCrSgrCdcq2Or/view?usp=sharing, copy full_toxcast folder into MGraphDTA/visualization/data.  
- Second, cd MGraphDTA/visualization, and run preprocessing.py using  
  `python preprocessing.py`  
- Third, run visualization_mgnn.py using  
  `python visualization_mgnn.py`  
  and you will the visualization results in MGraphDTA/visualization/results folders. Note that MGraphDTA/visualization/pretrained_model folder contains a pre-trained model that can be used to produce heatmaps. If you want to test Grad-AAM in your own model, please replace this pre-trained model with your own one and modify the path in the visualization_mgnn.py file.

