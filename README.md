# LWinNN

This repository contains the implementation of local window nearest neighbors (LWinNN). LWinNN is a method for industrial anomaly detection that uses nearest neighbors in local windows. LWinNN focusses on creating invariance to minor translations, a problem much simpler than other methods tackle. This makes for a much lighter algorithm with minimal train and test time, but achieves higher accuracy than competitors. In our paper we explain our reasoning why other transformation variance cannot be created easily, and also why it does not have to be created. 

## Quick start guide
Using LWinNN requires 4 steps:
1. Clone the repository
2. Downloading the datasets
3. Create a python or conda environment
4. Run the main.py file in terminal

### Cloning the repository
Clone the repository using:
`git clone https://github.com/anonymoushoneybadger75/LWinNN.git`

### Downloading the datasets
The datasets we use in our paper are the [MVTec-AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) and the [VisA dataset](https://github.com/amazon-science/spot-diff). Follow the instructions on the linked pages to download and extract the datasets and place them in a new folder in current root called `Data`. The VisA page links to a GitHub page that also provides scripts to restructure the folders, but we just use the default format upon downloading. Our code will likely not work properly with other folder structures than the default.
The default folder names for these datasets are 'mvtec-anomaly-detection' and 'VisA_20220922'. If these foldernames differ, change this in the src/Dataloaders/AD_Dataset.py file.

### Creating an environment
We provide a requirements.txt file to create a conda environment. As many problems can still occur, our advise for creating an environment manually is to create an environment with python==3.10.15, [install the most recent (stable) version of pytorch with the right GPU setup](https://pytorch.org/get-started/locally/), and install the full version of [anomalib](https://github.com/openvinotoolkit/anomalib). For our hardware setup these commands were sufficient:
```shell
conda create -n lwinnn python=3.10.15
# find the torch version compatible with your OS on https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install anomalib[full]
```
Anomalib is only needed for the IAD-specific AUPRO metric. We provide an Anomalib-free branch that uses AUROC for segmentation scores instead of AUPRO. Some packages may have to be installed manually (like Pandas) when not installing Anomalib.

### Run main.py
Running main.py:
`python main.py`
Runs LWinNN with default settings, for the bottle category of the MVTec-AD dataset. Main.py will print anomaly detection score in AUROC, anomaly segmentation in AUPRO, and train and test time in seconds to console. 

## Using LWinNN with other settings
The main functionality of LWinNN is as follows:
1. main.py creates a dataloader and gives it to LWinNN_Backend.py
2. LWinNN_Backend.py uses Embedder.py to extract an embedding for batches of images.
3. LWinNN_Backend.py then reshapes the embeddings and gives them to LWinNN_Model.py.
4. a. If training, LWinNN_Model.py writes all train embeddings to memory.
4. b. If testing, LWinNN_Model.py uses local window nearest neighbors between test and train embeddings to calculate patch anomaly scores.
5. Finally, LWinNN_Model.py calculates anomaly detection and gives results back to LWinNN_Backend.py. 


### Other settings
The main.py file has a number of options. Changing to a different category or dataset is done as follows:
`python main.py --dataset visa --category capsules --dataset_path 'alternative_path'`
The VisA and MVTec dataset have different default folder structures and specifying both the dataset and a path is needed. --dataset_path refers to the location of the dataset folder, not the dataset folder itself.

When hardware resources are limited, the amount of training samples or batch size can be configured with 
`python main.py --limit_train_samples 100 --batch_size 16`

Other hardware settings can be configured with
`python main.py --num_workers 4 --gpu_type cuda --gpu_number 2`
Our code provides (unstable) support for mps devices. Clearing cache on mps seems to set tensors to zero in some cases, greatly disturbing results. Enabling MPS therefore disables manual cache emptying. 

Window size is our only hyperparameter and can be configured with
`python main.py --window_size 7`

And Embedding extraction details can be configured with
`python main.py --normalize False --preserve_aspect_ratio False --pool False --interpolation_mode nearest`

Finally all scores and parameters can be written to a csv file in the results folder with
`python main.py --write_results trial.csv`

### Benchmarking
To benchmark on both datasets, run benchmark.sh in terminal with
`bash benchmark.sh`
Hardware details and (hyper)parameters can be changed in this file accordingly. When using a GPU, running this script will only take a few minutes. 

### Ablation study
To reproduce the ablation study from our paper, configure the hardware settings in ablation_study.sh and execute with:
`bash ablation_study.sh`
Depending on the hardware setup, running this script can take several hours. 
