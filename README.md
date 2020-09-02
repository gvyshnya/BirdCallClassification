# Bird Call Classification
The repo contains various materials created as a part of Cornell Birdcall Identification project (https://www.kaggle.com/c/birdsong-recognition/)

# Files and Folders

## Python Scripts

- *Birds Recognition Feature Selection.py* - the script to implement the parallelized audio feature extraction flow using multiprocessing lib
- *Birds Recognition Feature Selection_single_process.py* - the script to implement the single-process 
- *dask_mp_feature_extraction.py* - the script implementing the parallelized audio feature extraction using Dask
- *ray_mp_feature_extraction.py* - the script implementing the parallelized audio feature extraction using Ray
- *utils.py* - the module with various audio feature extraction utility functions
- *config.py* - the configuration module of the project solution

## Data

- *data* - the folder to download the competition data as published in https://www.kaggle.com/c/birdsong-recognition/data
- *interim_data* - the folder for working interim output of various scripts within the solution
- *audio_features_data* - the set of audio features extracted from each audio file provided as a part of the training set

**Note:** the files in *audio_features_data* subfolder have been produced via running *Birds Recognition Feature Selection.py* 

# References and Publications

The materials of this repo are used as suppelmentary assets for several publications below

- 'Pythonic Audio Feature Extraction in the Age of Parallel and Distributed Computing' (https://medium.com/@gvyshnya/pythonic-audio-feature-extraction-in-the-age-of-parallel-and-distributed-computing-793b27641b6d)
- 'Parallelized Audio Feature Extraction Study' (https://www.kaggle.com/c/birdsong-recognition/discussion/179662)
- 'Audio Feature Extraction: Best Practices' (https://www.kaggle.com/c/birdsong-recognition/discussion/172573)

