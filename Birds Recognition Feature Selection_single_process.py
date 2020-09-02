#!/usr/bin/env python
# coding: utf-8

# # Preface
# 
# We will inherit the feature engineering effort from
# https://www.kaggle.com/andradaolteanu/birdcall-recognition-eda-and-audio-fe and
# https://www.kaggle.com/parulpandey/eda-and-audio-processing-with-python to generate the set of features.
# 
# Then we will apply the analytical feature selection methods against a *lightgbm* modeller to see which features
# would work the best from a *lightgbm* stand-point.
#

# Ref.
# - https://musicinformationretrieval.com/index.html
# - https://www.kaggle.com/andradaolteanu/birdcall-recognition-eda-and-audio-fe
# - How I Understood: What features to consider while training audio files?
# - https://towardsdatascience.com/how-i-understood-what-features-to-consider-while-training-audio-files-eedfb6e9002b
# - Coronavirus: Using Machine Learning to Triage COVID-19 Patients -
# https://towardsdatascience.com/coronavirus-using-machine-learning-to-triage-covid-19-patients-980e62489fd4
# - The # dummy’s guide to MFCC - https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd
# - How to apply machine learning and deep learning methods to audio analysis -
# https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc

# # Feature Engineering Flow
#

import multiprocessing as mp
import datetime as dt
import pandas as pd
from typing import List, Dict, Tuple
import warnings

import utils as u
import config as c

warnings.filterwarnings('ignore')

# read data
in_kaggle = False


def get_data_file_path(is_in_kaggle: bool) -> Tuple[str, str]:
    train_path = ''
    test_path = ''

    if is_in_kaggle:
        # running in Kaggle, inside the competition
        train_path = '../input/birdsong-recognition/train.csv'
        test_path = '../input/birdsong-recognition/test.csv'
    else:
        # running locally
        train_path = 'data/train.csv'
        test_path = 'data/test.csv'

    return train_path, test_path


def get_base_train_audio_folder_path(is_in_kaggle: bool) -> str:
    folder_path = ''
    if is_in_kaggle:
        folder_path = '../input/birdsong-recognition/train_audio/'
    else:
        folder_path = 'data/train_audio/'
    return folder_path


def extract_feautres(trial_audio_file_path):
    # process data frame
    function_start_time = dt.datetime.now()
    print("Started a file processing at ", function_start_time)

    df0 = u.extract_feature_means(trial_audio_file_path)

    function_finish_time = dt.datetime.now()
    print("Fininished the file processing at ", function_finish_time)

    processing = function_finish_time - function_start_time
    print("Processed the file: ", trial_audio_file_path, "; processing time: ", processing)

    return df0


if __name__ == "__main__":
    start_time = dt.datetime.now()
    print("Started at ", start_time)

    # Import data
    train_set_path, test_set_path = get_data_file_path(in_kaggle)
    train_csv = pd.read_csv(train_set_path)
    test_csv = pd.read_csv(test_set_path)

    # Create some time features
    train_csv['year'] = train_csv['date'].apply(lambda x: x.split('-')[0])
    train_csv['month'] = train_csv['date'].apply(lambda x: x.split('-')[1])
    train_csv['day_of_month'] = train_csv['date'].apply(lambda x: x.split('-')[2])

    print("There are {:,} unique bird species in the dataset.".format(len(train_csv['species'].unique())))

    print(list(train_csv.columns))

    print(train_csv.head(10))

    print(test_csv.head(10))

    # Creating Interval for *duration* variable
    train_csv['duration_interval'] = ">500"
    train_csv.loc[train_csv['duration'] <= 100, 'duration_interval'] = "<=100"
    train_csv.loc[(train_csv['duration'] > 100) & (train_csv['duration'] <= 200), 'duration_interval'] = "100-200"
    train_csv.loc[(train_csv['duration'] > 200) & (train_csv['duration'] <= 300), 'duration_interval'] = "200-300"
    train_csv.loc[(train_csv['duration'] > 300) & (train_csv['duration'] <= 400), 'duration_interval'] = "300-400"
    train_csv.loc[(train_csv['duration'] > 400) & (train_csv['duration'] <= 500), 'duration_interval'] = "400-500"

    # Create Full Path so we can access data more easily
    base_dir = get_base_train_audio_folder_path(in_kaggle)
    train_csv['full_path'] = base_dir + train_csv['ebird_code'] + '/' + train_csv['filename']

    print(train_csv.head(10))

    ignore_list = []

    # started from comred
    final_data = list([species for species in c.LABELS if species not in ignore_list])

    corrupted_files = ["data/train_audio/lotduc/XC195038.mp3"]

    for ebird in final_data:
        file_start_time = dt.datetime.now()
        print("Starting to process a new species: ", ebird, "; at ", file_start_time)
        ebird_data = train_csv[train_csv['species'] == ebird]

        short_file_name = ebird_data['ebird_code'].unique()[0]
        print("Short file name: ", short_file_name)

        result = []

        for index, row in ebird_data.iterrows():
            # process each audio file
            if row['full_path'] not in corrupted_files:
                df = extract_feautres(row['full_path'])

                result.append(df)
            else:
                print("Skipping corrupted file: ", row['full_path'])

        # combine chunks with transformed data into a single training set
        extracted_features = pd.concat(result)

        # save extracted features to CSV
        output_path = "".join([c.TRANSFORMED_DATA_PATH, short_file_name, ".csv"])
        extracted_features.to_csv(output_path, index=False)
        file_end_time = dt.datetime.now()
        print("Finished processing: ", ebird, "; at ", file_end_time)
        elapsed = file_end_time - file_start_time
        print(ebird, ": ", elapsed, " s elapsed")

    print('We are done. That is all, folks!')
    finish_time = dt.datetime.now()
    print("Finished at ", finish_time)
    elapsed = finish_time - start_time
    print("Elapsed time: ", elapsed)

# Timeout reasons
#
# 1/ processing > 5 min
# Started a file processing at  2020-08-29 18:52:27.380628 Fininished the file processing at
# 2020-08-29 18:58:09.833741 Processed the file:  data/train_audio/comrav/XC246425.mp3 ; processing time:
# 0:05:42.453113
#
# 2/ File processing issues: data/train_audio/lotduc/XC195038.mp3 ;
# Started a file processing at
# 2020-08-29 19:06:06.555931 Traceback (most recent call last): File
# "D:/gvyshnya/trainings/kaggle_competitions/2020/birds-recognition/dev/Birds Recognition Feature
# Selection_single_process.py", line 147, in <module> df = extract_feautres(row['full_path']) File
# "D:/gvyshnya/trainings/kaggle_competitions/2020/birds-recognition/dev/Birds Recognition Feature
# Selection_single_process.py", line 73, in extract_feautres df0 = u.extract_feature_means(trial_audio_file_path)
# File "D:\gvyshnya\trainings\kaggle_competitions\2020\birds-recognition\dev\utils.py", line 180,
# in extract_feature_means y, sr = librosa.load(audio_file_path) File
# "C:\Users\USER-PC\anaconda\lib\site-packages\librosa\core\audio.py", line 171, in load y = resample(y, sr_native,
# sr, res_type=res_type) File "C:\Users\USER-PC\anaconda\lib\site-packages\librosa\core\audio.py", line 529,
# in resample ratio = float(target_sr) / orig_sr ZeroDivisionError: float division by zero
#
# 3/ processing > 5 min:
#    Processed the file:  data/train_audio/snobun/XC487556.mp3 ; processing time:  0:05:26.742162
#
#