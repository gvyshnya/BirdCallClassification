import pandas as pd
import numpy as np
import librosa

import config as c

# ref.: https://musicinformationretrieval.com/basic_feature_extraction.html
def extract_features(audio_file_path: str) -> pd.DataFrame:
    # config settings
    number_of_mfcc = c.NUMBER_OF_MFCC

    # 1. Importing 1 file
    y, sr = librosa.load(audio_file_path)

    # Trim leading and trailing silence from an audio signal (silence before and after the actual audio)
    signal, _ = librosa.effects.trim(y)

    # 2. Fourier Transform
    # Default FFT window size
    n_fft = c.N_FFT  # FFT window size
    hop_length = c.HOP_LENGTH  # number audio of frames between STFT columns (looks like a good default)

    # Short-time Fourier transform (STFT)
    d_audio = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))

    # 3. Spectrogram
    # Convert an amplitude spectrogram to Decibels-scaled spectrogram.
    db_audio = librosa.amplitude_to_db(d_audio, ref=np.max)

    # 4. Create the Mel Spectrograms
    s_audio = librosa.feature.melspectrogram(signal, sr=sr)
    s_db_audio = librosa.amplitude_to_db(s_audio, ref=np.max)

    # 5 Zero crossings

    # #6. Harmonics and Perceptrual
    # Note:
    #
    # Harmonics are characteristichs that represent the sound color
    # Perceptrual shock wave represents the sound rhythm and emotion
    y_harm, y_perc = librosa.effects.hpss(signal)

    # 7. Spectral Centroid
    # Note: Indicates where the ”centre of mass” for a sound is located and is calculated
    # as the weighted mean of the frequencies present in the sound.

    # Calculate the Spectral Centroids
    spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sr)[0]
    spectral_centroids_delta = librosa.feature.delta(spectral_centroids)
    spectral_centroids_accelerate = librosa.feature.delta(spectral_centroids, order=2)

    # spectral_centroid_feats = np.stack((spectral_centroids, delta, accelerate))  # (3, 64, xx)

    # 8. Chroma Frequencies¶
    # Note: Chroma features are an interesting and powerful representation
    # for music audio in which the entire spectrum is projected onto 12 bins
    # representing the 12 distinct semitones ( or chromas) of the musical octave.

    # Increase or decrease hop_length to change how granular you want your data to be
    hop_length = c.HOP_LENGTH

    # Chromogram
    chromagram = librosa.feature.chroma_stft(signal, sr=sr, hop_length=hop_length)

    # 9. Tempo BPM (beats per minute)¶
    # Note: Dynamic programming beat tracker.

    # Create Tempo BPM variable
    tempo_y, _ = librosa.beat.beat_track(signal, sr=sr)

    # 10. Spectral Rolloff
    # Note: Is a measure of the shape of the signal. It represents the frequency below which a specified
    #  percentage of the total spectral energy(e.g. 85 %) lies.

    # Spectral RollOff Vector
    spectral_rolloff = librosa.feature.spectral_rolloff(signal, sr=sr)[0]

    # spectral flux
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)

    # Spectral Bandwidth¶
    # The spectral bandwidth is defined as the width of the band of light at one-half the peak
    # maximum (or full width at half maximum [FWHM]) and is represented by the two vertical
    # red lines and λSB on the wavelength axis.
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(signal, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(signal, sr=sr, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(signal, sr=sr, p=4)[0]

    audio_features = {
        "file_name": audio_file_path,
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(signal)[0, 0],
        "zero_crossings": np.sum(librosa.zero_crossings(signal, pad=False)),
        "spectrogram": db_audio[0, 0],
        "mel_spectrogram": s_db_audio[0, 0],
        "harmonics": y_harm[0],
        "perceptual_shock_wave": y_perc[0],
        "spectral_centroids": spectral_centroids[0],
        "spectral_centroids_delta": spectral_centroids_delta[0],
        "spectral_centroids_accelerate": spectral_centroids_accelerate[0],
        "chroma1": chromagram[0, 0],
        "chroma2": chromagram[1, 0],
        "chroma3": chromagram[2, 0],
        "chroma4": chromagram[3, 0],
        "chroma5": chromagram[4, 0],
        "chroma6": chromagram[5, 0],
        "chroma7": chromagram[6, 0],
        "chroma8": chromagram[7, 0],
        "chroma9": chromagram[8, 0],
        "chroma10": chromagram[9, 0],
        "chroma11": chromagram[10, 0],
        "chroma12": chromagram[11, 0],
        "tempo_bpm": tempo_y,
        "spectral_rolloff": spectral_rolloff[0],
        "spectral_flux": onset_env[0],
        "spectral_bandwidth_2": spectral_bandwidth_2[0],
        "spectral_bandwidth_3": spectral_bandwidth_3[0],
        "spectral_bandwidth_4": spectral_bandwidth_4[0],
    }

    # extract mfcc feature
    mfcc_df = extract_mfcc_features(audio_file_path,
                                    signal,
                                    sample_rate=sr,
                                    number_of_mfcc=number_of_mfcc)

    df = pd.DataFrame.from_records(data=[audio_features])

    df = pd.merge(df, mfcc_df, on='file_name')

    return df

    # librosa.feature.mfcc(signal)[0, 0]


def extract_mfcc_features(audio_file_name: str,
                          signal: np.ndarray,
                          sample_rate: int,
                          number_of_mfcc: int) -> pd.DataFrame:
    # another MFCC approach
    # as suggested by https://github.com/Cocoxili/DCASE2018Task2/blob/master/data_transform.py,
    # https://arxiv.org/abs/1810.12832, and https://www.kaggle.com/c/freesound-audio-tagging
    mfcc_alt = librosa.feature.mfcc(y=signal, sr=sample_rate,
                                    n_mfcc=number_of_mfcc)
    delta = librosa.feature.delta(mfcc_alt)
    accelerate = librosa.feature.delta(mfcc_alt, order=2)

    mfcc_features = {
        "file_name": audio_file_name,
    }

    for i in range(0, number_of_mfcc):
        # dict.update({'key3': 'geeks'})

        # mfcc coefficient
        key_name = "".join(['mfcc', str(i)])
        mfcc_value = mfcc_alt[i, 0]
        mfcc_features.update({key_name: mfcc_value})

        # mfcc delta coefficient
        key_name = "".join(['mfcc_delta_', str(i)])
        mfcc_value = delta[i, 0]
        mfcc_features.update({key_name: mfcc_value})

        # mfcc accelerate coefficient
        key_name = "".join(['mfcc_accelerate_', str(i)])
        mfcc_value = accelerate[i, 0]
        mfcc_features.update({key_name: mfcc_value})



    df = pd.DataFrame.from_records(data=[mfcc_features])
    return df

# ref.: https://musicinformationretrieval.com/basic_feature_extraction.html
def extract_feature_means(audio_file_path: str) -> pd.DataFrame:
    # config settings
    number_of_mfcc = c.NUMBER_OF_MFCC

    # 1. Importing 1 file
    y, sr = librosa.load(audio_file_path)

    # Trim leading and trailing silence from an audio signal (silence before and after the actual audio)
    signal, _ = librosa.effects.trim(y)

    # 2. Fourier Transform
    # Default FFT window size
    n_fft = c.N_FFT  # FFT window size
    hop_length = c.HOP_LENGTH  # number audio of frames between STFT columns (looks like a good default)

    # Short-time Fourier transform (STFT)
    d_audio = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))

    # 3. Spectrogram
    # Convert an amplitude spectrogram to Decibels-scaled spectrogram.
    db_audio = librosa.amplitude_to_db(d_audio, ref=np.max)

    # 4. Create the Mel Spectrograms
    s_audio = librosa.feature.melspectrogram(signal, sr=sr)
    s_db_audio = librosa.amplitude_to_db(s_audio, ref=np.max)

    # 5 Zero crossings

    # #6. Harmonics and Perceptrual
    # Note:
    #
    # Harmonics are characteristichs that represent the sound color
    # Perceptrual shock wave represents the sound rhythm and emotion
    y_harm, y_perc = librosa.effects.hpss(signal)

    # 7. Spectral Centroid
    # Note: Indicates where the ”centre of mass” for a sound is located and is calculated
    # as the weighted mean of the frequencies present in the sound.

    # Calculate the Spectral Centroids
    spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sr)[0]
    spectral_centroids_delta = librosa.feature.delta(spectral_centroids)
    spectral_centroids_accelerate = librosa.feature.delta(spectral_centroids, order=2)

    # spectral_centroid_feats = np.stack((spectral_centroids, delta, accelerate))  # (3, 64, xx)

    # 8. Chroma Frequencies¶
    # Note: Chroma features are an interesting and powerful representation
    # for music audio in which the entire spectrum is projected onto 12 bins
    # representing the 12 distinct semitones ( or chromas) of the musical octave.

    # Increase or decrease hop_length to change how granular you want your data to be
    hop_length = c.HOP_LENGTH

    # Chromogram
    chromagram = librosa.feature.chroma_stft(signal, sr=sr, hop_length=hop_length)

    # 9. Tempo BPM (beats per minute)¶
    # Note: Dynamic programming beat tracker.

    # Create Tempo BPM variable
    tempo_y, _ = librosa.beat.beat_track(signal, sr=sr)

    # 10. Spectral Rolloff
    # Note: Is a measure of the shape of the signal. It represents the frequency below which a specified
    #  percentage of the total spectral energy(e.g. 85 %) lies.

    # Spectral RollOff Vector
    spectral_rolloff = librosa.feature.spectral_rolloff(signal, sr=sr)[0]

    # spectral flux
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)

    # Spectral Bandwidth¶
    # The spectral bandwidth is defined as the width of the band of light at one-half the peak
    # maximum (or full width at half maximum [FWHM]) and is represented by the two vertical
    # red lines and λSB on the wavelength axis.
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(signal, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(signal, sr=sr, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(signal, sr=sr, p=4)[0]

    audio_features = {
        "file_name": audio_file_path,
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(signal)[0]),
        "zero_crossings": np.sum(librosa.zero_crossings(signal, pad=False)),
        "spectrogram": np.mean(db_audio[0]),
        "mel_spectrogram": np.mean(s_db_audio[0]),
        "harmonics": np.mean(y_harm),
        "perceptual_shock_wave": np.mean(y_perc),
        "spectral_centroids": np.mean(spectral_centroids),
        "spectral_centroids_delta": np.mean(spectral_centroids_delta),
        "spectral_centroids_accelerate": np.mean(spectral_centroids_accelerate),
        "chroma1": np.mean(chromagram[0]),
        "chroma2": np.mean(chromagram[1]),
        "chroma3": np.mean(chromagram[2]),
        "chroma4": np.mean(chromagram[3]),
        "chroma5": np.mean(chromagram[4]),
        "chroma6": np.mean(chromagram[5]),
        "chroma7": np.mean(chromagram[6]),
        "chroma8": np.mean(chromagram[7]),
        "chroma9": np.mean(chromagram[8]),
        "chroma10": np.mean(chromagram[9]),
        "chroma11": np.mean(chromagram[10]),
        "chroma12": np.mean(chromagram[11]),
        "tempo_bpm": tempo_y,
        "spectral_rolloff": np.mean(spectral_rolloff),
        "spectral_flux": np.mean(onset_env),
        "spectral_bandwidth_2": np.mean(spectral_bandwidth_2),
        "spectral_bandwidth_3": np.mean(spectral_bandwidth_3),
        "spectral_bandwidth_4": np.mean(spectral_bandwidth_4),
    }

    # extract mfcc feature
    mfcc_df = extract_mfcc_feature_means(audio_file_path,
                                    signal,
                                    sample_rate=sr,
                                    number_of_mfcc=number_of_mfcc)

    df = pd.DataFrame.from_records(data=[audio_features])

    df = pd.merge(df, mfcc_df, on='file_name')

    return df

    # librosa.feature.mfcc(signal)[0, 0]

def extract_mfcc_feature_means(audio_file_name: str,
                          signal: np.ndarray,
                          sample_rate: int,
                          number_of_mfcc: int) -> pd.DataFrame:
    # another MFCC approach
    # as suggested by https://github.com/Cocoxili/DCASE2018Task2/blob/master/data_transform.py,
    # https://arxiv.org/abs/1810.12832, and https://www.kaggle.com/c/freesound-audio-tagging
    mfcc_alt = librosa.feature.mfcc(y=signal, sr=sample_rate,
                                    n_mfcc=number_of_mfcc)
    delta = librosa.feature.delta(mfcc_alt)
    accelerate = librosa.feature.delta(mfcc_alt, order=2)

    mfcc_features = {
        "file_name": audio_file_name,
    }

    for i in range(0, number_of_mfcc):
        # dict.update({'key3': 'geeks'})

        # mfcc coefficient
        key_name = "".join(['mfcc', str(i)])
        mfcc_value = np.mean(mfcc_alt[i])
        mfcc_features.update({key_name: mfcc_value})

        # mfcc delta coefficient
        key_name = "".join(['mfcc_delta_', str(i)])
        mfcc_value = np.mean(delta[i])
        mfcc_features.update({key_name: mfcc_value})

        # mfcc accelerate coefficient
        key_name = "".join(['mfcc_accelerate_', str(i)])
        mfcc_value = np.mean(accelerate[i])
        mfcc_features.update({key_name: mfcc_value})

    df = pd.DataFrame.from_records(data=[mfcc_features])
    return df


# Extracting Features from Sounds
# ref.: https://www.kaggle.com/andradaolteanu/birdcall-recognition-eda-and-audio-fe
def extract_audio_features_prototype(audio_file_path: str):
    # 1. Importing 1 file
    y, sr = librosa.load(audio_file_path)

    print('y:', y, '\n')
    print('y shape:', np.shape(y), '\n')
    print('Sample Rate (KHz):', sr, '\n')
    # Verify length of the audio
    print('Check Len of Audio:', 661794 / sr)

    # Trim leading and trailing silence from an audio signal (silence before and after the actual audio)
    audio_file, _ = librosa.effects.trim(y)

    # the result is an numpy ndarray
    print('Audio File:', audio_file, '\n')
    print('Audio File shape:', np.shape(audio_file))

    # 2. Fourier Transform
    # Default FFT window size
    n_fft = 2048  # FFT window size
    hop_length = 512  # number audio of frames between STFT columns (looks like a good default)

    # Short-time Fourier transform (STFT)
    d_audio = np.abs(librosa.stft(audio_file, n_fft=n_fft, hop_length=hop_length))

    # 3. Spectrogram

    # Convert an amplitude spectrogram to Decibels-scaled spectrogram.
    db_audio = librosa.amplitude_to_db(d_audio, ref=np.max)
    print("db_audio: ", db_audio)

    # 4. Create the Mel Spectrograms
    s_audio = librosa.feature.melspectrogram(audio_file, sr=sr)
    s_db_audio = librosa.amplitude_to_db(s_audio, ref=np.max)

    print("s_db_audio: ", s_db_audio)
    print("Len of s_db_audio:", len(s_db_audio))

    # #5. zero crossing rate
    # Note: the rate at which the signal changes from positive to negative or bac
    # Total zero_crossings in our 1 song
    zero_crossings = librosa.zero_crossings(audio_file, pad=False)
    print("Zero crossings:", zero_crossings)

    # #6. Harmonics and Perceptrual
    # Note:
    #
    # Harmonics are characteristichs that represent the sound color
    # Perceptrual shock wave represents the sound rhythm and emotion
    y_harm, y_perc = librosa.effects.hpss(audio_file)  # TODO: decide what to do with the values obtained
    print("Harmonics: ", y_harm)
    print("Perceptrual:", y_perc)

    # 7. Spectral Centroid
    # Note: Indicates where the ”centre of mass” for a sound is located and is calculated
    # as the weighted mean of the frequencies present in the sound.

    # Calculate the Spectral Centroids
    spectral_centroids = librosa.feature.spectral_centroid(audio_file, sr=sr)[0]
    # Shape is a vector
    print('Centroids:', spectral_centroids, '\n')
    print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')

    # 8. Chroma Frequencies¶
    # Note: Chroma features are an interesting and powerful representation
    # for music audio in which the entire spectrum is projected onto 12 bins
    # representing the 12 distinct semitones ( or chromas) of the musical octave.

    # Increase or decrease hop_length to change how granular you want your data to be
    hop_length = 5000

    # Chromogram
    chromagram = librosa.feature.chroma_stft(audio_file, sr=sr, hop_length=hop_length)
    print("Chromatogram: ", '\n')
    print(chromagram)
    print('\n')
    print('Chromogram shape:', chromagram.shape)

    # 9. Tempo BPM (beats per minute)¶
    # Note: Dynamic programming beat tracker.

    # Create Tempo BPM variable
    tempo_y, _ = librosa.beat.beat_track(audio_file, sr=sr)
    print("Tempo BPM: ", tempo_y)

    # 10. Spectral Rolloff
    # Note: Is a measure of the shape of the signal. It represents the frequency below which a specified
    #  percentage of the total spectral energy(e.g. 85 %) lies.

    # Spectral RollOff Vector
    spectral_rolloff = librosa.feature.spectral_rolloff(audio_file, sr=sr)[0]
    print("Spectral RollOff Vector: ", '\n')
    print(spectral_rolloff)

    S, phase = librosa.magphase(librosa.stft(audio_file))
    print("Another way to calculate the spectral roll-off:", '\n')
    print(librosa.feature.spectral_rolloff(S=S, sr=sr))

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio_file, sr=sr)
    print("MFCC: ", '\n')
    print(mfcc)
    print(len(mfcc))
    print(mfcc.shape)

    mfcc0 = librosa.feature.mfcc(y=audio_file, sr=sr)[0]
    print("MFCC0: ", mfcc0)

    # another MFCC approach
    # as suggested by https://github.com/Cocoxili/DCASE2018Task2/blob/master/data_transform.py,
    # https://arxiv.org/abs/1810.12832, and https://www.kaggle.com/c/freesound-audio-tagging
    mfcc_alt = librosa.feature.mfcc(y=audio_file, sr=sr,
                                    n_mfcc=20)
    delta = librosa.feature.delta(mfcc_alt)
    accelerate = librosa.feature.delta(mfcc_alt, order=2)

    feats = np.stack((mfcc_alt, delta, accelerate))  # (3, 64, xx)

    print("Alternative MFCC:")
    print("Dimensions:")
    print(mfcc_alt.shape)
    print("-----------------------------------")
    print("Stacked values:")
    print(feats)

    # spectral flux
    onset_env = librosa.onset.onset_strength(y=audio_file, sr=sr)
    print("Spectral flux:", '\n')
    print(onset_env)

    # pitches
    pitches, magnitudes = librosa.piptrack(y=audio_file, sr=sr)
    print("pitches:", '\n')
    print(pitches)

    # Spectral Bandwidth¶
    # The spectral bandwidth is defined as the width of the band of light at one-half the peak
    # maximum (or full width at half maximum [FWHM]) and is represented by the two vertical
    # red lines and λSB on the wavelength axis.
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(audio_file, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(audio_file, sr=sr, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(audio_file, sr=sr, p=4)[0]

    print("Spectral bandwidth:")
    print('-----------------------------')
    print(spectral_bandwidth_2)
    print('-----------------------------')
    print(spectral_bandwidth_3)
    print('-----------------------------')
    print(spectral_bandwidth_4.shape)
    print(spectral_bandwidth_4)

    experimental_feature_list = extract_features(audio_file_path)
    print("experimental_FeatureList:")
    print(experimental_feature_list.head())
    print(experimental_feature_list.info())

    print("Selective df features:")
    print(experimental_feature_list['spectral_centroids'])
    print(experimental_feature_list['spectral_centroids_delta'])
    print(experimental_feature_list['spectral_centroids_accelerate'])