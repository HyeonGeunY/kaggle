import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import collections import Counter
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from typing import List

sns.set(style="ticks", context="talk",font_scale = 1.2)
plt.style.use("seaborn-paper")
plt.subplots_adjust(wspace=1)

ROOTDIR = ""

class Dataviz():
    def __init__(self, sample_rate, n_fft, hop_length, n_mels):
        self.root_dir = ROOTDIR
        
        # edit metadata
        self.meta = pd.read_csv(fr"{self.root_dir}/train_metadata.csv")
        self.meta['type'] = self.meta['type'].apply(lambda x : ast.literal_eval(x))
        #
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, 
                                                              n_fft=self.n_fft, 
                                                              hop_length=self.hop_length, 
                                                              n_mels=self.n_mels)
    
    
    def feature_distribution(self, feature: str):
        plt.figure(figsize=(20, 6))

        sns.countplot(self.meta[feature])
        plt.xticks(rotation=90)
        plt.title(f"Distribution of {feature} Labels", fontsize=20)

        plt.show()
        
    
    def most_common(self, feature: str, k: int):
        
        if isinstance(self.meta[feature].iloc[0], list):
    
            top = Counter([typ.lower() for lst in self.meta[feature] 
                                           for typ in lst])

            top = dict(top.most_common(k))

            plt.figure(figsize=(20, 6))

            sns.barplot(x=list(top.keys()), y=list(top.values()), palette='hls')
            plt.title(f"Top {k} song {feature}")
            
        else:
            top = Counter([f.lower() for f in self.meta[feature]])

            top = dict(top.most_common(k))

            plt.figure(figsize=(20, 6))

            sns.barplot(x=list(top.keys()), y=list(top.values()), palette='hls')
            plt.title(f"Top {k} song {feature}")

        plt.show()
            
    
    def waveform(self, indices: List[str]):
        if isinstance(indices, int):
            indices = [indices]
        n = len(indices)
                                  
        if n == 1:    
            idx = indices[0]
            fig, ax = plt.subplots(1, 1, figsize=(20, 5))
            fig.suptitle("Sound Waves", fontsize=15)
            
            signal_1, sample_rate = torchaudio.load(f"{self.root_dir}/train_audio/{self.meta.filename.iloc[idx]}")
            # The audio data consist of two things-
            # Sound: sequence of vibrations in varying pressure strengths (y)
            # Sample Rate: (sample_rate) is the number of samples of audio carried per second, measured in Hz or kHz

            sns.lineplot(x=np.arange(len(signal_1[0,:].detach().numpy())), y=signal_1[0,:].detach().numpy(), ax=ax, color='#4400FF')
            ax.set_title(f"Audio {idx}")
        
        elif n > 1:
            fig, ax = plt.subplots(n, 1, figsize=(20, 5 * n))
            fig.suptitle("Sound Waves", fontsize=15)
            
            for i, idx in enumerate(indices):

                signal, sample_rate = torchaudio.load(f"{self.root_dir}/train_audio/{self.meta.filename.iloc[idx]}")
                # The audio data consist of two things-
                # Sound: sequence of vibrations in varying pressure strengths (y)
                # Sample Rate: (sample_rate) is the number of samples of audio carried per second, measured in Hz or kHz

                sns.lineplot(x=np.arange(len(signal[0,:].detach().numpy())), y=signal[0,:].detach().numpy(), ax=ax[i], color='#4400FF')
                ax[i].set_title(f"Audio {idx}")

        else:
            print("n should be more than or equal to 1")
            assert n >= 1
            
        plt.show()
        
    
    def mel_spectrogram(self, indices: List[str]):
        
        if isinstance(indices, int):
            indices = [indices]
            
        n = len(indices)
                                  
        if n == 1:    
            idx = indices[0]
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            fig.suptitle("Mel Spectrogram", fontsize=15)
            
            signal, sample_rate = torchaudio.load(f"{self.root_dir}/train_audio/{self.meta.filename.iloc[idx]}")
            # The audio data consist of two things-
            # Sound: sequence of vibrations in varying pressure strengths (y)
            # Sample Rate: (sample_rate) is the number of samples of audio carried per second, measured in Hz or kHz
            
            mel = self.mel_converter(signal)

            ax.imshow(mel.log2()[0,:,:].detach().numpy(), aspect='auto', cmap='cool')
            ax.set_title(f"Audio {idx}")
                    
        elif n > 1:
            fig, ax = plt.subplots(n, 1, figsize=(10,  7 * n))
            fig.suptitle("Mel Spectrogram", fontsize=15)
            
            for i, idx in enumerate(indices):

                signal, sample_rate = torchaudio.load(f"{self.root_dir}/train_audio/{self.meta.filename.iloc[idx]}")
                # The audio data consist of two things-
                # Sound: sequence of vibrations in varying pressure strengths (y)
                # Sample Rate: (sample_rate) is the number of samples of audio carried per second, measured in Hz or kHz

                mel = self.mel_converter(signal)
                ax[i].imshow(mel.log2()[0,:,:].detach().numpy(), aspect='auto', cmap='cool')
                ax[i].set_title(f"Audio {idx}")

        else:
            print("n should be more than or equal to 1")
            assert n >= 1
            
        plt.show()
            
        
        
class Dataset(Dataset, Dataviz):
    
    def __init__(self, root_dir, sample_rate, n_fft, hop_length, n_mels, duration):
        self.root_dir = ROOTDIR
        self.meta = pd.read_csv(fr"{self.root_dir}/train_metadata.csv")
        
        self.meta['type'] = self.meta['type'].apply(lambda x : ast.literal_eval(x))
        encoder = LabelEncoder()
        self.meta['primary_label_encoded'] = encoder.fit_transform(self.meta['primary_label'])
        
        self.audio_paths = self.root_dir + "/" + "train_audio" + "/" + self.meta.filename.values
        self.labels = self.meta.primary_label_encoded.values
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.num_samples = sample_rate * duration
        
        self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, 
                                                              n_fft=self.n_fft, 
                                                              hop_length=self.hop_length, 
                                                              n_mels=self.n_mels)
        
    def __len__(self):
        return len(self.audio_paths)
    
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        signal, sample_rate = torchaudio.load(audio_path) # loaded the audio
        
        # Now we first checked if the sample rate is same as TARGET_SAMPLE_RATE and if it not equal we perform resampling
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            signal = resampler(signal)
        
        # Next we check the number of channels of the signal
        #signal -> (num_channels, num_samples) - Eg.-(2, 14000) -> (1, 14000)
        if signal.shape[0]>1:
            signal = torch.mean(signal, axis=0, keepdim=True)
        
        # Lastly we check the number of samples of the signal
        #signal -> (num_channels, num_samples) - Eg.-(1, 14000) -> (1, self.num_samples)
        # If it is more than the required number of samples, we truncate the signal
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        
        # If it is less than the required number of samples, we pad the signal
        if signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
        
        # Finally all the process has been done and now we will extract mel spectrogram from the signal
        mel = self.mel_converter(signal)
        
        # For pretrained models, we need 3 channel image, so for that we concatenate the extracted mel
        image = torch.cat([mel, mel, mel])
        
        # Normalized the image
        max_val = torch.abs(image).max()
        image = image / max_val
        
        label = torch.tensor(self.labels[idx])
        
        return image, label