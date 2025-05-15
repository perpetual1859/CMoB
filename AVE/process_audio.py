import csv
import os

import numpy as np
import torchaudio
import torch

## save path of processed spectrogram
save_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS-SE'

## file path of wav files
audio_path='/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audios'


## the list of all wav files
csv_file = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Annotations.txt'


data = []
with open(csv_file) as f2:
  files = f2.readlines()
  print(len(files))
  for item in files:
      item = item.split('&')
      name = item[1]
      start_time = item[3]
      end_time = item[4]
      print(name)

      if os.path.exists(audio_path + '/' + name + '.wav'):
        data.append(name)
        # print(name)
        # exit(0)

for name in data:
  waveform, sr = torchaudio.load(audio_path + '/'+ name + '.wav')
  # if waveform.dim() > 1:
  #     waveform = waveform.mean(dim=0, keepdim=True)
  # sr =16000

  waveform = waveform - waveform.mean()
  norm_mean = -4.503877
  norm_std = 5.141276

  #设计SE
  start_time = int(start_time)
  end_time = int(end_time)
  start_frame = int(start_time * sr)
  end_frame = int(end_time * sr)
  waveform_segment = waveform[:,start_frame:end_frame]

  fbank = torchaudio.compliance.kaldi.fbank(waveform_segment, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

  # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
  #                                           window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

  target_length = 1024
  n_frames = fbank.shape[0]
  # print(n_frames)
  p = target_length - n_frames

  # cut and pad
  if p > 0:
      m = torch.nn.ZeroPad2d((0, 0, 0, p))
      fbank = m(fbank)
  elif p < 0:
      fbank = fbank[0:target_length, :]
  fbank = (fbank - norm_mean) / (norm_std * 2)

  print(fbank.shape)
  np.save(save_path + '/'+ name + '.npy',fbank)
