import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# Loading the audio file
audio_file = glob('test.wav')[0]  # assumes 'test.wav' is present in the directory

# Use librosa to load the audio
audio, sr = librosa.load(audio_file)

# Play audio in Jupyter/IPython
ipd.display(ipd.Audio(audio, rate=sr))

# Split silent and non-silent intervals
non_silent_intervals = librosa.effects.split(audio, top_db=20)
print(non_silent_intervals)
