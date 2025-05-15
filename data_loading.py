from datasets import load_dataset
import matplotlib.pyplot as plt
import librosa
import json
import sounddevice as sd


# load data in datasets format

def _load_dataset_from_json(filename = 'example.json'):
    '''loads a json file in the datasets format.'''
    dataset = load_dataset('json', data_files=filename, field='data',
        cache_dir = '../example_cache_dir')
    return dataset

def load_dataset_with_audio(dataset = None):
    '''loads the audio data in the dataset.'''
    if dataset is None:
        dataset = _load_dataset_from_json()
    dataset = dataset.map(_load_audio_item)
    return dataset['train']

# inspect data

def load_json(filename = 'example.json'):
    '''loads the json file with audio and transcription info.'''
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def play_example(example_index = 0, filename = 'example.json', plot = False):
    '''play audio and show transcription for an item.'''
    data = load_json(filename)
    example = data['data'][example_index]
    audio = _load_audio(example)
    if plot: plot_audio(audio)
    print(f'Playing example {example_index} of {len(data["data"])}')
    print(f'sentence: {example["sentence"]}')
    sd.play(audio, 16000)
    sd.wait()

def plot_audio(audio):
    '''visualize the audio signal.'''
    plt.ion()
    plt.plot(audio)
    xticks = plt.gca().get_xticks()
    plt.xticks(xticks, xticks / 16000)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(0, len(audio)) 
    plt.show()
    

# audio loading helper functions

def load_audio(filename, start = 0.0, end = None, channel = None):
    '''loads the audio file with librosa. Resamples the file to 16kHz.
    filename: the filename of the audio file
    start: the start time in seconds
    end: the end time in seconds
    channel: the channel to load (1-indexed) -> channel can be 1 or 2
             if audio is mono set channel to None
             if you want remix audio to mono set channel to None
    '''
    if not end: duration = None
    else: duration = end - start
    if channel is None:
        audio, sr = librosa.load(filename, sr = 16000, offset = start,
            duration = duration)
    else:
        audio, sr = librosa.load(filename, sr = 16000, offset = start,
            duration = duration, mono = False)
        audio = audio[channel - 1]
    return audio

def _load_audio(item):
    '''uses the metadata in the json file to load the audio.'''
    audiofilename = item['filename']
    start = item['start_time']
    end = item['end_time']
    channel = item['channel']
    audio = load_audio(audiofilename, start = start, end = end, 
        channel = channel)
    return audio

def _load_audio_item(item):
    '''map helper function to load audio in the datasets format.'''
    filename = item['filename']
    item['audio'] = {}
    item['audio']['array'] = _load_audio(item)
    item['audio']['sampling_rate'] = 16000
    return item

