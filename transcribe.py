import data_loading
import add_helper_files
import glob
import os
import time
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ProcessorWithLM


'''
More info about pipelines for ASR see:
https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline
'''
from transformers import AutomaticSpeechRecognitionPipeline as ap

default_recognizer_dir = '../example/checkpoint-200/'

def load_model(recognizer_dir = default_recognizer_dir):
    model = Wav2Vec2ForCTC.from_pretrained(recognizer_dir)
    return model

def load_processor(recognizer_dir = default_recognizer_dir):
    processor = Wav2Vec2Processor.from_pretrained(recognizer_dir)
    return processor

def load_pipeline(recognizer_dir=None, model = None,chunk_length_s = 10,
    device = -1, copy_helper_files = False):
    '''
    loads a pipeline object that can transcribe audio files
    recognizer_dir      directory that stores the wav2vec2 model
    model               preloaded wav2vec model (speeds up loading pipeline)
    chunk_length_s      chunking duration of long audio files
                        wav2vec2 is memory hungry, the pipeline employs
                        a sliding window to handle long audio files
                        and the edge effects from chunking
    '''
    print('using device:',device)
    if not recognizer_dir: recognizer_dir = default_recognizer_dir
    print('using recognizer_dir:',recognizer_dir)
    if copy_helper_files:
        add_helper_files.add_helper_files(recognizer_dir)
    if not model:
        print('loading model:',recognizer_dir)
        model = load_model(recognizer_dir)
    print('loading processor')
    p= load_processor(recognizer_dir)
    print('loading pipeline')
    pipeline = ap(
        feature_extractor =p.feature_extractor,
        model = model,
        tokenizer = p.tokenizer,
        chunk_length_s = chunk_length_s,
        device = device
    )
    return pipeline


def decode_audiofile(filename, pipeline, start=0.0, end=None,
    timestamp_type = 'word', channel = None):
    '''
    transcribe an audio file with pipeline object
    loads the audio with librosa
    '''
    a = data_loading.load_audio(filename,start=start,end=end)
    output = pipeline(a, return_timestamps = timestamp_type)
    return output


def transcribe_ifadv_item(item, pipeline):
    filename = item['filename']
    start = item['start_time']
    end = item['end_time']
    channel = item['channel']
    output = decode_audiofile(filename, pipeline, start = start, end = end,
        timestamp_type = None, channel = channel)
    item['hyp'] = output['text']
    return item
