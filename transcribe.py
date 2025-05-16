from utils import extract_audio
from utils import add_helper_files
from utils import locations
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

default_recognizer_dir = '../sampa_dutch_960_100000/best/'

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
    timestamp_type = 'word'):
    '''
    transcribe an audio file with pipeline object
    loads the audio with librosa
    '''
    a = extract_audio.load_audio(filename,start=start,end=end)
    output = pipeline(a, return_timestamps = timestamp_type)
    return output


def _make_decoding_output_filename(audio_filename, output_dir, extension):
    '''
    make a filename for transcription based on audio_filename
    and output_dir.
    '''
    filename = audio_filename.replace('.wav',extension)
    filename = output_dir + filename.split('/')[-1]
    return filename

def save_pipeline_output_to_files(output,audio_filename,output_dir=''):
    table = pipeline_output2table(output)
    ctm = pipeline_output2ctm(output,audio_filename)
    save(_table2str(table),audio_filename,'.table',output_dir)
    save(_table2str(ctm,sep = ' '),audio_filename,'.ctm',output_dir)
    save(output['text'],audio_filename,'.txt',output_dir)

def save(t, audio_filename, extension, output_dir = ''):
    '''save pipeline output to a file.'''
    filename= _make_decoding_output_filename(audio_filename, output_dir,
        extension)
    print('saving to:',filename)
    try:
        with open(filename,'w') as fout:
            fout.write(t)
    except PermissionError:
        print('could not write file to', filename, 'due to a permission error')

def pipeline_output2table(output):
    '''convert pipeline output to table (word\tstart\tend).'''
    table = []
    for d in output['chunks']:
        start, end = d['timestamp']
        table.append([d['text'], start, end])
    return table

def stem_filename(filename):
    if '/' in filename: filename = filename.split('/')[-1]
    if '.' in filename:
        filename= filename.split('.')[:-1]
        if len(filename) > 1: filename = '.'.join(filename)
        else: filename = filename[0]
    return filename

def pipeline_output2ctm(output, filename):
    filename = stem_filename(filename)
    table = pipeline_output2table(output)
    ctm = []
    for line in table:
        word, start, end = line
        duration = round(end - start,2)
        line = [filename,1,start,duration,word,'1.00']
        ctm.append(line)
    return ctm


def _table2str(table, sep = '\t'):
    '''convert output table to string.'''
    output = []
    for line in table:
        output.append(sep.join(list(map(str,line))))
    return '\n'.join(output)

class Transcriber:
    '''transcribe audio files in input_dir or the audio file filename.'''
    def __init__(self, model_dir = None, input_dir = None, output_dir = None,
        model = None, pipeline = None, device = -1, filename = '',
        timestamp_type = 'word'):
        '''transcribe audio files in input_dir
        model_dir       directory of the wav2vec2 model
        input_dir       directory for audio files that need to be transcribed
        output_dir      directory for output files
        '''
        self.model_dir = model_dir if model_dir else default_recognizer_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = device
        self.filename = filename
        self.timestamp_type = timestamp_type
        if pipeline: self.pipeline = pipeline
        elif model:
            self.model = model
            self.pipeline = load_pipeline(model = model,device = device)
        else: self.pipeline = load_pipeline(recognizer_dir = self.model_dir,
            device = device)
        self.transcribed_audio_files = {}
        self.did_transcription= False

    def load_audio_filenames(self):
        self.ok = True
        if self.input_dir:
            self.audio_filenames = glob.glob(self.input_dir + '*.wav')
        elif self.filename:
            self.audio_filenames = [self.filename]
        else: self.ok = False
        m ='transcribed audio files'
        m += ' '.join(self.transcribed_audio_files.keys())

    def transcribe(self):
        self.load_audio_filenames()
        self.did_transcription= False
        for filename in self.audio_filenames:
            if filename not in self.transcribed_audio_files.keys():
                print('transcribing: ',filename)
                try: o = decode_audiofile(filename, self.pipeline,
                    timestamp_type = self.timestamp_type)
                except ValueError: return
                save_pipeline_output_to_files(o, filename,self.output_dir)
                self.transcribed_audio_files[filename] = o
                self.did_transcription = True









def transcribe(args):
    device, input_dir, output_dir, filename = pre_checks(args)
    if args.keep_alive_minutes == None: args.keep_alive_minutes = 0
    keep_alive_seconds = args.keep_alive_minutes * 60
    timestamp_type = 'char' if args.label_timestamps else 'word'
    print('using timestamp type:',timestamp_type)
    print('loading transcriber')
    transcriber = Transcriber(args.model_dir, input_dir, output_dir,
        device = device, filename = args.filename,
        timestamp_type = timestamp_type)
    if not _check_transcriber_ok(transcriber): return
    print('start transcribing')
    last_transcription = time.time()
    while True:
        transcriber.transcribe()
        if transcriber.did_transcription:
            last_transcription = time.time()
        time.sleep(1)
        if time.time() - last_transcription > keep_alive_seconds:
            break
    print('closing down transcriber')
    return transcriber.transcribed_audio_files


def transcribe_ifadv_item(item, pipeline):
    filename = locations.ifadv_wav_16khz_dir + item['filename']
    start = item['start_time']
    end = item['end_time']
    output = decode_audiofile(filename, pipeline, start = start, end = end,
        timestamp_type = None)
    item['hyp'] = output['text']
    return item
