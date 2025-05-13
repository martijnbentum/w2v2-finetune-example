import data_loading
import json
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC

processor = None

def load_vocab(vocab_filename = 'vocab.json'):
    ''' load the vocabulary used for finetuning.'''
    with open(vocab_filename) as fin:
        vocab = json.load(fin)
    return vocab

def load_tokenizer(vocab_filename = 'vocab.json'):
    ''' load the tokenizer used for finetuning.'''
    vocab = load_vocab(vocab_filename)
    tokenizer = Wav2Vec2CTCTokenizer(vocab, unk_token = '[UNK]', 
        pad_token = '[PAD]',word_delimiter_token = '|')
    return tokenizer

def load_feature_extractor():
    ''' load the feature extractor used to preprocess audio.'''
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
        sampling_rate=16000, padding_value=0.0, do_normalize=True,
        return_attention_mask=True)
    return feature_extractor

def load_processor(vocab_filename = 'vocab.json',):
    global processor
    tokenizer = load_tokenizer(vocab_filename)
    feature_extractor = load_feature_extractor()
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
        tokenizer=tokenizer)
    return processor

def _preprocess_item(item):
    ''' map function to preprocess the dataset for finetuning.
    normalizes the audio samples and maps the transcription to the
    vocabulary ids used for finetuning.
    '''
    audio = item['audio']

    item['input_values'] = processor(audio['array'],
        sampling_rate = audio['sampling_rate']).input_values[0]
    item['input_length'] = len(item['input_values'])

    with processor.as_target_processor():
        item['labels'] = processor(item['sentence']).input_ids
    return item

def preprocess_dataset(dataset = None, vocab_filename = 'vocab.json'):
    ''' load the dataset and preprocess it.'''
    if dataset is None:
        dataset = data_loading.load_dataset()
    processor = load_processor(vocab_filename)
    dataset = list(map(_preprocess_item, dataset))
    return dataset

