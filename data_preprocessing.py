import data_loading
import json
import matplotlib.pyplot as plt
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC

processor = None

def load_tokenizer(vocab_filename = 'vocab.json'):
    ''' load the tokenizer used for finetuning.'''
    tokenizer = Wav2Vec2CTCTokenizer(vocab_filename, unk_token = '[UNK]', 
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

    item['labels'] = processor(text = item['sentence']).input_ids
    return item

def preprocess_dataset(dataset = None, vocab_filename = 'vocab.json'):
    ''' load the dataset and preprocess it.'''
    if dataset is None:
        dataset = data_loading.load_dataset_with_audio()
    _ = load_processor(vocab_filename)
    dataset = dataset.map(_preprocess_item,
        remove_columns = dataset.column_names)
    return dataset

#inspect data

def load_vocab(vocab_filename = 'vocab.json'):
    ''' load the vocabulary used for finetuning.'''
    with open(vocab_filename) as fin:
        vocab = json.load(fin)
    return vocab

def input_ids_to_text(input_ids, vocab_filename = 'vocab.json'):
    ''' convert the input ids to text.'''
    processor = load_processor(vocab_filename)
    temp = processor.batch_decode(input_ids, skip_special_tokens=True)
    text = ''
    for char in temp:
        if char: text += char
        else: text += ' '
    return text

def plot_array(input_values):
    ''' plot the array.'''
    plt.ion()
    plt.plot(input_values)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized amplitude')
    xticks = plt.gca().get_xticks()
    plt.xticks(xticks, xticks / 16000)
    plt.xlim(0, len(input_values)) 
    plt.show()

def show_item(example_index = 0, dataset = None, vocab_filename = 'vocab.json'):  
    ''' show the item at example_index.'''
    if dataset is None:
        dataset = preprocess_dataset()
    example = dataset[example_index]
    plot_array( example['input_values'] )
    text = input_ids_to_text(example['labels'], vocab_filename)
    print(f'label: {example["labels"]}')
    print(f'text:  {''.join(text)}')
    print(f'vocab: {load_vocab()}')
    
    

