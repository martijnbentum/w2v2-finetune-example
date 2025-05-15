import data_preprocessing
from transformers import Wav2Vec2ForCTC

def load_model(model_name = "facebook/wav2vec2-xls-r-300m", 
    vocab_filename = 'vocab.json', cache_dir = '../example_cache_dir'):
    ''' load the model used for finetuning.
    model_name: the name of the model to load (currently set to a model on
                huggingface if you want to use a local model you should
                set it to the directory with config.json and pytorch_model.bin
                or model.safetensors)
    vocab_filename: the name of the vocabulary file (default is vocab.json)
    cache_dir: the directory to cache the model in 
               (default is ../example_cache_dir)
    '''
    processor = data_preprocessing.load_processor(vocab_filename)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        cache_dir = cache_dir,
    )
    model.freeze_feature_encoder() # freezes the CNN block
    return model
