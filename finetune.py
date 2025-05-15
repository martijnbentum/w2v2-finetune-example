import data_collator
import data_preprocessing
import evaluate
import load_model
import numpy as np
from pathlib import Path
from transformers import TrainingArguments
from transformers import Trainer

wer_metric = evaluate.load('wer')

def load_training_arguments(experiment_name = '../example/', 
    num_train_epochs = 200, warmup_steps = 100, learning_rate = 3e-4,
    per_device_train_batch_size = 6, eval_steps =30, save_steps = 30,
    group_by_length = True, fp16= False, use_cpu = True):
    '''load the training arguments for finetuning.
    experiment_name: the name of the experiment (used to save the model)
    num_train_epochs: the number of epochs to train for (epoch is a complete
                      pass through the dataset)
    warmup_steps: the number of steps to warm up the learning rate from
                  0 to learning_rate. A typical value is between 5-10% of
                  num_train_steps
    learning_rate: the learning rate to use for finetuning, values between
                   3e-4 - 5e-5 (i.e. 0.0003 - 0.00005) are common
    per_device_train_batch_size: the batch size per device to use for finetuning
                batch size = per_device_train_batch_size * num_gpus
    eval_steps: the number of steps to evaluate the model on the validation set
    save_steps: the number of steps to finetune before saving the model
    group_by_length: whether to group the data by length or not
                speeds up training by reducing padding
    fp16:   normally set to True, but training on CPU is not supported
            Whether to use fp16 16-bit (mixed) precision training instead 
            of 32-bit training. (can only be used with GPUs) 
    use_cpu: whether to use cpu or not (set to True if you want to train on
            CPU, set to False if you want to train on GPU)
    '''
    path = Path(experiment_name)
    if not path.exists(): path.mkdir(parents=True, exist_ok=True)
    print('learning rate:', learning_rate)
    training_args = TrainingArguments(
        output_dir=experiment_name,
        group_by_length=group_by_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        fp16=fp16,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_total_limit=3,
        push_to_hub=False,
        use_cpu= use_cpu,
        report_to=["wandb"]
    )
    return training_args

def load_trainer():
    '''load the trainer for finetuning.'''
    processor = data_preprocessing.load_processor()
    dc = data_collator.DataCollatorCTCWithPadding(
        processor = processor, padding = True)
    training_args = load_training_arguments()
    model = load_model.load_model()
    dataset = data_preprocessing.preprocess_dataset()
    trainer = Trainer(
        model = model,
        args = training_args,
        compute_metrics = compute_metrics,
        data_collator = dc,
        train_dataset = dataset,
        eval_dataset = dataset,
        tokenizer = processor.feature_extractor,
    )
    return trainer


def compute_metrics(pred):
    processor = data_preprocessing.load_processor()
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis = -1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
