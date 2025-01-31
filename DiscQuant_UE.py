import torch
import pandas as pd
import re
import spacy
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
import torchtext
import time
from tqdm import tqdm
import argparse
import csv

# nlp library of Pytorch
from torchtext import data

import warnings as wrn
wrn.filterwarnings('ignore')

from models import LSTMNet
import site
import os
os.environ['SP_DIR'] = '/opt/conda/lib/python3.11/site-packages'

# Argument parsing
parser = argparse.ArgumentParser(description="evaluateence Script")

parser.add_argument('--bit_width', type=int, default=8, help='bit width for quantization')
parser.add_argument('--dataset', type=str, choices=['ag_news', 'dbpedia'], required=True, help='Dataset name (e.g., ag_news, dbpedia)')
parser.add_argument('--cuda', type=int, default=0, help='CUDA device number (e.g., 0, 1, 2)')
parser.add_argument('--class_dist', type=float, nargs='+', help='Class distribution as a list of floats (e.g., 1.0 0 0 0)', default= [0.25, 0.25, 0.25, 0.25])

args = parser.parse_args()


SEED = 2021

torch.manual_seed(SEED)
torch.backends.cuda.deterministic = True

# def clean_text(text):
#     cleaned_text = re.sub(r'[^A-Za-z0-9]+', ' ', str(text))
#     return cleaned_text

# # Load and preprocess the data files
# def load_and_preprocess(file_path):
#     df = pd.read_csv(file_path, header=None, delimiter='\t') # Assuming tab-separated values in .data files
#     df[1] = df[1].apply(clean_text) # Assuming the text is in the second column
#     cleaned_file_path = file_path.replace('.data', '_cleaned.data')
#     df.to_csv(cleaned_file_path, index=False, header=False)
#     return cleaned_file_path

# cleaned_train_file = load_and_preprocess('./data/ag_news/data/train.data')
# cleaned_valid_file = load_and_preprocess('./data/ag_news/data/valid.data')
# cleaned_test_file = load_and_preprocess('./data/ag_news/data/test.data')

cleaned_train_file = 'data/ag_news/train_clean.csv'
cleaned_valid_file = 'data/ag_news/valid_clean.csv'
cleaned_test_file = 'data/ag_news/test_clean.csv'

spacy_en = spacy.load('en_core_web_sm')

def spacy_tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

LABEL = data.LabelField()
TEXT = data.Field(tokenize=spacy_tokenizer, batch_first=True, include_lengths=True)
fields = [("label", LABEL), ("text", TEXT)]

training_data = data.TabularDataset(path=cleaned_train_file, format="csv", fields=fields, skip_header=True)
validation_data = data.TabularDataset(path=cleaned_valid_file, format="csv", fields=fields, skip_header=True)
test_data = data.TabularDataset(path=cleaned_test_file, format="csv", fields=fields, skip_header=True)

print(vars(training_data.examples[0]))

train_data,valid_data = training_data, validation_data

TEXT.build_vocab(train_data,
                 min_freq=5)

LABEL.build_vocab(train_data)
# Count the number of instances per class
label_counts = {LABEL.vocab.itos[i]: LABEL.vocab.freqs[LABEL.vocab.itos[i]] for i in range(len(LABEL.vocab))}
print("Number of instances per class:", label_counts)

print("Size of text vocab:",len(TEXT.vocab))

print("Size of label vocab:",len(LABEL.vocab))

TEXT.vocab.freqs.most_common(10)

# Creating GPU variable
device = torch.device(f'cuda:{args.cuda}')
#device = torch.device('cuda')
print(f'Using device: {device}')

BATCH_SIZE=32
print("Batch size initialized to", BATCH_SIZE)

# Count the number of instances per class
label_counts = {LABEL.vocab.itos[i]: LABEL.vocab.freqs[LABEL.vocab.itos[i]] for i in range(len(LABEL.vocab))}
print("Number of instances per class:", label_counts)

from collections import defaultdict
import random

def sample_subset_by_class(dataset, label_field, fraction=0.3, seed=2021):
    """
    dataset: TorchText Dataset (e.g., train_data)
    label_field: The corresponding LabelField object (e.g., LABEL)
    fraction: Fraction of the data to sample from each class
    seed: Random seed for reproducibility
    """
    random.seed(seed)  # for reproducible sampling
    
    # Group examples by label
    label_to_examples = defaultdict(list)
    for ex in dataset.examples:
        # ex.label is numeric, e.g. 0, 1, 2, ...
        # If you want to group by string label, do:
        # label_str = label_field.vocab.itos[ex.label]
        # and then use label_str as the key
        label_idx = ex.label
        label_to_examples[label_idx].append(ex)
    
    # For each class, sample a fraction of the examples
    sampled_examples = []
    for label_idx, examples_list in label_to_examples.items():
        n = len(examples_list)
        sample_size = max(1, int(n * fraction))  # ensure at least 1 sample
        sampled = random.sample(examples_list, sample_size)
        sampled_examples.extend(sampled)
    
    # Create a new TorchText Dataset from these sampled examples
    # (the dataset constructor takes a list of Examples and a dict of fields)
    new_dataset = data.Dataset(sampled_examples, fields={
        "label": label_field,
        "text": TEXT  # Or whatever your text field is named
    })
    
    return new_dataset

def sample_uneven_by_class(dataset, label_field, global_fraction, class_distribution, seed=2021):

    random.seed(seed)
    
    total_size = len(dataset)
    total_sample_size = int(total_size * global_fraction)
    
    # Group examples by *numeric label index*
    label_to_examples = defaultdict(list)
    for ex in dataset.examples:
        # Typically ex.label is a *string*, so convert to numeric index
        label_str = ex.label
        numeric_idx = label_field.vocab.stoi[label_str]
        label_to_examples[numeric_idx].append(ex)
    
    # number of classes based on your specified distribution
    num_classes = len(class_distribution)

    # Alternatively, you could infer the number of distinct labels from the dataset:
    # num_classes_in_data = len(label_to_examples.keys())

    # Check for mismatch
    if num_classes != len(label_to_examples):
        raise ValueError(
            f"Mismatch in # of classes: distribution has {num_classes} entries, "
            f"but data has {len(label_to_examples)} distinct labels."
        )
    
    # Compute how many examples from each class we *want* to sample
    samples_per_class = []
    for i in range(num_classes):
        desired_count = int(class_distribution[i] * total_sample_size)
        samples_per_class.append(desired_count)
    
    # Because of integer rounding, sum might not match total_sample_size
    sum_samples = sum(samples_per_class)
    diff = total_sample_size - sum_samples
    
    # Distribute leftover or shortage due to rounding
    idx = 0
    while diff > 0:
        samples_per_class[idx] += 1
        diff -= 1
        idx = (idx + 1) % num_classes
    while diff < 0:
        if samples_per_class[idx] > 0:
            samples_per_class[idx] -= 1
            diff += 1
        idx = (idx + 1) % num_classes
    
    # Now sample from each class
    sampled_examples = []
    for class_idx in range(num_classes):
        examples_list = label_to_examples[class_idx]
        to_sample = min(samples_per_class[class_idx], len(examples_list))
        
        # random.sample raises an error if to_sample > len(examples_list), hence min(...)
        sampled = random.sample(examples_list, to_sample)
        sampled_examples.extend(sampled)
    
    # Build a new TorchText Dataset from the sampled examples
    new_dataset = data.Dataset(
        sampled_examples,
        fields={
            "label": label_field,
            "text": dataset.fields["text"],  # or rename if your text field is different
        }
    )
    
    return new_dataset

class_dist =  args.class_dist
print("Class Distribution:", class_dist)

calib_data = sample_uneven_by_class(train_data, LABEL, 0.1, class_dist, seed=2021)
gpfq_data = sample_uneven_by_class(train_data, LABEL, 0.05, class_dist, seed=2021)

# calib_data = sample_subset_by_class(train_data, LABEL, fraction=0.05, seed=2021)
# gpfq_data = sample_subset_by_class(train_data, LABEL, fraction=0.05, seed=2021)

def print_dataset_label_distribution(dataset, label_field):
    label_counts = defaultdict(int)

    for ex in dataset.examples:
        # ex.label is a string here
        label_idx_int = label_field.vocab.stoi[ex.label]  # convert string -> int
        label_counts[label_idx_int] += 1

    print("Class distribution in dataset:")
    for label_idx, count in label_counts.items():
        label_str = label_field.vocab.itos[label_idx]  # int -> string
        print(f"  Label index: {label_idx} | Label string: '{label_str}' | Count: {count}")

# After creating calib_dataset, call:
print_dataset_label_distribution(calib_data, LABEL)

train_iterator,validation_iterator = data.BucketIterator.splits(
    (train_data,valid_data),
    batch_size = BATCH_SIZE,
    # Sort key is how to sort the samples
    sort_key = lambda x:len(x.text),
    sort_within_batch = True,
    device = device
)

test_iterator = data.BucketIterator(
    test_data,
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device
)

calib_iterator = data.BucketIterator(
    calib_data,
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device
)

gpfq_iterator = data.BucketIterator(
    gpfq_data,
    batch_size=16,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device
)

SIZE_OF_VOCAB = len(TEXT.vocab)
EMBEDDING_DIM = 100
NUM_HIDDEN_NODES = 100
NUM_OUTPUT_NODES = len(LABEL.vocab)
NUM_LAYERS = 1
BIDIRECTION = False
DROPOUT = 0.2
BIT_WIDTH = args.bit_width

print(SIZE_OF_VOCAB)
print(NUM_OUTPUT_NODES)

model = LSTMNet(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH).to(device)
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()

print(torch.cuda.is_available())

model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
#criterion = nn.BCELoss()
#criterion = criterion.to(device)

model

def multi_class_accuracy(preds, y):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(correct)
    return acc

def evaluate(model,iterator,criterion):
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    # deactivate the dropouts
    model.eval()
    
    # Sets require_grad flat False
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            text,text_lengths = batch.text
            text, text_lengths = text.to(device), text_lengths.to(device)
            batch_label = batch.label.to(device)
            predictions = model(text,text_lengths).squeeze()
              
            #compute loss and accuracy
            loss = criterion(predictions, batch_label)
            acc = multi_class_accuracy(predictions, batch_label)
            
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


epoch_save_path = f'ModelParameter_disc_{args.dataset}.pth'
checkpoint = torch.load(epoch_save_path)
model.load_state_dict(checkpoint)
print("Full Precision Model Loaded from", epoch_save_path)

torch.cuda.empty_cache()
print("Full Precision Model Evaluation Result:")
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')

device = torch.device("cpu")
model=model.to(device)
print("Model moved to cpu for quantization")

from brevitas.graph.quantize import preprocess_for_quantize
from ptq_common import quantize_model, apply_bias_correction, apply_act_equalization

pre_model = preprocess_for_quantize(
            model,
            equalize_iters=20,
            equalize_merge_bias=True,
            merge_bn=True,
            channel_splitting_ratio=0.0,
            channel_splitting_split_input=False)

dtype = getattr(torch, 'float')
quant_model = quantize_model(
        pre_model.to(device),
        dtype=dtype,
        device=device,
        backend='layerwise',
        scale_factor_type='float_scale',
        bias_bit_width=32,
        weight_bit_width=BIT_WIDTH,
        weight_narrow_range=False,
        weight_param_method='stats',
        weight_quant_granularity='per_tensor',
        weight_quant_type='sym',
        layerwise_first_last_bit_width=BIT_WIDTH,
        act_bit_width=BIT_WIDTH,
        act_param_method='stats',
        act_quant_percentile=99.9,
        act_quant_type='sym',
        quant_format='int',
        layerwise_first_last_mantissa_bit_width=4,
        layerwise_first_last_exponent_bit_width=3,
        weight_mantissa_bit_width=4,
        weight_exponent_bit_width=3,
        act_mantissa_bit_width=4,
        act_exponent_bit_width=3).to(device) 

device = torch.device(f'cuda:{args.cuda}')
device = torch.device('cuda:3')
model=model.to(device)
quant_model=quant_model.to(device)
print("Model moved back to cuda after quantization")

# print("Raw Quant Model Evaluation Result:")
# test_loss, test_acc = evaluate(quant_model, test_iterator, criterion)
# print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')

from brevitas.graph.calibrate import calibration_mode
#from tqdm import tqdm
def calibrate_model(model, iterator, num_calibration_steps=1000):

    model.eval()  # Set model to evaluation mode
    
    # Switch to calibration mode for Brevitas-based calibration
    with calibration_mode(model):
        # Ensure no gradients are being calculated
        with torch.no_grad():
            step = 0
            for batch in tqdm(iterator, desc="Applying Calibration"):
                # Get inputs and lengths from the batch
                text, text_lengths = batch.text
                text, text_lengths = text.to(device), text_lengths.to(device)
                
                # Forward pass to accumulate statistics for calibration
                _ = model(text, text_lengths)
                
                # Increment step and check if calibration step limit is reached
                step += 1
                if step >= num_calibration_steps:
                    break
    
    print("Calibration done.")
    return model

from brevitas.graph.gpfq import gpfq_mode
def apply_gpfq(iterator, model, act_order=False, p=1.0, use_gpfa2q=False, accumulator_bit_width=None):

    model.eval()  # Set the model to evaluation mode
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    with torch.no_grad():
        # Use gpfq_mode from Brevitas for applying GPFQ
        with gpfq_mode(model,
                       p=p,
                       use_quant_activations=True,
                       act_order=act_order
                       #use_gpfa2q=use_gpfa2q,
                       #accumulator_bit_width=accumulator_bit_width
                      ) as gpfq:
            gpfq_model = gpfq.model  # Get the GPFQ-modified model

            for i in range(gpfq.num_layers):  # Loop through each layer to apply GPFQ
                for batch in tqdm(iterator, desc="Applying GPFQ"):
                    # Unpack the text and text lengths from the batch
                    text, text_lengths = batch.text
                    text, text_lengths = text.to(device), text_lengths.to(device)
                    
                    # Forward pass through the GPFQ model
                    gpfq_model(text, text_lengths)

                # Update GPFQ after processing each layer
                gpfq.update()

    print("GPFQ applied successfully.")
    return model

torch.cuda.empty_cache()
print("Calibrating the quantized model...")
cal_model=calibrate_model(quant_model, calib_iterator)
# print("Calibrated Model Evaluation Result:")
# test_loss, test_acc = evaluate(cal_model, test_iterator, criterion)
# print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')

print("Applying GPFQ on the calibrated model...")
torch.cuda.empty_cache()
gpfq_model = apply_gpfq(calib_iterator, cal_model, act_order=False, p=1.0, use_gpfa2q=False)

print("Quantization is completed! Evaluating the quantized model...")
torch.cuda.empty_cache()
test_loss, test_acc = evaluate(gpfq_model, test_iterator, criterion)
print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')

SAVE_FOLDER = f"DiscQuantResultsWithUnevenDist_{args.dataset}"

# Check if the folder exists, if not, create it
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Define the filename dynamically based on bit width
filename = f"ModelParameter_Disc_{BIT_WIDTH}bit_" + "_".join(map(str, class_dist)) + ".pth"
model_save_path = os.path.join(SAVE_FOLDER, filename)

# Save the model
torch.save(gpfq_model.state_dict(), model_save_path)

print(f"Model saved at: {model_save_path}")

import csv

# Define the CSV file path
csv_file_path = os.path.join(SAVE_FOLDER, "evaluation_results.csv")

# Define the header for the CSV file
csv_headers = ["Bit Width", "Class 1", "Class 2", "Class 3", "Class 4", "Test Loss", "Test Accuracy"]

# Check if the CSV file exists
file_exists = os.path.isfile(csv_file_path)

# Open the CSV file in append mode
with open(csv_file_path, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # If the file does not exist, write the header
    if not file_exists:
        csv_writer.writerow(csv_headers)

    # Add a new row with the evaluation results
    csv_writer.writerow([BIT_WIDTH, class_dist[0], class_dist[1], class_dist[2], class_dist[3], test_loss, test_acc])

print(f"Evaluation results logged in: {csv_file_path}")
