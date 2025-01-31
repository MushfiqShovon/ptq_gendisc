import os
import re
import time
import spacy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
import pandas as pd
from models import Gen
import warnings as wrn
import logging
from tqdm import tqdm
import argparse
import csv

# Argument parsing
parser = argparse.ArgumentParser(description="Inference Script")

parser.add_argument('--bit_width', type=int, default=8, help='bit width for quantization')
parser.add_argument('--dataset', type=str, choices=['ag_news', 'dbpedia'], required=True, help='Dataset name (e.g., ag_news, dbpedia)')
parser.add_argument('--cuda', type=int, default=0, help='CUDA device number (e.g., 0, 1, 2)')

args = parser.parse_args()

SEED = 2021

wrn.filterwarnings('ignore')
os.environ['SP_DIR'] = '/opt/conda/lib/python3.11/site-packages'
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cuda.deterministic = True

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
print(device)

MODE = 'inference' # 'train' or 'inference' or 'none'

train_data = pd.read_csv(f'data/{args.dataset}/train.csv', header=None, usecols=[0,2])
train_data.columns = ['label', 'text']
valid_data = pd.read_csv(f'data/{args.dataset}/valid.csv', header=None, usecols=[0,2])
valid_data.columns = ['label', 'text']
test_data = pd.read_csv(f'data/{args.dataset}/test.csv', header=None, usecols=[0,2])
test_data.columns = ['label', 'text']

TRAIN_SIZE = len(train_data)
VALID_SIZE = len(valid_data)
TEST_SIZE = len(test_data)
print(TRAIN_SIZE, VALID_SIZE, TEST_SIZE)

# def clean_text(text):
#     return re.sub(r'[^A-Za-z0-9]+', ' ', str(text))

# train_data['text'] = train_data['text'].apply(clean_text)
# valid_data['text'] = valid_data['text'].apply(clean_text)
# test_data['text'] = test_data['text'].apply(clean_text)

# train_data.to_csv('data/ag_news3/train_clean.csv', index=False, header=False)
# valid_data.to_csv('data/ag_news3/valid_clean.csv', index=False, header=False)
# test_data.to_csv('data/ag_news3/test_clean.csv', index=False, header=False)

spacy_en = spacy.load('en_core_web_sm')

MAX_LEN = 80
def spacy_tokenizer(text):
    tokens = [tok.text for tok in spacy_en.tokenizer(text)]
    return tokens[:MAX_LEN]

LABEL = data.LabelField()
TEXT = data.Field(tokenize=spacy_tokenizer, batch_first=True, include_lengths=True)
fields = [('label', LABEL), ('text', TEXT)]

train_dataset = data.TabularDataset(path=f'data/{args.dataset}/train_clean.csv', format='csv', fields=fields, skip_header=True)
valid_dataset = data.TabularDataset(path=f'data/{args.dataset}/valid_clean.csv', format='csv', fields=fields, skip_header=True)
test_dataset = data.TabularDataset(path=f'data/{args.dataset}/test_clean.csv', format='csv', fields=fields, skip_header=True)

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

# Implement the logic
if args.dataset == 'ag_news':
    calib_dataset = sample_subset_by_class(train_dataset, LABEL, fraction=0.1, seed=2021)
    gpfq_dataset = sample_subset_by_class(train_dataset, LABEL, fraction=0.05, seed=2021)
elif args.dataset == 'dbpedia':
    calib_dataset = sample_subset_by_class(train_dataset, LABEL, fraction=0.014, seed=2021)
    gpfq_dataset = sample_subset_by_class(train_dataset, LABEL, fraction=0.005, seed=2021)
    train_dataset = sample_subset_by_class(train_dataset, LABEL, fraction=0.25, seed=2021)   
    test_dataset = sample_subset_by_class(test_dataset, LABEL, fraction=0.15, seed=2021)
else:
    raise ValueError(f"Unexpected dataset: {args.dataset}")


print("Reduced dataset created!")

TRAIN_SIZE = len(train_dataset)
VALID_SIZE = len(valid_data)
TEST_SIZE = len(test_dataset)
print(TRAIN_SIZE, VALID_SIZE, TEST_SIZE)

#print(vars(train_dataset.examples[0]))

TEXT.build_vocab(
    train_dataset,
    max_size=40000,    # keep only top 40k tokens by frequency
    min_freq=5         # (optional) only include tokens appearing >= 5 times
)
LABEL.build_vocab(train_dataset)

label_counts = {LABEL.vocab.itos[i]: LABEL.vocab.freqs[LABEL.vocab.itos[i]] for i in range(len(LABEL.vocab))}
print("Number of instances per class:", label_counts)

print("Size of text vocab:",len(TEXT.vocab))

print("Size of label vocab:",len(LABEL.vocab))

TEXT.vocab.freqs.most_common(10)


BATCH_SIZE = 32

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_dataset, valid_dataset),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device
)

test_iterator = data.BucketIterator(
    test_dataset,
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device
)

calib_iterator = data.BucketIterator(
    calib_dataset,
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device
)

gpfq_iterator = data.BucketIterator(
    gpfq_dataset,
    batch_size=16,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device
)

VOCAB_SIZE= len(TEXT.vocab)
WORD_EMB_DIM = 100
LABEL_EMB_DIM = 100
HID_DIM = 100
NLAYERS = 1
NCLASS = len(LABEL.vocab)
DROPOUT = 0
USE_CUDA = torch.cuda.is_available()
TIED = False
USE_BIAS = False
CONCAT_LABEL = 'hidden'
AVG_LOSS = False
ONE_HOT = False
BIT_WIDTH = args.bit_width

LR = 1e-4
LOG_INTERVAL = 200
CLIP = 1.0
LOGGING = logging.INFO

model = Gen(VOCAB_SIZE, WORD_EMB_DIM, LABEL_EMB_DIM, HID_DIM, NLAYERS, NCLASS, DROPOUT, USE_CUDA, TIED, USE_BIAS, CONCAT_LABEL, AVG_LOSS, ONE_HOT).to(device)
criterion = nn.CrossEntropyLoss(reduce=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

def init_hidden(model, bsz):
    weight = next(model.parameters())
    # Return hidden state and cell state as 2D tensors
    return (weight.new_zeros(NLAYERS, HID_DIM),
            weight.new_zeros(NLAYERS, HID_DIM))

def evaluate(valid_iterator, model, criterion, mode='valid', model_state=0):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    cnt = 0
    
    with torch.no_grad():
        for batch in tqdm(valid_iterator, desc=f"Evaluating ({mode})", leave=True):
            sents = [torch.tensor(row) for row in batch.text[0]]
            labels = batch.label
            # y_exts = [torch.full((batch.text[0].shape[1],), labels[i], dtype=torch.long) for i in range(len(labels))]
            y_exts = []
            for y_label in range(NCLASS):
                y_ext = []
                for d in sents:
                    y_ext.append(torch.LongTensor([y_label] * (len(d) - 1)))
                y_exts.append(y_ext)
            
            
            hidden = init_hidden(model, len(sents))
            x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
            x_pred = nn.utils.rnn.pack_sequence([s[1:] for s in sents])

            # p_y = torch.FloatTensor([0.071] * len(seq_len))

            losses = []
            for y_ext in y_exts:
                y_ext = nn.utils.rnn.pack_sequence(y_ext)

                if device.type == 'cuda':
                    x, y_ext, x_pred, labels = x.to(device), y_ext.to(device), x_pred.to(device), labels.to(device)

                # output (batch_size, )
                hidden = init_hidden(model, len(sents))
                
                out = model(x, x_pred, y_ext, hidden, criterion, model_state)
                
                loss_matrix = criterion(out, x_pred.data)

                LM_loss = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(
                    loss_matrix, x.batch_sizes))[0].transpose(0,1)
                sum_loss = torch.sum(LM_loss, dim = 1)
                
                losses.append(sum_loss)

            losses = torch.cat(losses, dim=0).view(-1, len(sents))
            prediction = torch.argmin(losses, dim=0)

            num_correct = (prediction == labels).float().sum()

            total_loss += torch.sum(torch.min(losses, dim=0)[0]).item()
            total_correct += num_correct.item()
            cnt += 1

    return total_loss / cnt, total_correct / (VALID_SIZE if mode == 'valid' else TEST_SIZE if mode == 'test' else '0') * 100.0

model.load_state_dict(torch.load('gen_lstm_spacy_dbpedia2.pth'))
model.eval()
        
fp_test_loss, fp_test_acc = evaluate(test_iterator, model, criterion, mode='test', model_state='fp')
print('=' * 89)
print(f'FP Test Loss: {fp_test_loss} | FP Test Acc:  {fp_test_acc} | Bitwidth: {BIT_WIDTH}')
print('=' * 89)

from brevitas.graph.quantize import preprocess_for_quantize
from ptq_common import quantize_model, apply_bias_correction, apply_act_equalization

dtype = getattr(torch, 'float')
device = torch.device('cpu')
print("device is set to ",device)
print("Quantizing the model to a bitwidth of ", BIT_WIDTH)
quant_model = quantize_model(
        model.to(device),
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
        act_quant_percentile=99.99,
        act_quant_type='sym',
        quant_format='int',
        layerwise_first_last_mantissa_bit_width=4,
        layerwise_first_last_exponent_bit_width=3,
        weight_mantissa_bit_width=4,
        weight_exponent_bit_width=3,
        act_mantissa_bit_width=4,
        act_exponent_bit_width=3).to(device) 

print("Quantization completed!")
device = torch.device(f'cuda:{args.cuda}')
print("device is set back to ",device)
model=model.to(device)
quant_model=quant_model.to(device)

from brevitas.graph.calibrate import calibration_mode
#from tqdm import tqdm
def calibrate_model(model, iterator, num_calibration_steps=600):

    model.eval()  # Set model to evaluation mode
    
    # Switch to calibration mode for Brevitas-based calibration
    with calibration_mode(model):
        # Ensure no gradients are being calculated
        with torch.no_grad():
            step = 0
            #for batch in tqdm(iterator, desc="Calibrating", total=num_calibration_steps, leave=True):
            for batch in tqdm(iterator, desc="Calibrating"):
                sents = [torch.tensor(row) for row in batch.text[0]]
                labels = batch.label
                # y_exts = [torch.full((batch.text[0].shape[1],), labels[i], dtype=torch.long) for i in range(len(labels))]
                y_ext = []
                for d, lbl in zip(sents, batch.label):
                    y_ext.append(torch.LongTensor([lbl] * (len(d) - 1)))

                
                
                hidden = init_hidden(model, len(sents))
                x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
                x_pred = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
                y_ext = nn.utils.rnn.pack_sequence(y_ext)

                # Move to GPU if available
                if next(model.parameters()).is_cuda:
                    x = x.to(device)
                    x_pred = x_pred.to(device)
                    y_ext = y_ext.to(device)

                # Forward pass to gather calibration stats
                _ = model(x, x_pred, y_ext, hidden, criterion, model_state='quant')
                    
    

                
                # Increment step and check if calibration step limit is reached
                step += 1
                #print(step)
                if step >= num_calibration_steps:
                    break
    
    print("Calibration done.")
    return model

from brevitas.graph.gpfq import gpfq_mode

def apply_gpfq(iterator, model, act_order=False, p=1.0, use_gpfa2q=False, accumulator_bit_width=None):
    """
    Apply Generalized Post-Training Quantization (GPFQ) to a model.
    
    Args:
        iterator: Data iterator for processing batches.
        model: Calibrated quantized model.
        act_order: Whether to reorder activations for optimization.
        p: A hyperparameter for GPFQ adjustments.
        use_gpfa2q: Use alternative quantization method (GPFA2Q).
        accumulator_bit_width: Bit width for accumulators (optional).
    
    Returns:
        GPFQ-applied model.
    """
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device

    with torch.no_grad():
        # Use gpfq_mode from Brevitas for applying GPFQ
        with gpfq_mode(model,
                       p=p,
                       use_quant_activations=True,
                       act_order=act_order
                       # use_gpfa2q=use_gpfa2q,
                       # accumulator_bit_width=accumulator_bit_width
                      ) as gpfq:
            gpfq_model = gpfq.model  # Get the GPFQ-modified model

            for i in range(gpfq.num_layers):  # Loop through layers
                for batch in tqdm(iterator, desc="Applying GPFQ"):
                    # Process batch
                    sents = [torch.tensor(row) for row in batch.text[0]]
                    y_ext = []
                    for d, lbl in zip(sents, batch.label):
                        y_ext.append(torch.LongTensor([lbl] * (len(d) - 1)))
                    
                    hidden = init_hidden(model, len(sents))
                    x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
                    x_pred = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
                    y_ext = nn.utils.rnn.pack_sequence(y_ext)

                    if device.type == 'cuda':
                        x, y_ext, x_pred = x.to(device), y_ext.to(device), x_pred.to(device)

                    # Forward pass through the GPFQ model
                    gpfq_model(x, x_pred, y_ext, hidden, criterion, model_state='quant')

                # Update GPFQ after processing each layer
                gpfq.update()

    print("GPFQ applied successfully.")
    return model
    
torch.cuda.empty_cache()
print("Calibrating the model...")
cal_model=calibrate_model(quant_model, calib_iterator)

torch.cuda.empty_cache()
print("Applying GPFQ on the calibrated model...")
gpfq_model = apply_gpfq(gpfq_iterator, cal_model, act_order=False, p=1.0, use_gpfa2q=False)

torch.cuda.empty_cache()
print("Evaluating the model...")
test_loss, test_acc = evaluate(test_iterator, cal_model, criterion, mode='test', model_state='quant')
print('=' * 89)
print(f'Test Loss: {test_loss} | Test Acc after calibration:  {test_acc} | Bitwidth: {BIT_WIDTH}')
print('=' * 89)

SAVE_FOLDER = f"GenQuantResults_{args.dataset}"

# Check if the folder exists, if not, create it
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Define the filename dynamically based on bit width
model_save_path = os.path.join(SAVE_FOLDER, f"ModelParameter_Gen_{BIT_WIDTH}bit.pth")

# Save the model
torch.save(cal_model.state_dict(), model_save_path)

print(f"Model saved at: {model_save_path}")

import csv

# Define the CSV file path
csv_file_path = os.path.join(SAVE_FOLDER, "evaluation_results.csv")

# Define the header for the CSV file
csv_headers = ["Bit Width", "Test Loss", "Test Accuracy"]

# Check if the CSV file exists
file_exists = os.path.isfile(csv_file_path)

# Open the CSV file in append mode
with open(csv_file_path, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # If the file does not exist, write the header
    if not file_exists:
        csv_writer.writerow(csv_headers)

    # Add a new row with the evaluation results
    csv_writer.writerow([BIT_WIDTH, test_loss, test_acc])

print(f"Evaluation results logged in: {csv_file_path}")

