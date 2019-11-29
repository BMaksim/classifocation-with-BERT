import torch
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from multiprocessing import Pool, cpu_count
from tools import *
import convert_examples_to_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





BERT_MODEL = 'bert-base-cased'
TASK_NAME = 'bert'
OUTPUT_DIR = f'outputs/{TASK_NAME}/'
REPORTS_DIR = f'reports/{TASK_NAME}_evaluation_report/'


TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
epohs = 1
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"


processor = BinaryClassificationProcessor()
train_examples = processor.get_train_examples("data/")
train_examples_len = len(train_examples)
label_list = processor.get_labels()


num_train_optimization_steps = int(
    train_examples_len / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * epohs

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

label_map = {label: i for i, label in enumerate(label_list)}
train_examples_for_processing = [(example, label_map, 128, tokenizer, 'classification') for example in train_examples]

process_count = cpu_count() - 1
with Pool(process_count) as p:
    train_features = list(p.imap(convert_examples_to_features.convert_example_to_feature, train_examples_for_processing))

with open("data/" + "train_features.pkl", "wb") as f:
    pickle.dump(train_features, f)


model = BertForSequenceClassification.from_pretrained("cache/" + 'cased_base_bert_pytorch.tar.gz', cache_dir="cache/", num_labels=2)
model.to(device)



param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=LEARNING_RATE,
                     warmup=WARMUP_PROPORTION,
                     t_total=num_train_optimization_steps)

global_step = 0
nb_tr_steps = 0
tr_loss = 0

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)

model.train()
for _ in range(int(epohs)):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))

        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        loss.backward()
        print("\r%f" % loss, end='')

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

model_to_save = model.module if hasattr(model, 'module') else model

output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(OUTPUT_DIR)



