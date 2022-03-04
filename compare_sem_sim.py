from cgitb import small
from tkinter import dialog
from reader import MultiWozReader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from config import global_config as cfg
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
reader = MultiWozReader(tokenizer)
model = GPT2LMHeadModel.from_pretrained(cfg.gpt_path).cuda()

all_batches = reader.get_batches('train')
data_iterator = reader.get_nontranspose_data_iterator(all_batches)

this_dict = dict()
smallest_idx = None

for batch_idx, dial_batch in tqdm(enumerate(data_iterator)):
    for dialogue in dial_batch:
        dialogue_len = len(dialogue)
        for turn_idx, turn_text in enumerate(dialogue):
            this_turn_idx = turn_idx - dialogue_len
            if smallest_idx is None or this_turn_idx < smallest_idx:
                smallest_idx = this_turn_idx
            with torch.no_grad():
                usr_vec = model(torch.tensor(turn_text['user']).cuda())[0]
                resp_vec = model(torch.tensor(turn_text['resp']).cuda())[0]
                cos_sim = cosine_similarity(usr_vec.unsqueeze(-1), resp_vec.transpose(0,1).unsqueeze(0), dim=1)
                rbert = cos_sim.max(dim=1)[0].mean()
                pbert = cos_sim.max(dim=0)[0].mean()
                fbert = 2 * (pbert * rbert) / (pbert + rbert)
                if this_turn_idx not in this_dict:
                    this_dict[this_turn_idx] = []
                this_dict[this_turn_idx].append(fbert)

for turn in range(smallest_idx, 1):
    if turn in this_dict:
        print(turn, np.mean(this_dict[turn]))

            