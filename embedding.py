import os
from rdflib import Graph, BNode
from om.ont import tokenize
import itertools
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, TensorDataset

import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]



def batched(iterable, n):
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch

base_path = ''
out_base = ''

models = [('GritLM/GritLM-7B', 'gritlm-7b'), ('infgrad/stella-base-en-v2', 'stella-base')]


for md, mn in tqdm(models):

    tokenizer = AutoTokenizer.from_pretrained(md)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModel.from_pretrained(md)
    if mn == 'gritlm-7b':
        model.half()
    model.eval()
    model.cuda(0)
    
    
    for p, d, fs in os.walk(base_path):
        for f in tqdm(fs):
            if f.endswith('.owl'):
                if mn == 'gritlm-7b' and f in {'agronomicTaxon.owl', 'dbpedia-light.owl'}:
                    continue
                ont_name = f
                print(f)
                g = Graph().parse(os.path.join(p, f))
                
                subs = set(g.subjects())
                props = set(g.predicates())
                objs = set(g.objects())
                
                ks = []
                sents = []
                
                for s in subs.union(props, objs):
                
                    if type(s) == BNode:
                        continue
                        
                    
                    if s.startswith('http://'):
                        txt = ' '.join(tokenize(s.split('#')[-1]))
                    else:
                        txt = s
                        
                    ks.append(re.sub(r'\n+', ' ', s))
                    sents.append(txt)
                    
                encoded_input = tokenizer(sents, padding="longest", return_tensors="pt", max_length=512, truncation=True)
                
                embs = []
    
                for i, a in tqdm(DataLoader(TensorDataset(encoded_input['input_ids'], encoded_input['attention_mask']), batch_size=2)):
                    with torch.no_grad():
                        outputs = model(input_ids=i.cuda(0), attention_mask=a.cuda(0))
                        embeddings = average_pool(outputs.last_hidden_state, a.cuda(0))
                        sentence_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        embs.extend(sentence_embeddings.cpu())
                
                embs = torch.stack(embs)
                embl = embs.tolist()
                eln = []
                for l in embl:
                    eln.append(' '.join([str(v) for v in l]))
                    
                with open(os.path.join(out_base, f'{ont_name}-{mn}'), 'w') as f:
                    f.write(f'{len(embs)}\n')
                    
                    f.writelines([f'{k}\n' for k in ks])
                    f.writelines([f'{l}\n' for l in eln])

