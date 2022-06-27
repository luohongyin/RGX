import json
import torch

import random

def convert_ctx(ctx, ans):
    ctx = ctx.replace(ans, '<mask>')
    return f'{ctx} </s> {ans}'

def convert_squad(sq):
    if 'processed' in sq:
        return sq['context']

    if 'description' in sq:
        print(sq['context'])
        abort()
        return sq['context']

    if 'title' in sq:
        title = sq['title'].replace('_', ' ')
    else:
        title = 'None'
    try:
        ans = sq['answers']['text'][0]
    except:
        print(sq)
        abort()
    ans_len = len(ans)
    ans_st = sq['answers']['answer_start'][0]
    ans_ed = ans_st + ans_len
    ctx = sq['context'][:ans_st] + '<mask>' + sq['context'][ans_ed:]
    if 'evidence' in sq:
        evi = sq['evidence']
        return f'{ctx} </s> {ans} </s> {evi}'
    else:
        return f'{ctx} </s> {ans}'

def load_data(dataset, tokenizer, ctx_max_length=128, ques_max_length=64):
    try:
        dataset = json.load(open(dataset, 'r'))
    except:
        dataset = dataset

    ctxs_raw = [x['context'] for x in dataset]
    ctxs_masked = [convert_squad(x) for x in dataset]
    questions = [x['question'] for x in dataset]
    ans_ppl = torch.Tensor([0 for x in dataset])

    # ans_ppl = torch.Tensor([x['ans_ppl'] for x in dataset])
    ans_st_char = [x['answers']['answer_start'][0] for x in dataset]
    ans_text = [x['answers']['text'][0] for x in dataset]

    ctx_encode = tokenizer(
            ctxs_masked,
            max_length=ctx_max_length,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True
    )

    ques_encode = tokenizer(
            questions,
            max_length=ques_max_length,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True
    )
    ctx_idx = ctx_encode['input_ids']
    ctx_attn_mask = ctx_encode['attention_mask']
    ctx_set = list(zip(ctx_idx, ctx_attn_mask))

    ques_idx = ques_encode['input_ids']
    ques_attn_mask = ques_encode['attention_mask']
    ques_set = list(zip(ques_idx, ques_attn_mask))

    return ctx_set, ques_set, # ans_ppl, ctxs_raw, ctxs_masked, questions, ans_st_char, ans_text

def shuffle_training_batch(ctx_set, ques_set):
    dataset = list(zip(ctx_set, ques_set))
    random.shuffle(dataset)
    ctx_set_new = [x[0] for x in dataset]
    ques_set_new = [x[1] for x in dataset]
    return ctx_set_new, ques_set_new
