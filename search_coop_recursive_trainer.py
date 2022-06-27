import re
import sys
import copy
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from ques_pretrain_data import *
from datasets import *
from torch.distributions import Categorical
from nltk.tokenize import word_tokenize, sent_tokenize
from run_qa_func import run_qa
from qgen_merge_new import ans_pair_metric

from transformers import (
    BartTokenizer,
    BartTokenizerFast,
    BartForConditionalGeneration,
    ElectraTokenizerFast,
    ElectraForQuestionAnswering
)

question_words = set([
    'what',
    'who',
    'where',
    'which',
    'why',
    'when',
    'how'
])

def count_ques_word(question):
    n = 0
    for w in question.split(' '):
        if w in question_words:
            n += 1
    return n

def post_proc_ques(question):
    global question_words
    words_list = question.split(' ')
    if len(words_list) < 3:
        return question

    for qword in question_words:
        if qword in words_list[0]:
            words_list[0] = qword
            break

    while words_list[0] == words_list[1]:
        words_list = words_list[1:]

    new_question = ' '.join(words_list)
    return new_question

def em(data, idx, num_class, max_step=100):

    data_list = list(data.items())
    scores = torch.Tensor([[v[idx] for k, v in data_list]]).cuda()
    num_case = scores.size(1)
    max_score = scores.max().item()
    min_score = scores.min().item()

    if num_class == 2:
        # num_class += 1
        label = torch.Tensor([0, min_score]).cuda().unsqueeze(1)
        # label = torch.Tensor([0, min_score / 2, min_score]).cuda().unsqueeze(1)
    elif num_class == 3:
        label = torch.Tensor([0, max_score / 2, max_score]).cuda().unsqueeze(1)

    step_id = 0
    new_label = torch.zeros_like(label)
    # print(label)
    while True:
        distance = torch.abs(scores - label)
        _, cluster = distance.min(0)
        # print(cluster)
        for i in range(num_class):
            new_mean = torch.Tensor([scores[0][j] for j in range(num_case) if cluster[j] == i]).cuda()
            # print(new_mean)
            new_mean = new_mean.mean()
            new_label[i][0] = new_mean
        # print(new_label)
        # abort()
        gap = torch.abs(label - new_label).mean()
        if step_id > max_step or gap < scores.max() * 1e-3:
            break
        step_id += 1
        label = new_label

    # print(scores)
    # print(label)
    # print(cluster)
    for i in range(num_case):
        data_list[i][1].append(cluster[i].item())

    return dict(data_list)

def split_squad(squad):
    squad_old = []
    squad_new = []
    for sq in squad:
        # print(sq)
        # print('====================')
        sq[1] = list(set(sq[1]))
        sq[1] = [x for x in sq[1] if len(x) > 0]

        num_ans_old = len(sq[0]['answers']['text'])
        num_ans_new = len(sq[1])
        ctx = sq[0]['context']

        for i in range(num_ans_old):
            new_squad = copy.deepcopy(sq[0])
            new_squad['answers']['answer_start'] = new_squad['answers']['answer_start'][i: i+1]
            new_squad['answers']['text'] = new_squad['answers']['text'][i: i+1]
            new_squad['question'] = new_squad['question'][i]
            squad_old.append(new_squad)

        for i in range(num_ans_new):
            ans = sq[1][i]
            # print(ans)
            pattern = fr"\b{ans}\b"
            # print(pattern)

            try:
                y = re.search(pattern, ctx)
            except:
                continue

            if not y or ans not in ctx:
                continue

            new_squad = copy.deepcopy(sq[0])
            new_squad['answers'] = {'text': [ans], 'answer_start': [ctx.index(ans)]}
            new_squad['question'] = ''
            squad_new.append(new_squad)

    return squad_old, squad_new

def qgen(squads, tok, model):
    ctxs= [x['context'] for x in squads]
    answers = [x['answers']['text'][0] for x in squads]
    ctx_mask = [x.replace(y, '<mask>') for x, y in zip(ctxs, answers)]
    ctx_new = ['{} </s> {}'.format(x, y) for x, y in zip(ctx_mask, answers)]

    # inputs = tok(ctx_new, return_tensors='pt')
    inputs = tok.batch_encode_plus(
        ctx_new,
        max_length=320,
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        gen_ids, gen_logits = model.generate(input_ids = inputs['input_ids'].cuda(),
                                             num_beams=1,
                                             max_length=64,
                                             early_stopping=True,
                                             attention_mask = inputs['attention_mask'].cuda(),
                                            )


    gen_ques_text = [
        tok.decode(g, skip_special_tokens=True) for g in gen_ids
    ]#[0]

    for i in range(len(squads)):
        squads[i]['question'] = gen_ques_text[i]
    # squad['question'] = gen_ques_text
    return squads

def new_squad_gen(squad_dataset):
    batch_size = 64
    tok = BartTokenizer.from_pretrained('facebook/bart-large')
    # model = BartForConditionalGeneration.from_pretrained('../model_ft_file/qgen_gen_new.pt')
    model = BartForConditionalGeneration.from_pretrained('model_file/ques_gen_raw.pt')
    model = model.cuda()
    new_dataset = []
    num_total_batch = len(squad_dataset) // batch_size
    num_proc_batch = 0

    for i in range(0, len(squad_dataset), batch_size):
        batch = [squad_dataset[j] for j in range(i, min(i + batch_size, len(squad_dataset)))]
        new_dataset += qgen(batch, tok, model)
        num_proc_batch += 1
        if num_proc_batch % 100 == 0:
            print('Processed {} of {} batches'.format(num_proc_batch, num_total_batch))

    random.shuffle(new_dataset)
    return new_dataset

def overlap(span_set, sent_id, st, ed):
    if sent_id not in span_set:
        span_set[sent_id] = set([])
        return False
    for cand_st, cand_ed in span_set[sent_id]:
        if ed < st:
            return True
        if not (ed < cand_st or st > cand_ed):
            return True
    return False

def sample(ae_list):
    span_set = {}
    ae_txt_set = set([])
    num_sample = 0
    for ae_t, sent_id, st, ed in ae_list:
        if ae_t in ae_txt_set:
                continue
        if not overlap(span_set, sent_id, st, ed):
            ae_txt_set.add(ae_t)
            span_set[sent_id].add((st, ed))
            num_sample += 1
        if num_sample > 14:
            break
    ae_txt_list = list(ae_txt_set)
    return ae_txt_list

def sample_with_score(ae_list_scores):
    ae_list = [x[0] for x in ae_list_scores]
    ae_scores = [x[1][0] for x in ae_list_scores]
    ae_ques = [x[1][-1] for x in ae_list_scores]
    span_set = {}
    ae_txt_set = set([])
    ae_txt_list = []
    num_sample = 0
    ae_id = -1
    # print(ae_list_scores[:16])
    # abort()

    for ae_t, sent_id, st, ed, st_char, ed_char, _ in ae_list:
        ae_id += 1

        if ae_t in ae_txt_set:
                continue

        if not overlap(span_set, sent_id, st, ed):

            ae_txt_set.add(ae_t)
            ae_txt_list.append((ae_t, ae_scores[ae_id], ae_ques[ae_id], sent_id, st_char, ed_char))

            span_set[sent_id].add((st, ed))
            num_sample += 1

        if num_sample > 6:
            break
        # ae_id += 1

    # ae_txt_list = list(ae_txt_set)
    # ae_txt_list = sorted(ae_txt_list, key = lambda x: x[1], reverse=True)
    return ae_txt_list

def my_sent_tokenize(psg):
    sents = sent_tokenize(psg)
    new_sents = []
    for sent in sents:
        sent_words = sent.split(' ')
        if len(sent_words) < 64:
            new_sents.append(sent)
        else:
            for i in range(0, len(sent_words), 20):
                new_txt = ' '.join(sent_words[i: i + 20])
                new_sents.append(new_txt)
    return new_sents

def new_ae_detect(ans2item):
    new_ae_list = []

    for i, ans_item in enumerate(ans2item):
        k, v = ans_item
        ans_txt, ans_st_char = k

        if len(ans_txt) == 0:
            continue

        evi_list = [x[0][1] for x in v]
        score_list = [x[1][0] for x in v]

        if max(score_list) < 15000:
            new_ae_list.append(ans_txt, ans_st_char)
            # new_ae_list += [(ans_txt, x, ans_st_char) for x in evi_list]

    return [list(x) for x in set(new_ae_list)]

def merge_ans2item(ans2item):
    rgx_dict = {}

    for i, ans_item in enumerate(ans2item):
        k, v = ans_item
        k_tup = tuple(k)
        if k_tup not in rgx_dict:
            rgx_dict[k_tup] = v
        else:
            rgx_dict[k_tup] = v + rgx_dict[k_tup]

    new_ans2item = [[list(k), v] for k, v in rgx_dict.items()]
    return new_ans2item

def mask_ctx(ctx, title, ae, qgen_tok):
    sent_list = sent_tokenize(ctx)

    if len(ae) == 8:
        ae_t, evi_t, sent_id, st, ed, st_char, ed_char, value = ae
        evi_t = evi_t.strip()
        sent_list[sent_id] = sent_list[sent_id][:st_char] + '<mask>' + sent_list[sent_id][ed_char:]
        masked_ctx = ' '.join(sent_list)
    elif len(ae) == 7:
        ae_t, sent_id, st, ed, st_char, ed_char, value = ae
        evi_t = None
        sent_list[sent_id] = sent_list[sent_id][:st_char] + '<mask>' + sent_list[sent_id][ed_char:]
        masked_ctx = ' '.join(sent_list)
    elif len(ae) == 3:
        ae_t, evi_t, st_char = ae
        ed_char = st_char + len(ae_t)
        masked_ctx = f'{ctx[:st_char]}<mask>{ctx[ed_char:]}'
    else:
        evi_t = None
        ae_t, st_char = ae
        ed_char = st_char + len(ae_t)
        masked_ctx = f'{ctx[:st_char]}<mask>{ctx[ed_char:]} </s> {ae_t}'
        return ctx, masked_ctx, 0, masked_ctx.index('<mask>')
    # print(masked_ctx)
    # print('-----------------')
    # print(ctx)
    # print('-----------------')
    # print(ae_t)
    # abort()

    masked_enc = qgen_tok(
        masked_ctx,
        return_offsets_mapping=True,
    )

    # masked_enc['offset_mapping'][-1][1] = len(masked_ctx)

    ans_pos = masked_enc['input_ids'].index(qgen_tok.mask_token_id)
    ans_len = len(qgen_tok.encode(ae_t)) - 2
    ans_end_pos = ans_pos + ans_len - 1

    evi_len = len(qgen_tok.encode(evi_t)) - 2
    max_ctx_length = 448 - ans_len - 3 - evi_len

    offset_token = 0

    while offset_token + max_ctx_length - 1 < ans_end_pos:
        offset_token += 128
    end_token = offset_token + max_ctx_length - 1
    end_token = min(end_token, len(masked_enc['input_ids']) - 1)

    ctx_st_char = masked_enc['offset_mapping'][offset_token][0]
    if end_token == len(masked_enc['offset_mapping']) - 1:
        ctx_ed_char = len(masked_ctx)
    else:
        ctx_ed_char = masked_enc['offset_mapping'][end_token][1]

    ctx_masked = f'{masked_ctx[ctx_st_char: ctx_ed_char]} </s> {ae_t} </s> {evi_t}'
    ctx = ctx[ctx_st_char:]
    try:
        return ctx, ctx_masked, ctx_st_char, ctx_masked.index('<mask>')
    except:
        print('test ctx_masked')
        print(len(masked_enc['offset_mapping']))
        print(end_token)
        print(ctx_ed_char)
        print(ctx)
        print('-----------------')
        print(ctx_masked)
        abort()

def mask_ctx_post(ctx, title, ae, qgen_tok):
    # sent_list = sent_tokenize(ctx)
    # ae_t, sent_id, st, ed, st_char, ed_char, value = ae
    # sent_list[sent_id] = sent_list[sent_id][:st_char] + '<mask>' + sent_list[sent_id][ed_char:]
    # masked_ctx = ' '.join(sent_list)

    ae_t, st_char = ae
    ed_char = st_char + len(ae_t)
    masked_ctx = ctx[:st_char] + '<mask>' + ctx[ed_char:]

    masked_enc = qgen_tok(
        masked_ctx,
        return_offsets_mapping=True,
    )

    # masked_enc['offset_mapping'][-1][1] = len(masked_ctx)

    ans_pos = masked_enc['input_ids'].index(qgen_tok.mask_token_id)
    ans_len = len(qgen_tok.encode(ae_t)) - 2
    ans_end_pos = ans_pos + ans_len - 1

    title_len = len(qgen_tok.encode(title)) - 1
    max_ctx_length = 448 - ans_len - 3 - title_len

    offset_token = 0

    while offset_token + max_ctx_length - 1 < ans_end_pos:
        offset_token += 128
    end_token = offset_token + max_ctx_length - 1
    end_token = min(end_token, len(masked_enc['input_ids']) - 1)

    ctx_st_char = masked_enc['offset_mapping'][offset_token][0]
    if end_token == len(masked_enc['offset_mapping']) - 1:
        ctx_ed_char = len(masked_ctx)
    else:
        ctx_ed_char = masked_enc['offset_mapping'][end_token][1]

    ctx_masked = f'{masked_ctx[ctx_st_char: ctx_ed_char]} </s> {title} </s> {ae_t}'
    ctx = ctx[ctx_st_char:]
    try:
        return ctx, ctx_masked, ctx_st_char, ctx_masked.index('<mask>') - 2
    except:
        print('test ctx_masked')
        print(len(masked_enc['offset_mapping']))
        print(end_token)
        print(ctx_ed_char)
        print(ctx)
        print('-----------------')
        print(ctx_masked)
        abort()

def span_loc(aet, ctx, tok):
    # print(aet)
    aei = tok.encode(aet)[1: -1]
    ae_len = len(aei)
    st = 0
    ed = 0
    for i in range(len(ctx)):
        if aei == ctx[i: i + ae_len]:
            st = i
            ed = i + ae_len - 1
            return (st, ed)

def ans_loc(ctx, ans, st_char, tokenizer):

    start_loc = 0
    end_loc = 0
    ctx_enc = tokenizer(ctx, return_offsets_mapping=True, verbose=False)
    ans_enc = tokenizer(ans, return_offsets_mapping=True)

    ans_len = len(ans)
    ctx_num_tok = len(ctx_enc['input_ids'])

    if True:
        if True:
            st_char = st_char
            ed_char = st_char + ans_len - 1
            for j in range(1, ctx_num_tok):
                cur_offset = ctx_enc['offset_mapping'][j]
                if st_char >= cur_offset[0] and st_char < cur_offset[1]:
                    start_loc = j
                if ed_char >= cur_offset[0] and ed_char < cur_offset[1]:
                    end_loc = j
                    break

    if start_loc > 447:
        start_loc = 0
        end_loc = 0
    end_loc = min(end_loc, 447)

    return start_loc, end_loc

def filter_qa(ans2item_list):

    results = []

    for i in range(len(ans2item_list)):
        k, v = ans2item_list[i]

        # print(v[-1])

        v = [(ae, qg) for ae, qg in v if\
             k[0].lower() not in qg[-1] and\
             len(qg[-1].split(' ')) > 5 and\
             count_ques_word(qg[-1]) < 2
             ]

        if len(v) == 0:
            continue

        final_scores = torch.Tensor([qg[0] for ae, qg in v])
        min_ans_ppl = v[-1][1][3]

        max_fs, mfs_idx = final_scores.max(0)
        # min_ppl, mppl_idx = ans_ppl.min(0)

        # if max_fs < 100 or min_ans_ppl > 0.01:
        #     continue

        ques_list = [qg[-1] for ae, qg in v]
        # ques_list = [qg[-1] for ae, qg in v if qg[3] < 0.01]

        ques_dict = {}
        for q in ques_list:
            if q in ques_dict:
                ques_dict[q] += 1
            else:
                ques_dict[q] = 1

        ques_list_vote = [q for q in ques_dict.items() if q[1] > 1]

        if len(ques_list_vote) > 0:
            ques_list_count = sorted(ques_list_vote, key = lambda x: x[1], reverse=True)
            ques = ques_list_count[0][0]
            results.append([k, post_proc_ques(ques), max_fs.item(), min_ans_ppl / len(k[0].split(' '))])
            continue

        ques = v[-1][1][-1]
        results.append([k, post_proc_ques(ques), max_fs.item(), min_ans_ppl / len(k[0].split(' '))])

    return results

def logits_to_ans_loc(start_logits, end_logits, mode='joint'):
    bs, seq_len = start_logits.size()

    st_loc = None
    ed_loc = None
    pos_idx = torch.range(0, seq_len - 1).cuda().unsqueeze(0)

    if mode == 'greedy_left_to_right':
        _, st_loc = start_logits.max(1, keepdim=True)
        seq_mask = (pos_idx < st_loc).float() * 1000000
        st_loc = st_loc.squeeze(1)
        _, ed_loc = (end_logits - seq_mask).max(1)

    if mode == 'greedy_right_to_left':
        _, ed_loc = end_logits.max(1, keepdim=True)
        seq_mask = (pos_idx > ed_loc).float() * 1000000
        ed_loc = ed_loc.squeeze(1)
        _, st_loc = (start_logits - seq_mask).max(1)

    if mode == 'joint':
        ones_mask_l = torch.ones(seq_len, seq_len).cuda().tril(-1) * 1000000
        ones_mask_u = torch.ones(seq_len, seq_len).cuda().triu(10) * 1000000

        start_logits = start_logits.unsqueeze(2)
        end_logits = end_logits.unsqueeze(1)
        pred_logits = start_logits + end_logits - ones_mask_l.unsqueeze(0) - ones_mask_u.unsqueeze(0)
        pred_logits_1d = pred_logits.view(bs, -1)

        _, loc = pred_logits_1d.max(1)
        st_loc = loc // seq_len
        ed_loc = loc % seq_len

    return st_loc, ed_loc

def proc_title(title):
    return title.replace('_', ' ') # + ' <s>'

def proc_ctx(squad, aes, qgen_tok):
    # ctx, ctx_masked, st_char
    aes = [x for x in aes if x[0] in squad['context']]
    title = proc_title(squad['title'])
    ctx_new = [mask_ctx(squad['context'], title, x, qgen_tok) for x in aes]
    ctx = [x[0] for x in ctx_new]
    ctx_masked = [x[1] for x in ctx_new]
    st_offsets = [x[2] for x in ctx_new]
    ans_st_char = [x[3] for x in ctx_new]
    return ctx, ctx_masked, st_offsets, ans_st_char, aes

def proc_ctx_post(squad, aes, qgen_tok):
    aes = [x for x in aes if x[0] in squad['context']]
    title = proc_title(squad['title'])
    # print(aes[0])
    # abort()
    ctx_new = [mask_ctx_post(squad['context'], title, x, qgen_tok) for x in aes]
    ctx = [x[0] for x in ctx_new]
    ctx_masked = [x[1] for x in ctx_new]
    st_offsets = [x[2] for x in ctx_new]
    ans_st_char = [x[3] for x in ctx_new]
    return ctx, ctx_masked, st_offsets, ans_st_char, aes

def get_squad_ppl(squad, tok, model, ext_tok, ext_model, case_id,
                  model_args, data_args, training_args):

    squad, aes = squad
    ans_template = {'text': [], 'answer_start': []}

    squad['title'] = 'title'
    ctx_new, ctx_masked, char_offsets, ans_st_char, aes = proc_ctx(squad, aes, tok)

    aes_set_init = set([x[0] for x in aes])

    aes = [tuple(x) for x in aes]

    ctx_encoded = ext_tok(
        ctx_new,
        return_attention_mask=True,
        return_tensors='pt',
        return_offsets_mapping=True,
        max_length=448,
        padding='longest',
        truncation=True,
    )

    num_aes = len(aes)

    ans_locs = [ans_loc(c, x[0], y, ext_tok) for c, x, y in zip(ctx_new, aes, ans_st_char)]

    ctx_enc = ctx_encoded['input_ids'].cuda()
    ctx_attn_mask = ctx_encoded['attention_mask'].cuda()

    try:
        start_positions = torch.LongTensor([x[0] for x in ans_locs]).cuda()
        end_positions = torch.LongTensor([x[1] for x in ans_locs]).cuda()
    except:
        print(aes)
        print(ans_locs)
        abort()

    ques_max_length = 64

    start_positions += ques_max_length
    end_positions += ques_max_length

    cur_bs = len(ctx_new)

    # ------------------------------------------------------
    #
    # Question generation -
    # 1. Generating token ids and logits
    # 2. Get log probabilities of generated tokens for RL
    #
    #
    # ------------------------------------------------------

    search_batch_size = 96

    inputs = tok.batch_encode_plus(
        ctx_masked,
        max_length=448,
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    ae_log_probs_list = []
    gen_ques_text = []

    for i in range(0, inputs['input_ids'].size(0), search_batch_size):
        model.eval()
        with torch.no_grad():
            gen_outputs = model.generate(
                input_ids = inputs['input_ids'][i: i + search_batch_size].cuda(),
                num_beams = 1,
                max_length = ques_max_length,
                early_stopping = True,
                attention_mask = inputs['attention_mask'][i: i + search_batch_size].cuda(),
                do_sample = True,
                return_dict_in_generate = True,
                output_scores = True
            )
        
        # print(gen_outputs)
        # sys.exit()
        gen_ids = gen_outputs.sequences
        gen_logits = torch.cat(
            [x.unsqueeze(1) for x in gen_outputs.scores], dim = 1
        )

        c = Categorical(logits = gen_logits)
        gen_log_probs = c.log_prob(gen_ids[:, 1:])
        zero_mask = torch.zeros_like(gen_log_probs)
        inf_mask = torch.log(zero_mask)
        gen_log_probs = torch.where(gen_log_probs==inf_mask, zero_mask, gen_log_probs)

        non_zeros = (gen_log_probs < 0).float().sum(1)
        sum_log_probs = gen_log_probs.sum(1)
        ae_log_probs = sum_log_probs / non_zeros
        ae_log_probs_list.append(ae_log_probs)

        gen_ques_text += [
            tok.decode(g, skip_special_tokens=True) for g in gen_ids
        ]

    ae_dict = {}
    ae_log_probs = torch.cat(ae_log_probs_list, dim=0)
    try:
        gen_ques_text = [post_proc_ques(x) for x in gen_ques_text]
    except:
        gen_ques_text = gen_ques_text

    # ------------------------------------------------------
    #
    # Question Answering -
    # 1. Encode generated questions with ELECTRA tokenizer
    # 2. Extract answers with concatenated inputs
    #
    #
    # ------------------------------------------------------

    gen_cases = [
        {'context': squad['context'], 'question': x, 'answers': ans_template} for x in gen_ques_text
    ]

    # print(training_args)
    # abort()
    _, ans_pred = run_qa(gen_cases, ext_tok, ext_model, model_args, data_args, training_args)
    # print(ans_pred['0'])
    # abort()

    # ------------------------------------------------------
    #
    # Get search results -
    # 1. Sort answer entities
    # 2. Sample answer entities
    #
    #
    # ------------------------------------------------------

    for i, ae in enumerate(aes):
        ae_len = len(ae[0].split(' '))
        pred = ans_pred[str(i)]
        ae_dict[ae] = [
            ans_pair_metric(ae[0], pred['text'], None)[1] * 10000 + ae[-1],
            ae_log_probs[i].item(),
            pred['gap'],
            - (pred['start_logit'] + pred['end_logit']),
            pred['probability'],
            pred['top_ans'],
            (pred['text'], pred['offsets'][0]),
            gen_ques_text[i],
        ] # + ae_len * 10
        if ae_dict[ae][0] > 1000:
            ae_dict[ae][0] += ae_len * 100

    ae_sorted = sorted(
        ae_dict.items(),
        key = lambda x: x[1][3] / x[1][4],
        reverse = True
    )
    ae_sorted = [list(x) for x in ae_sorted]

    ans_ppl_list = [v[3] for k, v in ae_sorted]

    ans2item_dict = {}
    for k, v in ae_sorted:
        ans_pred = v[-2]
        if ans_pred in ans2item_dict:
            ans2item_dict[ans_pred].append([k, v])
        else:
            ans2item_dict[ans_pred] = [[k, v]]

    ans2item_list = [[k, v] for k, v in ans2item_dict.items()]

    '''
    if True:
        print(squad['context'])
        print(squad['title'])

        for i in range(len(ans2item_list)):
            k, v = ans2item_list[i]
            print('\n')
            print(k)
            for kk, vv in v:
                print(kk)
                print(vv)
                print('----------------------------')
        # print(ans_ppl_list)
        print('\n###############################################')

        for x in ae_selected:
            print(x)
            print('---------------------')

        abort()
    # '''

    # return [squad, ae_selected]
    return ans2item_list


def count_pair_punkt(input_txt):
    if input_txt.count('(') != input_txt.count(')'):
        return False
    if input_txt.count('"') % 2 != 0:
        return False
    return True


def valid_ans_item(sq_case, ans_item):
    ae, qg = ans_item
    if ae[0] in ae[1] or ae[1] in ae[0]:
        return False
    ans_pred, ans_st = qg[-2]
    ans_pred = ans_pred.lower()
    ques_gen = qg[-1].lower()
    if 'name' in qg[-1]:
        return False

    ctx = sq_case['context']
    ans_words = word_tokenize(ans_pred)
    ctx_words = word_tokenize(ctx.lower())
    num_ans_words = len(ans_words)

    st_tok = -1
    ed_tok = -1
    for i in range(len(ctx_words)):
        if ctx_words[i: i + num_ans_words] == ans_words:
            st_tok = i - 1
            ed_tok = i + num_ans_words
            break

    if ans_pred.lower() in ques_gen.lower() and ans_pred != '':
        return False
    if not count_pair_punkt(ans_pred) or not count_pair_punkt(ques_gen):
        return False
    if f'{ctx_words[st_tok]} {ctx_words[ed_tok]}' in ques_gen and ans_pred != '':
        return False
    return True


def filter_ans2item(sq_case, ans2item):
    for i in range(len(ans2item)):
        # print(ans2item[i])
        # abort()
        ans2item[i][1] = [x for x in ans2item[i][1] if valid_ans_item(sq_case, x)]
        # print(len(ans2item[i][1]))
    ans2item = [x for x in ans2item if len(x[1]) > 0]
    return ans2item


def search_qa(squad, tok, model, ext_tok, ext_model, case_id,
              model_args, data_args, training_args):

    sq_case, aer_list = squad
    ans2item = []

    num_iter = 0
    while len(aer_list) > 0:
        try:
            new_ans2item = get_squad_ppl([sq_case, aer_list], tok, model, ext_tok, ext_model, case_id,
                                         model_args, data_args, training_args)

            # json.dump([sq_case, new_ans2item], open('log/test.json', 'w'))
            # new_ans2item = filter_ans2item(sq_case, new_ans2item)
            ans2item += new_ans2item
            # ans2item = merge_ans2item(ans2item)
            # aer_list = new_ae_detect(ans2item)
        except:
            abort()
            break
        num_iter += 1
        if num_iter > 0:
            break

    for i in range(len(ans2item)):
        ans2item[i][1] = sorted(ans2item[i][1], key = lambda x: x[1][3], reverse=True)

    new_squad = [sq_case, ans2item]
    return new_squad

if __name__ == '__main__':

    data = json.load(open(f'splits/{sys.argv[2]}/squad_aer_{sys.argv[1]}.json', 'r')) #[9: 10]
    tok = BartTokenizerFast.from_pretrained('model_file/bart-tokenizer.pt')

    model = BartForConditionalGeneration.from_pretrained(
        # 'model_ft_file/ques_gen_squad_v2_cfs_ft.pt'
        'model_ft_file/ques_gen_squad_test.pt'
    ).cuda()

    ext_tok = ElectraTokenizerFast.from_pretrained("model_file/electra-tokenizer.pt")
    ext_model = ElectraForQuestionAnswering.from_pretrained(
        'model_file/ext_sqv2.pt',
        # 'model_file/ext_nq.pt',
        return_dict=True
    )
    ext_model = ext_model.cuda()
    ext_model.train()

    squad_new_complete = []
    print(f'Processing {len(data)} SQuAD passages')

    for i, squad in enumerate(data):
        sq_case, aer_list = squad
        ans2item = []

        num_iter = 0
        while len(aer_list) > 0:
            try:
                new_ans2item = get_squad_ppl([sq_case, aer_list], tok, model, ext_tok, ext_model, i)
                # print(len(new_ans2item))
                ans2item += new_ans2item
                # ans2item = merge_ans2item(ans2item)
                # aer_list += new_ae_detect(ans2item)
            except:
                break
            num_iter += 1
            if num_iter > 0:
                break

        if len(ans2item) > 0:
            # print(len(ans2item))
            # abort()
            # ans2item = merge_ans2item(ans2item)
            new_squad = [sq_case, ans2item]
            squad_new_complete.append(new_squad)

        if i % 100 == 0:
            print(f'Processed {i} passages')

    json.dump(
        squad_new_complete,
        open(f'splits_ft/{sys.argv[2]}/squad_aer_reranked_{sys.argv[1]}_rec.json', 'w')
    )
    print(f'Split {sys.argv[1]} processing finished')
