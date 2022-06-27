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
from rgx_unit_test import *
from run_qa_func import run_qa, load_args
from qgen_merge_new import ans_pair_metric

import stanza
import datasets
# datasets.set_progress_bar_enabled(False)

from nltk.corpus import stopwords
from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    ElectraTokenizerFast,
    ElectraForQuestionAnswering
)

en_sw = stopwords.words('english')
en_sw = set(en_sw)

question_words = set([
    'what',
    'who',
    'where',
    'which',
    'why',
    'when',
    'how'
])

def generate_ques(input_txt, tok, model, do_sample=True, batch_size=16, max_length=512):
    gen_ques_text = []
    for i in range(0, len(input_txt), batch_size):
        batch_txt = input_txt[i: i + batch_size]
        
        inputs = tok(
            batch_txt,
            max_length = max_length,
            truncation = True,
            return_attention_mask = True,
            padding = 'longest',
            return_tensors = 'pt'
        )

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids = inputs['input_ids'].cuda(),
                num_beams=5,
                max_length=64,
                early_stopping=True,
                attention_mask = inputs['attention_mask'].cuda(),
                do_sample=do_sample
            )

        gen_ques_text += [
            tok.decode(g, skip_special_tokens=True) for g in gen_ids #['sequences']
        ]

    return gen_ques_text


def is_sw(word, en_sw):
    if word[0] == 'Ġ':
        word = word[1:]
    return word.lower() in en_sw


def get_sent_loss(token_list, loss_list):
    global en_sw
    word_loss_list = []
    cur_word = []

    for i, token in enumerate(token_list):
        if token[0] != 'Ġ' and token[-2:] != 's>':
            cur_word.append((token, loss_list[i]))
        elif len(cur_word) > 0:
            loss = max([x[1] for x in cur_word])
            word = ''.join([x[0] for x in cur_word])
            if not is_sw(word, en_sw):
                word_loss_list.append((''.join([x[0] for x in cur_word]), loss))
            cur_word = [(token, loss_list[i])]
        else:
            cur_word = [(token, loss_list[i])]
    
    loss = max([x[1] for x in cur_word])
    word = ''.join([x[0] for x in cur_word])
    if not is_sw(word, en_sw):
        word_loss_list.append((''.join([x[0] for x in cur_word]), loss))

    return word_loss_list


def get_ques_loss(input_txt, ques_labels, tok, model, loc_weight_dict=None,
                    batch_size=16, qe_txt=None, max_length=512):
    all_loss_list = []

    if qe_txt is None:
        qe_id_set = set([])
    else:
        qe_ids = tok(qe_txt)['input_ids'][1: -1]
        qe_ids += tok(f'a {qe_txt}')['input_ids'][2: -1]
        qe_ids += tok(qe_txt.lower())['input_ids'][1: -1]
        qe_ids += tok(f'a {qe_txt.lower()}')['input_ids'][2: -1]
        qe_id_set = set(qe_ids)
    
    for i in range(0, len(input_txt), batch_size):
        batch_txt = input_txt[i: i + batch_size]
        ques_batch = ques_labels[i: i + batch_size]
        cur_bs = len(batch_txt)

        inputs = tok(
            batch_txt,
            max_length = 512,
            truncation = True,
            return_attention_mask = True,
            return_offsets_mapping=True,
            padding = 'longest',
            return_tensors = 'pt'
        )

        ques_enc = tok(
            ques_batch,
            max_length = 128,
            truncation = True,
            return_attention_mask = True,
            padding = 'longest',
            return_tensors = 'pt'
        )

        ques_id_list = ques_enc['input_ids'][0]
        qe_mask = torch.ones_like(ques_id_list)

        for j in range(ques_id_list.size(0)):
            if ques_id_list[j].item() in qe_id_set:
                qe_mask[j] = 0
        
        ctx_input_ids = inputs['input_ids'].cuda()
        ctx_attn_mask = inputs['attention_mask'].cuda()
        ques_input_ids = ques_enc['input_ids'].cuda()
        ques_attn_mask = ques_enc['attention_mask'].cuda()

        ques_attn_mask *= qe_mask.cuda().unsqueeze(0)

        with torch.no_grad():
            results = model(
                input_ids = ctx_input_ids,
                attention_mask = ctx_attn_mask,
                labels = ques_input_ids
            )
        
        loss_tensor = results.loss.view(cur_bs, -1)
        tokens = tok.convert_ids_to_tokens(ques_input_ids[0])
        '''
        # loss_tensor = loss_tensor / loss_tensor.min(0)[0]
        loss_list = []
        
        for c in range(cur_bs):
            loss_t = loss_tensor[c].tolist()
            word_loss_list = get_sent_loss(tokens[1:], loss_t[1:])
            loss_list.append([x[1] for x in word_loss_list])
            # print(word_loss_list)
            # print('')
        
        loss_tensor = torch.Tensor(loss_list).cuda()
        loss_tensor = loss_tensor / loss_tensor.min(0)[0]

        # print(loss_tensor)

        loss_tensor = loss_tensor.mean(1)
        # print(loss_tensor)
        # sys.exit()
        # '''
        # loss_tensor = (loss_tensor * ques_attn_mask).sum(1) / ques_attn_mask.sum(1)
        # loss_tensor = loss_tensor[:, 1:].mean(1)

        if loc_weight_dict is None:
            loss_tensor = loss_tensor * ques_attn_mask
            loss_tensor = loss_tensor.sum(1) / ques_attn_mask.sum(1)
        
        all_loss_list.append(loss_tensor)
    
    all_loss_list = torch.cat(all_loss_list, dim=0)

    if loc_weight_dict is not None:
        for k, v in loc_weight_dict.items():
            for j, w in v.items():
                all_loss_list[k, j] *= 1 - w
        all_loss_list = all_loss_list[:, 1:].mean(1)

    return all_loss_list


def get_masked_ctx(squad, aer):
    ae_txt, ae_st = aer
    ae_ed = ae_st + len(ae_txt)
    context = squad['context']
    return f'{context[:ae_st]}<mask>{context[ae_ed:]} </s> {ae_txt}'


def get_new_squad(context, question='', ans_txt='', ans_st=0, sq_id=''):
    new_squad = {
        'context': context,
        'question': question,
        'answers' : {
            'text': [ans_txt],
            'answer_start': [ans_st]
        },
        'id': sq_id
    }
    return new_squad


def get_span_dict(context, sent_info):
    # doc = nlp_pipeline(context)
    words_info_list = sent_info.words
    d2u_dict = {}
    u2d_dict = {}
    
    for i, word in enumerate(words_info_list):
        head_idx = word.head - 1
        d2u_dict[i] = head_idx

        if head_idx not in u2d_dict:
            u2d_dict[head_idx] = [i]
        else:
            u2d_dict[head_idx].append(i)
        u2d_dict[head_idx] = sorted(u2d_dict[head_idx])
    
    for i, word in enumerate(words_info_list):
        head_idx = d2u_dict[i]
        while head_idx != -1:
            new_head = d2u_dict[head_idx]
            u2d_dict[new_head].append(i)
            head_idx = new_head
    
    for k in u2d_dict:
        u2d_dict[k].append(k)
        u2d_dict[k] = sorted(list(set(u2d_dict[k])))
    
    all_spans = {}
    span_tree = {}
    span_head_dict = {}

    for i, word in enumerate(words_info_list):
        if i not in u2d_dict:
            span_head_dict[(word.start_char, word.end_char)] = i
            continue
        
        cur_span = u2d_dict[i]
        span_st_char = words_info_list[cur_span[0]].start_char
        span_ed_char = words_info_list[cur_span[-1]].end_char
        span_offset = (span_st_char, span_ed_char)
        
        span_head_dict[span_offset] = (word.start_char, word.end_char)
        
        if word.head == 0:
            all_spans[span_offset] = 'root'
            span_tree['root'] = span_offset
            continue

        head_st = words_info_list[word.head - 1].start_char
        head_ed = words_info_list[word.head - 1].end_char

        up_span = u2d_dict[word.head - 1]
        up_span_st_char = words_info_list[up_span[0]].start_char
        up_span_ed_char = words_info_list[up_span[-1]].end_char
        
        left_offset = (up_span_st_char, head_ed)
        right_offset = (head_st, up_span_ed_char)
        up_span_offset = (up_span_st_char, up_span_ed_char)

        span_head_dict[left_offset] = (head_st, head_ed)
        span_head_dict[right_offset] = (head_st, head_ed)
        span_head_dict[up_span_ed_char] = (head_st, head_ed)

        # all_spans[left_offset] = up_span_offset
        if up_span_offset not in span_tree:
            span_tree[up_span_offset] = [left_offset, right_offset]
            if left_offset != up_span_offset:
                all_spans[left_offset] = up_span_offset
            if right_offset != up_span_offset:
                all_spans[right_offset] = up_span_offset
        
        if left_offset not in span_tree:
            span_tree[left_offset] = []
        if right_offset not in span_tree:
            span_tree[right_offset] = []
        
        if span_ed_char <= head_ed:
            span_tree[left_offset].append(span_offset)
            if span_offset != left_offset:
                all_spans[span_offset] = left_offset
        elif span_st_char >= head_st:
            span_tree[right_offset].append(span_offset)
            if span_offset != right_offset:
                all_spans[span_offset] = right_offset
        else:
            span_tree[up_span_offset].append(span_offset)
            if span_offset != up_span_offset:
                all_spans[span_offset] = up_span_offset
    
    ae_spans = [(context[x: y], x) for x, y in all_spans.keys()]
    for k, v in span_tree.items():
        if k == 'root':
            continue
        span_tree[k] = [x for x in v if x != k]
    # abort()
    return ae_spans, all_spans, span_tree, span_head_dict


def span_contain(a_offset, b_offset):
    return a_offset[0] >= b_offset[0] and a_offset[1] <= b_offset[1]


def search_ae_path(aer, span_tree, span_head_dict, rgx_data):

    if aer in rgx_data:
        return rgx_data[aer]['path']
    
    ae_txt, ae_st = aer
    ae_ed = ae_st + len(ae_txt)
    all_neg = False
    cur_offset = span_tree['root']

    path = []
    path_set = set([])
    depth = 0

    # print(f'Searching for span {(ae_st, ae_ed)}')
    
    while not all_neg:
        all_neg = True
        path.append({
            'offset': cur_offset, 'depth': depth, 'head': [span_head_dict[cur_offset]]
        })
        path_set.add(cur_offset)
        
        # print(cur_offset)
        if cur_offset not in span_tree:
            break
        
        for son_offset in span_tree[cur_offset]:
            if span_contain((ae_st, ae_ed), son_offset):
                if son_offset in path_set:
                    print(aer)
                    print(son_offset)
                    print(path)
                    abort()
                all_neg = False
                cur_offset = son_offset
                depth += 1
                break
    
    path_len = len(path)
    for i in range(len(path)):
        if i < path_len - 1:
            path[i]['next'] = path[i+1]['offset']
        else:
            path[i]['next'] = None
    return path


def mask_context_span(context, ae_ctx):
    ae_txt, ae_st = ae_ctx['ae']
    ae_ed = ae_st + len(ae_txt)
    ae_offset = (ae_st, ae_ed)

    ctx_st, ctx_ed = ae_ctx['ctx']['offset']
    ctx_depth = ae_ctx['ctx']['depth']

    next_span = ae_ctx['ctx']['next']
    hw_list = ae_ctx['ctx']['head']
    hw_list = [x for x in hw_list if next_span is None or span_contain(x, next_span)]
    ctx_txt = context[ctx_st: ctx_ed]

    if next_span is None:
        ae_ctx_st = ae_st - ctx_st
        ae_ctx_ed = ae_ctx_st + len(ae_txt)
        ctx_masked = f'{ctx_txt[:ae_ctx_st]}<mask>{ctx_txt[ae_ctx_ed:]} </s> {ae_txt}'
    else:
        next_st, next_ed = next_span
        next_ctx_st = next_st - ctx_st
        next_ctx_ed = next_ed - ctx_st
        perserve_words = sorted(
            list(set(hw_list + [ae_offset])), key = lambda x: x[0]
        )
        pword_txt = [
            context[st: ed] if (st, ed) != ae_offset else '<mask>' for st, ed in perserve_words
        ]
        replace_txt = ' '.join(pword_txt)
        ctx_masked = f'{ctx_txt[:next_ctx_st]}{replace_txt}{ctx_txt[next_ctx_ed:]} </s> {ae_txt}'

    # print(ae_ctx)
    # print(ctx_txt)
    # print(ctx_masked)
    # abort()
    return ctx_masked


def get_span_txt(context, offset):
    st, ed = offset
    return context[st: ed]


def get_squad_ppl(squad, path_list, rgx_data, tok, model, ext_tok, ext_model, case_id,
                    span_tree, nlp_pipeline, model_args, data_args, training_args):
    
    # Collect all spans and initialize new AE list
    all_offsets = set([
        (k[1], k[1] + len(k[0])) for k, v in path_list.items()
    ])
    all_offsets.update([
        (k[1], k[1] + len(k[0])) for k, v in rgx_data.items()
    ])
    new_ae_list = []
    ctx_len_char = len(squad['context'])
    
    # Generate questions for all AE + Span pairs
    ae_ctx_offset_list = []
    for ae, path in path_list.items():
        ae_t, ae_st = ae
        ae_ed = ae_st + len(ae_t)
        
        path_len = len(path)
        cur_heads = []

        for i in range(path_len):
            path_idx = path_len - 1 - i

            if len(path[path_idx]['head']) > 0 and span_contain(path[path_idx]['head'][0], (ae_st, ae_ed)):
                path[path_idx]['head'] = []
                
            
            path[path_idx]['head'] += cur_heads
            cur_heads = path[path_idx]['head']
            
        path = [{'offset': (0, ctx_len_char), 'depth': '0', 'next': None, 'head': []}] + path
        ae_ctx = [{'ae': ae, 'ctx': x} for x in path]
        ae_ctx_offset_list += ae_ctx
    
    masked_ctx_list = [
        mask_context_span(squad['context'], x) for x in ae_ctx_offset_list[:]
    ]
    # print('')
    # print(len(masked_ctx_list))
    # print('')
    ques_gen = generate_ques(masked_ctx_list, tok, model, do_sample=False)

    '''for i in range(len(masked_ctx_list)):
        print(masked_ctx_list[i])
        print(ques_gen[i])
        print('------------------------\n')
    print(len(ques_gen))
    abort()'''
    
    # Merge generated question based on AEs
    ae_ques_dict = {}
    for i, ae_ctx in enumerate(ae_ctx_offset_list):
        ae = ae_ctx['ae']
        ques_ae = ques_gen[i]
        if ae in ae_ques_dict:
            ae_ques_dict[ae][ques_ae] = -1
        else:
            ae_ques_dict[ae] = {ques_ae: -1}
    
    # answer generated questions recursively on the tree
    depth = 0
    max_depth = max([len(v) for k, v in path_list.items()])
    rgx_data_new = {}

    for depth in range(max_depth):
        aqd_list = []
        squad_gen_list = []
        for ae, path in path_list.items():
            if depth > len(path) - 1:
                continue
            cur_span_st, cur_span_ed = path[depth]['offset']
            span_txt = squad['context'][cur_span_st: cur_span_ed]
            
            aqd = [
                (ae, depth, q) for q, v in ae_ques_dict[ae].items() if v == -1
            ]
            if len(aqd) == 0:
                continue

            ae_squad = [
                get_new_squad(
                    get_span_txt(squad['context'], path[depth]['offset']),
                    x[2]
                ) for x in aqd
            ]
            
            aqd_list += aqd
            squad_gen_list += ae_squad
        
        if len(squad_gen_list) == 0:
            continue

        try:
            _, ans_pred = run_qa(
                squad_gen_list, ext_tok, ext_model, model_args, data_args, training_args
            )
        except:
            print(f'case_id = {case_id}')
            print(len(aqd))
            print(len(squad_gen_list))
            abort()

        for j in range(len(aqd_list)):
            aer, depth, q = aqd_list[j]
            ae_txt, ae_st = aer
            ae_ed = ae_st + len(ae_txt)
            ans_pred_txt = ans_pred[str(j)]['text']
            ans_pred_st, ans_pred_ed = ans_pred[str(j)]['offsets']
            
            cur_span = path_list[aer][depth]
            cur_span_st, cur_span_ed = cur_span['offset']
            ans_pred_st_psg = ans_pred_st + cur_span_st
            ans_pred_ed_psg = ans_pred_ed + cur_span_st

            aer_pred = (ans_pred_txt, ans_pred_st_psg)

            '''
            if aer_pred in path_list:
                ae_ques_dict[aer_pred][q] = depth
            elif aer_pred in rgx_data_new:
                rgx_data_new[aer_pred]['ques_val'][q] = depth
            elif aer_pred in rgx_data:
                rgx_data[aer_pred]['ques_val'][q] = depth
            else:
                rgx_data_new[aer_pred] = {
                    'path': search_ae_path(aer_pred, span_tree, rgx_data),
                    'ques_val': {q: depth}
                }
            '''
            
            if ae_st == ans_pred_st_psg and ae_ed == ans_pred_ed_psg:
                ae_ques_dict[aer][q] = depth
            
            if (ans_pred_st_psg, ans_pred_ed_psg) not in all_offsets:
                new_ae_txt = squad['context'][ans_pred_st_psg: ans_pred_ed_psg]
                new_ae_list.append((new_ae_txt, ans_pred_st))
    
    for ae in path_list:
        rgx_data_new[ae] = {
            'path': path_list[ae],
            'ques_val': ae_ques_dict[ae]
        }
    return rgx_data_new, rgx_data, new_ae_list


def search_qa(squad, tok, model, ext_tok, ext_model, case_id,
              nlp_pipeline, model_args, data_args, training_args):
    
    def update_rgx_data(rgx_data, new_data):
        for ae, rgx_dict in new_data.items():
            if ae in rgx_data:
                rgx_data[ae]['ques_val'].update(rgx_dict['ques_val'])
            else:
                rgx_data[ae] = rgx_dict
        return rgx_data
    
    def certified_ae(rgx_item):
        ae, rgx_dict = rgx_item
        certified = False
        rgx_dict['ques_val'] = {
            q: v for q, v in rgx_dict['ques_val'].items() if v > -1
        }
        return (ae, rgx_dict)

    sq_case, aer_list_raw = squad
    sent_info = nlp_pipeline(sq_case['context']).sentences[0]
    ae_spans, all_spans, span_tree, span_head_dict = get_span_dict(sq_case['context'], sent_info)

    if 'root' not in span_tree:
        return [sq_case, []]

    aer_list = [(x[0], x[4]) for x in aer_list_raw]

    new_aer_list = list(set(
        aer_list + ae_spans
    ))

    ans2item = []
    rgx_data = {}

    num_iter = 0
    while len(new_aer_list) > 0:
        try:
            path_list = {x: search_ae_path(x, span_tree, span_head_dict, rgx_data) for x in new_aer_list}
        except:
            print(sq_case)
            print(num_iter)
            print(span_tree)
            print(case_id)
            json.dump(list(span_tree.items()), open('log/test.json', 'w'))
            abort()

        new_rgx_data, rgx_data, new_aer_list = get_squad_ppl(
            sq_case, path_list, rgx_data, tok, model, ext_tok, ext_model,
            case_id, span_tree, nlp_pipeline, model_args, data_args, training_args
        )
        rgx_data = update_rgx_data(rgx_data, new_rgx_data)

        num_iter += 1
        if num_iter > 2:
            break
    
    rgx_data_sorted = sorted(rgx_data.items(), key = lambda x: x[0][1], reverse = False)
    rgx_data_sorted = [certified_ae(x) for x in rgx_data_sorted]

    rgx_data_cert = [x for x in rgx_data_sorted if len(x[1]['ques_val']) > 0]

    new_squad = [sq_case, rgx_data_cert]
    return new_squad


def search_qa_sample(squad, tok, model, ext_tok, ext_model, case_id,
              nlp_pipeline, model_args, data_args, training_args):
    
    num_samples = 10
    sq_case, aer_list_raw = squad
    aer_list = [(x[0], x[4]) for x in aer_list_raw]
    all_aer_list = []

    for aer in aer_list:
        all_aer_list += [aer for i in range(num_samples)]
    
    masked_ctx_list = [get_masked_ctx(sq_case, aer) for aer in all_aer_list]
    ques_gen_list = generate_ques(
        masked_ctx_list, tok, model, do_sample=True, batch_size=32
    )

    squad_gen_list = [
        get_new_squad(sq_case['context'], x) for x in ques_gen_list
    ]

    _, ans_pred = run_qa(
        squad_gen_list, ext_tok, ext_model, model_args, data_args, training_args
    )

    verified_dict = {}
    for i, aer in enumerate(all_aer_list):
        ae_txt, ae_st = aer
        ae_ed = ae_st + len(ae_txt)
        ques_gen = ques_gen_list[i]

        ans_pred_txt = ans_pred[str(i)]['text']
        ans_pred_st, ans_pred_ed = ans_pred[str(i)]['offsets']

        verify_condition = (ae_st, ae_ed) == (ans_pred_st, ans_pred_ed)

        if verify_condition:
            aer_pred = (ans_pred_txt, ans_pred_st)

            if aer_pred in verified_dict:
                verified_dict[aer_pred]['ques_val'][ques_gen] = 0
            else:
                verified_dict[aer_pred] = {
                    'ques_val': {ques_gen: 0}
                }
    
    rgx_data_sorted = sorted(verified_dict.items(), key = lambda x: x[0][1])
    
    new_squad = [sq_case, rgx_data_sorted]
    return new_squad


if __name__ == '__main__':
    
    model_args, data_args, training_args = load_args()
    nlp_pipeline = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse')

    data = json.load(open(
        f'splits/{data_args.dataset_name}/squad_aer_{data_args.data_split}.json', 'r'
    )) #[389: 390]
    tok = BartTokenizerFast.from_pretrained('model_file/bart-tokenizer.pt')

    model = BartForConditionalGeneration.from_pretrained(
        'model_file/ques_gen_squad.pt'
        # 'model_ft_file/ques_gen_squad_test.pt'
    ).cuda()
    model.eval()

    ext_tok = ElectraTokenizerFast.from_pretrained("model_file/electra-tokenizer.pt")
    ext_model = ElectraForQuestionAnswering.from_pretrained(
        'model_file/ext_sq.pt',
        # 'model_file/ext_nq.pt',
        return_dict=True
    )
    ext_model = ext_model.cuda()
    ext_model.eval()
    
    training_args.do_train = False
    training_args.do_eval = True
    training_args.disable_tqdm=True

    squad_new_complete = []
    print(f'Processing {len(data)} SQuAD passages')

    for i, squad in enumerate(data):
        print(f'Processing passage {i}')
        squad_new_complete.append(
            search_qa(
                squad, tok, model, ext_tok, ext_model, i,
                nlp_pipeline, model_args, data_args, training_args
            )
        )

        if i % 100 == 0:
            print(f'Processed {i} passages')

    # print(squad_new_complete[0][1][0])
    json.dump(
        squad_new_complete,
        open(
            f'splits_ft/{data_args.dataset_name}/squad_rgx_{data_args.data_split}_dpt_149.json', 'w'
        )
    )
    print(f'Split {data_args.dataset_name} processing finished')