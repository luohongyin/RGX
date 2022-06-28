import sys
import json
import copy
import random
import stanza


import torch
import torch.nn as nn

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import TreebankWordTokenizer as twt

from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    ElectraTokenizerFast,
    ElectraForQuestionAnswering,
    logging
)

from search_coop_recursive_trainer import search_qa
from ques_gen_ft import train_ques_gen
from ques_pretrain_data import load_data
from run_qa_func import load_args, run_qa
from noans_aug import span_tokenize, common_entities

from mix_model import mix
from search_rgx import get_span_dict

from stanza.pipeline.core import DownloadMethod
from aer_ext_deptree import aer

logging.set_verbosity_error()
stop_words = set(stopwords.words('english'))

def ans_loc_in_sent(offset_mapping, ans, st_char, tokenizer):

    start_loc = 0
    end_loc = 0
    ans_enc = tokenizer(ans, return_offsets_mapping=True)

    ans_len = len(ans)
    ctx_num_tok = len(offset_mapping)

    st_char = st_char
    ed_char = st_char + ans_len - 1
    for j in range(1, ctx_num_tok):
        cur_offset = offset_mapping[j]
        if st_char >= cur_offset[0] and st_char < cur_offset[1]:
            start_loc = j
        if ed_char >= cur_offset[0] and ed_char < cur_offset[1]:
            end_loc = j
            break

    if start_loc > 447:
        start_loc = 0
        end_loc = 0
    end_loc = min(end_loc, 447)

    return start_loc, end_loc, offset_mapping[start_loc][0], offset_mapping[end_loc][1]


def ans_sent_id(ans_st_char, sent_offsets):
    # sents = sent_tokenize(ctx)
    # sent_offsets = [ctx.index(x) for x in sents]
    sent_id = len(sents) - 1
    for i, sent_st in enumerate(sent_offsets):
        if sent_st > ans_st_char:
            sent_id = i
            break
    return sent_id, sents


def ans_loc(ans_info, sent_enc, sent_offsets, tok):
    ans_txt, ans_st_char = ans_info
    sent_id, sents = ans_sent_id(ans_st_char, sent_offsets)
    tok_offset_mapping = sent_enc['offset_mapping'][sent_id]
    ans_st_sent = ans_st_char - sent_offsets[sent_id]
    st_tok, ed_tok, st_char, ed_char = ans_loc_in_sent(tok_offset_mapping, ans_txt, ans_st_sent, tok)
    return [ans_txt, sent_id, st_tok, ed_tok, ans_st_char]


def load_model(qg_path, qa_path):
    tok = BartTokenizerFast.from_pretrained('model_file/bart-tokenizer.pt')
    model = BartForConditionalGeneration.from_pretrained(qg_path)

    ext_tok = ElectraTokenizerFast.from_pretrained("model_file/electra-tokenizer.pt")
    ext_model = ElectraForQuestionAnswering.from_pretrained(qa_path, return_dict=True)

    return tok, model, ext_tok, ext_model


def load_playground(dataset, mode='sample'):
    # dataset = json.load(open(data_path))
    if mode == 'all':
        return dataset
    elif mode == 'sample':
        return random.sample(dataset)
    else:
        return dataset[mode]


def build_ctx_word_dict(ctx, nltk_tok):
    ctx_words, ctx_offsets = span_tokenize(ctx)

    word_dict = {}
    psg_len = len(ctx_words)

    for i in range(psg_len):
        for j in range(i, min(i + 10, psg_len)):
            span = ctx[ctx_offsets[i][0]: ctx_offsets[j][1]]
            if span in word_dict:
                word_dict[span].append(ctx_offsets[i][0])
            else:
                word_dict[span] = [ctx_offsets[i][0]]

    return word_dict


def merge_list(l1, l2):
    return list(set(l1 + l2))


def build_entity_map(squad_aer_reranked, tok, ctx_word_dict):
    global stop_words
    sq, ans2item_list = squad_aer_reranked
    ctx = sq['context']
    ctx_words = word_tokenize(ctx.lower())

    sents = sent_tokenize(ctx)
    sent_offsets = [ctx.index(x) for x in sents]
    sent_enc = tok(sents, return_offsets_mapping=True)

    entity_list = []
    ques_list = []

    for ans_item in squad_aer_reranked[1][1:]:
        entity_list += [qg[-2] for ae, qg in ans_item[1]]
        ques_list += [qg[-1] for ae, qg in ans_item[1]]

    entity_list = [(x.lower(), y) for x, y in entity_list]
    ae_list = list(set([(x, y) for x, y in entity_list]))
    ques_list = list(set(ques_list))

    ques2ent_dict = {}
    ent2ques_dict = {}
    for i, question in enumerate(ques_list):
        qe_list = common_entities(ctx.lower(), ctx_word_dict, question.lower(),
                                       entity_list[i], stop_words, ae_list)

        if question not in ques2ent_dict:
            ques2ent_dict[question] = qe_list
        else:
            ques2ent_dict[question] = merge_list(ques2ent_dict[question], qe_list)

        for qe in qe_list:
            if qe[0] in ent2ques_dict:
                ent2ques_dict[qe[0]].append(question)
            else:
                ent2ques_dict[qe[0]] = [question]

    for k in ent2ques_dict:
        sorted_v = sorted(ent2ques_dict[k], key = lambda x: len(ques2ent_dict[x]), reverse=True)
        ent2ques_dict[k] = sorted_v

    for k in ques2ent_dict:
        sorted_v = sorted(ques2ent_dict[k], key = lambda x: len(x), reverse=True)
        ques2ent_dict[k] = sorted_v

    return ques2ent_dict, ent2ques_dict, ae_list


def get_key(query, ent2ques_dict):
    try:
        value = ent2ques_dict[query]
        return value
    except:
        for k in sorted(list(ent2ques_dict.keys()), key = lambda x: len(x), reverse=False):
            if query in k:
                return ent2ques_dict[k]
        return []


def overlap_questions(question, ctx, ctx_word_dict, ques2ent_dict, ent2ques_dict,
                      ae_list, init=False, visited_ent=None):
    global stop_words
    ques_link = []

    if init:
        qe_list = common_entities(ctx.lower(), ctx_word_dict, question.lower(), None, stop_words, ae_list)
    else:
        qe_list = ques2ent_dict[question]

    for qe in qe_list:
        if visited_ent is not None and qe[0] in visited_ent and qe[0] != '':
            continue
        ques_link += get_key(qe[0], ent2ques_dict)
    ques_link = list(set([x for x in ques_link if x != question]))
    ques_link = sorted(ques_link, key = lambda x: len(ques2ent_dict[x]), reverse=False)
    return ques_link, qe_list


def covered(prev_path, ret_qe_list, init_qe_list, ques2ent_dict):
    ret_qe_set = set([x[0] for x in ret_qe_list])
    '''
    print('\n')
    print(prev_path)
    print('--')
    print(ret_qe_set)
    print(init_qe_list)
    print('--')
    for x in prev_path[1:]:
        print(x)
        print(ques2ent_dict[x])
    print('---------------')
    # '''
    for qe in init_qe_list:
        if qe[0] == '':
            continue
        found = 0
        if qe[0] in ret_qe_set:
            continue
        for k in ret_qe_set:
            if qe[0] in k:
                found = 1
                break
        if found == 0:
            return 0
    return 1


def answer_search(ques, ctx, ctx_word_dict, nlp_pipeline, ques2ent_dict, ent2ques_dict, ae_list):
    step_id = 0
    ques_link_dict = {}
    ques_list = [{(ques,): {'qe_list': None, 'found': False}}]
    # prev_ques_set = set([ques])
    path_list = []

    ret_path_dict = {}

    while step_id < 5:
        new_ques_dict = {}
        found = False

        for prev_path in ques_list[-1]:
            if ques_list[-1][prev_path]['found']:
                continue
            if len(prev_path) <= 1:
                visited_ent = None
            else:
                visited_ent = set([x[0] for x in ques_list[-2][prev_path[:-1]]['qe_list']])

            ques_link, qe_list = overlap_questions(
                prev_path[-1], ctx, ctx_word_dict, ques2ent_dict, ent2ques_dict,
                ae_list, step_id == 0, visited_ent
            )
            ques_list[-1][prev_path]['qe_list'] = copy.deepcopy(qe_list)

            if step_id > 1:
                pp_path = prev_path[:-1]
                ques_list[-1][prev_path]['qe_list'] += ques_list[-2][pp_path]['qe_list']

            if step_id > 0:
                crate = covered(
                    prev_path, ques_list[-1][prev_path]['qe_list'],
                    ques_list[0][(ques,)]['qe_list'],
                    ques2ent_dict
                )

                if crate == 1:
                    ret_path_dict[prev_path] = ques_list[-1][prev_path]['qe_list']
                    ques_list[-1][prev_path]['found'] = True
                    found = True

            new_ques = [x for x in ques_link]
            # new_ques = [x for x in ques_link if x not in prev_ques_set]
            for nq in new_ques:
                new_path = prev_path + (nq,)
                new_ques_dict[new_path] = {'qe_list': None, 'found': found}
                # prev_ques_set.add(nq)

        print(len(new_ques_dict))
        ques_list.append(new_ques_dict)

        # if found:
        #     break

        step_id += 1
    if len(ret_path_dict) == 0:
        print('Not found')
    for k, v in ret_path_dict.items():
        for q in k[1:]:
            print(q)
            print(ques2ent_dict[q])
            print('--')
        print('---------------\n')
    return 'Not found'


def play_case(
        dataset, psg_info_list, rgx_data,
        tok, model, ext_tok, ext_model, split_id,
        num_epoch, batch_size, psg_id, model_args,
        data_args, do_train_args, do_eval_args, nlp_pipeline,
        eval_with_questions = False
    ):

    def get_ae_from_rgx(psg_txt, rgx_list):
        ae_list = []

        for rgx_case in rgx_list:
            sq, ae_qg_list = rgx_case
            sent_txt = sq['context']
            sent_st = psg_txt.index(sent_txt)

            ans_list = [x[0] for x in ae_qg_list]
            aer_reloc = [(ae_t, ae_st + sent_st) for ae_t, ae_st in ans_list]
            ae_list += aer_reloc

        return ae_list

    def get_ae_from_aer(psg_txt, rgx_list):
        ae_list = []

        for rgx_case in rgx_list:
            sq, aer_list_raw = rgx_case
            sent_txt = sq['context']

            sent_info = nlp_pipeline(sent_txt).sentences[0]
            ae_spans, all_spans, span_tree, span_head_dict = get_span_dict(
                sent_txt, sent_info
            )

            aer_list = [(x[0], x[4]) for x in aer_list_raw]
            new_aer_list = list(set(
                aer_list + ae_spans
            ))

            sent_st = psg_txt.index(sent_txt)

            # ans_list = [x for x in new_aer_list]
            aer_reloc = [(ae_t, ae_st + sent_st) for ae_t, ae_st in new_aer_list]
            ae_list += aer_reloc

        return ae_list

    def create_squad_case(context, question, ans_txt, ans_st):
        if ans_txt == '':
            ans_txt_list = []
            ans_st_list = []
        else:
            ans_txt_list = [ans_txt]
            ans_st_list = [ans_st]

        sq_case = {
            'context': context,
            'question': question,
            'answers': {
                'text': ans_txt_list,
                'answer_start': ans_st_list
            }
        }
        return sq_case

    def ans_item_to_squad(context, ans_item):
        ans_txt, ans_st = ans_item[-2]
        question = ans_item[-1]
        # print(question)
        # sys.exit()
        return create_squad_case(context, question, ans_txt, ans_st)

    def qa_for_qa(new_squad, ques_per_ans, no_ans_rate, with_no_ans=True):
        sq, ans2item_list = new_squad
        context = sq['context']

        has_ans_item = []
        no_ans_item = []

        for ans2item in ans2item_list:
            ae_pred, ans_item_list = ans2item

            for j in range(len(ans_item_list)):
                ans_item_list[j][1][-2] = ae_pred

            if len(ae_pred[0]) == 0:
                no_ans_item += ans_item_list
                continue

            has_ans_item += ans_item_list[-ques_per_ans:]

        num_has_ans = len(has_ans_item)
        num_no_ans = int(num_has_ans * no_ans_rate)

        no_ans_item = no_ans_item[-num_no_ans:]
        if with_no_ans:
            qa_list = [ans_item_to_squad(context, x[1]) for x in has_ans_item + no_ans_item]
        else:
            qa_list = [ans_item_to_squad(context, x[1]) for x in has_ans_item]
        return qa_list

    def qa_for_qg(new_squad, ques_per_ans):
        sq, ans2item_list = new_squad
        context = sq['context']

        ans_item = []
        for ans2item in ans2item_list:
            ae_pred, ans_item_list = ans2item
            if len(ae_pred[0]) == 0:
                continue
            ans_item += ans_item_list[:ques_per_ans]
            # ans_item += ans_item_list[-ques_per_ans:]

        # print(ans_item[0])
        # abort()
        qa_list = [ans_item_to_squad(context, x[1]) for x in ans_item]
        return qa_list

    # ext_model_raw = copy.deepcopy(ext_model)
    model = model.cuda()
    model.train()

    ext_model = ext_model.cuda()
    ext_model.train()

    do_eval_args.dropout_eval = False
    do_eval_args.do_predict = True

    psg_txt = dataset[0]['context']
    sent_st, sent_ed = psg_info_list[psg_id]['psg_offset']

    rgx_sent_list = rgx_data[sent_st: sent_ed]
    ae_list = get_ae_from_aer(psg_txt, rgx_sent_list)


    new_squad_list = []
    train_list_qg = []
    train_list_qa = []
    document = dataset[0]

    for epoch in range(num_epoch):
        train_list_qa = []
        train_list_qg = []
        print(f'Starting epoch {epoch}')

        rgx_gen_list = []

        for case_id, rgx in enumerate(rgx_sent_list):
            squad, ae_sent_list = rgx
            sent_st = document['context'].index(squad['context'])

            ae_sent_list = [(x[0], x[4]) for x in ae_sent_list]

            new_squad = search_qa(
                [squad, ae_sent_list],
                tok, model, ext_tok, ext_model,
                case_id, model_args, data_args, do_eval_args
            )
            _, ans2item = new_squad
            for i in range(len(ans2item)):
                ans2item[i][0] = (ans2item[i][0][0], ans2item[i][0][1] + sent_st)
            rgx_gen_list += ans2item

        new_squad = [document, rgx_gen_list]

        train_list_qa += qa_for_qa(new_squad, 1, 0.5, with_no_ans=False)
        train_list_qg += qa_for_qg(new_squad, 1)

        doc_qa_list = copy.deepcopy(train_list_qa)
        for i in range(len(doc_qa_list)):
            del doc_qa_list[i]['context']

        new_squad.append(doc_qa_list)
        new_squad_list.append(new_squad)

        random.shuffle(train_list_qa)
        random.shuffle(train_list_qg)

        print('Generation finished')
        break

        #############################################
        #
        # Tuning the question generation model
        #
        #############################################

        try:
            ctx_set_train, ques_set_train = load_data(
                train_list_qg,
                tok,
                ctx_max_length=512,
                ques_max_length=64,
            )
        except:
            continue

        model = train_ques_gen(model, ctx_set_train, ques_set_train, batch_size=batch_size)

        #############################################
        #
        # Tuning the question generation model
        #
        #############################################

        ext_model, results = run_qa(train_list_qa, ext_tok, ext_model,
                                model_args, data_args, do_train_args)

        #############################################
        #
        # Saving checkpoints
        #
        #############################################

        '''model_to_save = model.module if hasattr(model, 'module') else model
        ext_model_to_save = ext_model.module if hasattr(ext_model, 'module') else ext_model
        model_to_save.save_pretrained(
            f'model_ft_file/ques_gen_sp_{data_args.dataset_name}_{epoch}.pt'
        )
        ext_model_to_save.save_pretrained(
            f'model_ft_file/ext_sp_{data_args.dataset_name}_{epoch}.pt'
        )

        print('Epoch {} finished'.format(epoch))
        print('Checkpoint saved')
        print('-' * 89)
        print('')'''

    if eval_with_questions:
        final_eval_args = copy.deepcopy(do_eval_args)
        data_args.version_2_with_negative = False

        ext_model.eval()
        _, ans_pred = run_qa(dataset, ext_tok, ext_model,
                            model_args, data_args, final_eval_args)
        ans_pred = ans_pred['0']
    else:
        ans_pred = None

    return ans_pred, train_list_qa, train_list_qg, new_squad_list


if __name__ == '__main__':
    num_epoch = 1
    batch_size = 16

    model_args, data_args, training_args = load_args()
    do_train_args = copy.deepcopy(training_args)
    do_eval_args = copy.deepcopy(training_args)

    do_train_args.do_train = True
    do_train_args.do_eval = False
    do_eval_args.do_train = False
    do_eval_args.do_eval = True

    tok, model, ext_tok, ext_model = load_model(
        'model_file/ques_gen_squad.pt',
        'model_file/ext_sqv2.pt'
    )
    
    try:
        nlp_pipeline = stanza.Pipeline(
            'en',
            processors='tokenize,pos,lemma,depparse',
            download_method=DownloadMethod.REUSE_RESOURCES
        )
    except:
        stanza.download('en', processors='tokenize,pos,lemma,depparse')
        nlp_pipeline = stanza.Pipeline(
            'en',
            processors='tokenize,pos,lemma,depparse',
            download_method=DownloadMethod.REUSE_RESOURCES
        )

    print(f'Self-playing on the {data_args.dataset_name} dataset.')

    dataset = json.load(open(
        f'data/{data_args.dataset_name}/doc_data_{data_args.data_split}.json'
    )) #[:1]

    split_id = data_args.data_split
    squad_new_complete = []

    # psg_info_list = json.load(open(
    #     f'splits/{data_args.dataset_name}/psg_info_list_{split_id}.json'
    # ))
    # aer_data = json.load(open(
    #     f'splits/{data_args.dataset_name}/aer_ext_deptree_{split_id}.json'
    # ))
    # rgx_data = json.load(open(
    #     f'splits_ft/{data_args.dataset_name}/squad_rgx_{split_id}_dpt_149.json'
    # ))

    squad_aer_list, psg_info_list = aer(dataset, nlp_pipeline, tok, verbose=False)

    ans_pred_list = []
    qa_train_corpus = []
    qg_train_corpus = []
    new_squad_list = []

    for case_id, squad in enumerate(dataset):

        ans_pred, train_list_qa, train_list_qg, new_squad = play_case(
                [squad], psg_info_list, squad_aer_list,
                tok, copy.deepcopy(model),
                ext_tok, copy.deepcopy(ext_model), split_id, num_epoch, batch_size,
                case_id, model_args, data_args, do_train_args, do_eval_args, nlp_pipeline
        )

        ans_pred_list.append(ans_pred)
        qa_train_corpus += train_list_qa
        qg_train_corpus += train_list_qg
        new_squad_list += new_squad
        # break

    random.shuffle(qa_train_corpus)
    random.shuffle(qg_train_corpus)

    json.dump(
        ans_pred_list,
        open(f'data_gen/{data_args.dataset_name}/ans_pred_sp_{data_args.data_split}.json', 'w')
    )

    json.dump(qa_train_corpus, open(
        f'data_gen/{data_args.dataset_name}/qa_train_corpus_{data_args.data_split}.json', 'w'
    ))

    json.dump(qg_train_corpus, open(
        f'data_gen/{data_args.dataset_name}/qg_train_corpus_{data_args.data_split}.json', 'w'
    ))

    json.dump(new_squad_list, open(
        f'data_gen/{data_args.dataset_name}/rgx_{data_args.data_split}.json', 'w'
    ))

    print(f'\nDomain {data_args.dataset_name}, Split {data_args.dataset_name} processing finished\n')
