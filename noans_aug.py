import re
import sys
import json
import copy
import time
import random
from nltk.tokenize import word_tokenize

def span_tokenize(ctx):
    ctx_words = word_tokenize(ctx)
    ctx_sub = ctx
    ctx_offsets = []
    cur_subst = 0

    for i, word in enumerate(ctx_words):
        st = cur_subst + ctx_sub.index(word)
        ed = st + len(word)
        ctx_offsets.append((st, ed))
        ctx_sub = ctx[ed:]
        cur_subst = ed
    return ctx_words, ctx_offsets

def all_index(ctx, entity):
    return [x.start() for x in re.finditer(entity, ctx)]

def find_closest(ent_st_list, ans_st):
    if ans_st is None:
        return ent_st_list
    gap = 100000
    cur_st = -1
    for idx in ent_st_list:
        if abs(idx - ans_st) < gap:
            gap = abs(idx - ans_st)
            cur_st = idx
    return cur_st

def common_entities(ctx, ctx_words, question, ans_info, stop_words, entity_list):
    entities = {}
    ques_words, ques_offset = span_tokenize(question)
    ques_len = len(ques_words)

    for i, word in enumerate(ques_words):
        if word not in ctx_words or word in stop_words:
            continue
        j = 0

        while question[ques_offset[i][0]: ques_offset[i+j][1]] in ctx_words:
            if ques_words[i + j] in stop_words:
                j += 1
                if i + j == ques_len:
                    break
                continue
            span = question[ques_offset[i][0]: ques_offset[i+j][1]]
            entities[span] = ctx_words[span]
            j += 1
            if i+j == ques_len:
                break

    if ans_info is not None:
        ans_txt, ans_st = ans_info
    else:
        ans_txt = None
        ans_st = None

    valid_entities = []
    for k, v in entities.items():
        valid = True
        for k_cand, v_cand in entities.items():
            if k_cand == k:
                continue
            elif k in k_cand and len(v) == len(v_cand):
                valid = False
                break
        if valid:
            valid_entities.append((k, find_closest(ctx_words[k], ans_st)))

    valid_entities += [x for x in entity_list if x[0].lower() in question]
    # valid_entities = sorted(valid_entities, key = lambda x: len(x[0]), reverse=True)
    if ans_info is not None and ans_txt != '':
        valid_entities.append((ans_txt.lower(), ans_st))
    valid_entities = sorted(valid_entities, key = lambda x: len(x[0]), reverse=True)

    '''
    print('\n')
    print(question)
    print(ques_words)
    print(ans_info)
    print(entities)
    print(valid_entities)
    abort()
    '''
    return valid_entities

def word_tokenize(sent):
    return sent[:-1].split(' ')

def ques_sim(src_words, tgt_words):
    # src_words = set(word_tokenize(src_ques.lower()))
    common_words = src_words & tgt_words
    if len(common_words) == 0:
        return 0
    else:
        f1 = 2. / (len(src_words) / len(common_words) + len(tgt_words) / len(common_words))
        return f1


def ques_equal(src_ques, tgt_ques):
    if src_ques.lower() == tgt_ques.lower():
        return 1.
    else:
        return 0.


def match_ques(target_ques, cand_ques_list):
    similarities = [ques_sim(target_ques[-1], x[-1]) for x in cand_ques_list]
    target_ques = ''
    target_sim = -1
    for i, sim in enumerate(similarities):
        if sim > target_sim:
            target_sim = sim
            target_ques = cand_ques_list[i]#[:2]
    return target_ques, target_sim


def match_ques_all(target_ques, cand_ques_list, border_dict):
    similarities = [ques_sim(target_ques[-1], x[-1]) for x in cand_ques_list]
    category = ['confuse', 'sim_no_ans', 'sim_has_ans']
    result_dict = {}

    for c in category:
        result_dict[c] = {'ques': None, 'sim': -1}

    for i, sim in enumerate(similarities):
        # print(target_ques[-1])
        # print(cand_ques_list[i][-2])
        if cand_ques_list[i][-2] == target_ques[-2]:
            if cand_ques_list[i][-3][0] == '':
                continue
            else:
                cate = 'confuse'
        else:
            if cand_ques_list[i][-2] in border_dict:
                continue
            elif cand_ques_list[i][-3][0] == '':
                cate = 'sim_no_ans'
            else:
                cate = 'sim_has_ans'

        if sim > result_dict[cate]['sim']:
            result_dict[cate]['sim'] = sim
            result_dict[cate]['ques'] = cand_ques_list[i]#[:2]
        # print(result_dict)
        # abort()
    return result_dict


def new_squad(sq, question, has_ans=False):
    ans, question = question
    if has_ans:
        sq['answers'] = {'text': [ans[0]], 'answer_start': [ans[1]]}
    else:
        sq['answers'] = {'text': [], 'answer_start': []}
    sq['question'] = question
    return sq


def sample_ques(ques_list, n, mode):
    if mode == 'first':
        ques_set = set([])
        new_ques_list = []
        for q in ques_list:
            if q.lower() not in ques_set:
                new_ques_list.append(q)
                if len(new_ques_list) == n:
                    break
                ques_set.add(q.lower())
    return new_ques_list


def build_questions_noans(squad_aer):
    sq, ans_list = squad_aer
    no_ans_ques = [x[1] for x in ans_list[0][1]]

    no_ans_dict = {}
    for qg in no_ans_ques:
        no_ans_dict[qg[-1]] = qg
    no_ans_ques = [v for k, v in no_ans_dict.items()]

    has_ans_ques = []
    border_dict = build_border_dict(squad_aer)

    for ans_item in ans_list:
        has_ans_ques += [x[1] for x in ans_item[1]]

    # has_ans_ques = [x for x in has_ans_ques if x[-2][0].lower() not in x[-1].lower()]
    for i in range(len(has_ans_ques)):
        ques = has_ans_ques[i][-1].lower()
        tgt_words = set(word_tokenize(ques))
        has_ans_ques[i].append(tgt_words)

    rel_ques = [match_ques_all(x, has_ans_ques, border_dict) for x in no_ans_ques]
    new_noans_ques = []
    new_hasans_ques = []
    confuse_ques = []
    # '''
    for i in range(len(no_ans_ques)):
        if rel_ques[i]['confuse']['ques'] is None:
            continue
        if rel_ques[i]['confuse']['ques'][-2][-1] != '?':
            continue
        if rel_ques[i]['confuse']['ques'][-3][0] in rel_ques[i]['confuse']['ques'][-2]:
            continue
        if rel_ques[i]['sim_no_ans']['sim'] < 0.5:
            continue

        new_noans_ques.append(rel_ques[i]['sim_no_ans']['ques'][-3: -1])
        new_hasans_ques.append(rel_ques[i]['sim_has_ans']['ques'][-3: -1])
        confuse_ques.append(rel_ques[i]['confuse']['ques'][-3: -1])

    noans_sq_list = [new_squad(copy.deepcopy(sq), x) for x in new_noans_ques]
    hasans_sq_list = [new_squad(copy.deepcopy(sq), x, True) for x in new_hasans_ques]
    confuse_sq_list = [new_squad(copy.deepcopy(sq), x, True) for x in confuse_ques]
    return noans_sq_list, hasans_sq_list, confuse_sq_list


def contained_question(ans_item):
    ae, qg = ans_item
    if ae[0] in qg[-1]:
        return True
    if ae[0] in ae[1]:
        return True
    if ae[1] in ae[0]:
        return True
    if qg[-2][0] in qg[-1] and qg[-2][0] != '':
        return True
    if qg[-1][-1] != '?':
        return True
    return False


def top_noans_questions(squad_aer, n):
    sq, ans_list = squad_aer
    for k, v in ans_list:
        if k[0] == '':
            return [
                new_squad(copy.deepcopy(sq), x[1][-2:], False) for x in v if not contained_question(x)
            ][:n]


def border_questions(squad_aer):
    sq, ans_list = squad_aer
    qa_list = []
    for k, v in ans_list:
        qa_list += [x for x in v if not contained_question(x)]
    qa_list = sorted(qa_list, key = lambda x: x[-1][2], reverse=False)[:10]
    qa_list = [new_squad(copy.deepcopy(sq), x[1][-2:], x[1][-2][0] != '') for x in qa_list]
    # for ae, qg in qa_list:
    #     print(ae)
    #     print(qg)
    #     print('------------------')
    # print(sq['context'])
    return qa_list


def build_border_dict(squad_aer):
    sq, ans_list = squad_aer
    qg_dict = {}
    for ans, qg_list in ans_list:
        ans_t = ans[0]
        for ae, qg in qg_list:
            ques = qg[-1]
            if ae[0].lower() in ques.lower():
                continue
            if ques not in qg_dict:
                qg_dict[ques] = [ans_t]
            else:
                qg_dict[ques].append(ans_t)

    border_dict = {}
    for k, v in qg_dict.items():
        v_set = set(v)
        if len(v_set) < 2:
            continue
        # print(k)
        # print(v_set)
        # print('-' * 89)
        border_dict[k] = v_set

    # abort()
    return border_dict


if __name__ == '__main__':
    domain = sys.argv[1]
    qa_gen = json.load(open(f'data/{domain}/squad_aer_reranked_ft.json'))#[-1:]
    print(f'Processing {len(qa_gen)} passages')
    no_ans_squad = []
    has_ans_squad = []
    confuse_squad = []

    t1 = time.time()

    for i, squad_aer in enumerate(qa_gen):
        if i % 100 == 0:
            t2 = time.time()
            time_used = t2 - t1
            t1 = t2
            print(f'Processed {i} passages, using {time_used} seconds')
        no_ans, has_ans, confuse = build_questions_noans(squad_aer)
        # no_ans_squad += no_ans
        has_ans_squad += has_ans
        confuse_squad += confuse

    squad_train = json.load(open(f'data/{domain}/squad_train.json'))
    new_squad_train = squad_train + no_ans_squad + has_ans_squad
    random.shuffle(new_squad_train)

    print(f'Length of raw training set = {len(squad_train)}')
    print(f'Length of NoANS Aug set = {len(no_ans_squad)}')

    json.dump(new_squad_train, open(f'data/{domain}/squad_noans_aug.json', 'w'))
    json.dump(confuse_squad, open(f'data/{domain}/squad_confuse.json', 'w'))
