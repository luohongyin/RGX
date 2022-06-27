import re
import sys
import json
import torch
import string

from copy import deepcopy
from nltk.tokenize import sent_tokenize


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

def overlap(s1, e1, s2, e2):
    if s2 <= s1 and e2 >= s1:
        return True
    if s2 <= e1 and e2 >= e1:
        return True
    return False

def rm_overlap(results):
    span_list = []
    new_results = []
    num_results = len(results)
    for i in range(1, 1 + num_results):
        qg = results[num_results - i]
        st_char = qg[0][1]
        ed_char = st_char + len(qg[0][0]) - 1
        ol = 0
        for sp_st, sp_ed in span_list:
            if overlap(sp_st, sp_ed, st_char, ed_char):
                ol = 1
                break
        if not ol:
            span_list.append([st_char, ed_char])
            new_results.append(qg)
    new_results.reverse()
    return new_results

def most_relevent(context, results):

    stop_words = set([
        'am',
        'is',
        'are',
        'the'
    ])

    ctx_vocab = context.split(' ')
    big_vocab = [' '.join(context[i: i + 2]) for i in range(len(ctx_vocab))]
    tri_vocab = [' '.join(context[i: i + 3]) for i in range(len(ctx_vocab))]

    ctx_vocab = set(ctx_vocab)
    big_vocab = set(big_vocab)
    tri_voacb = set(tri_vocab)

    max_idx = -1
    max_crr = -1
    max_len = -1
    for i, aer in enumerate(results):
        ans_txt = aer[0][0]
        question = aer[1]

        if ans_txt.lower() in question:
            continue

        qwords = question.split(' ')

        crr = 0
        for j, qw in enumerate(qwords):
            if qw in stop_words:
                continue
            if qw in ctx_vocab:
                crr += 1
            if ' '.join(qwords[j: j+2]) in big_vocab:
                crr += 1
            if ' '.join(qwords[j: j+3]) in tri_vocab:
                crr += 1

        if crr > max_crr:
            max_crr = crr
            max_idx = i
            max_len = len(qwords)

        if crr == max_crr and len(qwords) > max_len:
            max_crr = crr
            max_idx = i
            max_len = len(qwords)

    results = results[max_idx: max_idx + 1]
    return results

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

def scale_list(values):
    min_v = min(values)
    scale = max([x - min_v for x in values])
    return scale, min_v

def filter_qa_vqa(ans2item_list, verbose=False):

    results = []

    for i in range(len(ans2item_list)):
        k, v = ans2item_list[i]

        v = [(ae, qg) for ae, qg in v if\
             k[0].lower() not in qg[-1] and\
             len(qg[-1].split(' ')) > 5 and\
             count_ques_word(qg[-1]) < 3 and\
             # (qg[0] > 500) and\
             qg[1] > -0.5 and \
             # qg[3] < 0.1 and\
             ' they ' not in qg[-1] and \
             # ' or ' not in qg[-1]
             # ' you ' not in qg[-1] and\
             'the name of' not in qg[-1]
             ]

        v = sorted(v, key = lambda x: x[1][3], reverse=True)

        # if k[0] == 'L-DOPA':
        #     print(v)
        #     print('---------------')
        #     abort()

        # print(v[0])
        # abort()

        if len(v) == 0:
            continue

        final_scores = torch.Tensor([qg[0] for ae, qg in v])
        min_ans_ppl = v[-1][1][3]

        ques_logp = torch.Tensor([qg[1] for ae, qg in v])
        max_logp = ques_logp.max()

        max_fs, mfs_idx = final_scores.max(0)

        # if max_fs < 100 and min_ans_ppl > 1:
        #     continue

        ques_list = [qg[-1] for ae, qg in v if qg[0] > 500 and qg[3] < 1]

        ques_dict = {}
        for q in ques_list:
            if q in ques_dict:
                ques_dict[q] += 1
            else:
                ques_dict[q] = 1

        ques_list_vote = [q for q in ques_dict.items() if q[1] > 1]

        coe_len = len(k[0].split(' ')) ** 0.5
        # if coe_len == 1:
        #     coe_len = 0.01

        if len(ques_list_vote) > 0:
            ques_list_count = sorted(ques_list_vote, key = lambda x: x[1], reverse=True)
            ques = ques_list_count[0][0]
            results.append([k, post_proc_ques(ques), max_fs.item(), max_logp, min_ans_ppl])
            continue

        v_new = [x for x in v]
        # v_new = [x for x in v if x[1][0] > 5000]
        # if len(v_new) == 0:
        #     v_new = [x for x in v if x[1][0] > 5000]
        ques = v_new[-1][1][-1]

        # if k[0] == 'iTunes Wi-Fi Music Store':
        #     print(v_new)
        #     abort()

        results.append([k, post_proc_ques(ques), max_fs.item(), max_logp, min_ans_ppl])

    '''
    re1 = sorted([x for x in results if x[-2] < 5000],
                 key = lambda x: x[-1], reverse = True)

    re2 = sorted([x for x in results if x[-2] < 15000 and x[-2] > 5000],
                 key = lambda x: x[-1], reverse = True)

    re3 = sorted([x for x in results if x[-2] > 5000],
                 key = lambda x: x[-1], reverse = True)
    '''

    # results = re1 + re2 + re3
    # results = [x for x in results if x[-1] < min_ans_ppl * 100]
    logp_scale, logp_base = scale_list([-x[-2] for x in results])
    loss_scale, loss_base = scale_list([x[-1] for x in results])
    fs_scale, fs_base = scale_list([x[2] for x in results])

    for i, res in enumerate(results):
        logp = res[-2]
        loss = res[-1]
        fs = res[2]
        v = (fs - fs_base) * 0 / fs_scale + \
            (-logp - logp_base) * 1 / logp_scale + \
            (loss - loss_base) * 1 / loss_scale
        results[i].append(v)

    results = sorted(
        results,
        key = lambda x: x[-1],
        reverse=True
    )

    # results = rm_overlap(results)
    # results = filter_overlap(results)

    if verbose:
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

        for x in results:
            print(x)
            print('---------------------\n')

        # abort()

    return results


def filter_qa(ans2item_list, verbose=False):

    results = []

    for i in range(len(ans2item_list)):
        k, v = ans2item_list[i]

        v = [(ae, qg) for ae, qg in v if\
             k[0].lower() not in qg[-1] and\
             len(qg[-1].split(' ')) > 5 and\
             count_ques_word(qg[-1]) < 2 and\
             (qg[0] > 500) and\
             # qg[3] < 0.1 and\
             ' they ' not in qg[-1] and ' or ' not in qg[-1] and ' you ' not in qg[-1] and\
             'the meaning of' not in qg[-1]
             ]

        v = sorted(v, key = lambda x: x[1][3], reverse=True)

        # if k[0] == 'iTunes Wi-Fi Music Store':
        #     print(v)
        #     abort()

        # print(v[0])
        # abort()

        if len(v) == 0:
            continue

        final_scores = torch.Tensor([qg[0] for ae, qg in v])
        min_ans_ppl = v[-1][1][3]

        max_fs, mfs_idx = final_scores.max(0)

        # if max_fs < 100 and min_ans_ppl > 0.01:
        #     continue

        ques_list = [qg[-1] for ae, qg in v if qg[0] > 500 and qg[3] < min_ans_ppl * 10]

        ques_dict = {}
        for q in ques_list:
            if q in ques_dict:
                ques_dict[q] += 1
            else:
                ques_dict[q] = 1

        ques_list_vote = [q for q in ques_dict.items() if q[1] > 1]

        coe_len = len(k[0].split(' ')) ** 0.5
        # if coe_len == 1:
        #     coe_len = 0.01

        if len(ques_list_vote) > 0:
            ques_list_count = sorted(ques_list_vote, key = lambda x: x[1], reverse=True)
            ques = ques_list_count[0][0]
            results.append([k, post_proc_ques(ques), max_fs.item(), min_ans_ppl / coe_len])
            continue

        v_new = [x for x in v if x[1][0] > 5000]
        # if len(v_new) == 0:
        #     v_new = [x for x in v if x[1][0] > 5000]
        ques = v_new[-1][1][-1]

        # if k[0] == 'iTunes Wi-Fi Music Store':
        #     print(v_new)
        #     abort()

        results.append([k, post_proc_ques(ques), max_fs.item(), min_ans_ppl / coe_len])

    re1 = sorted([x for x in results if x[-2] < 5000],
                 key = lambda x: x[-1], reverse = True)

    # re2 = sorted([x for x in results if x[-2] < 15000 and x[-2] > 5000],
    #              key = lambda x: x[-1], reverse = True)

    re2 = sorted([x for x in results if x[-2] > 5000],
                 key = lambda x: x[-1], reverse = True)

    results = re1 + re2 # + re3
    # results = [x for x in results if x[-1] < min_ans_ppl * 100]
    # results = sorted(results, key = lambda x: x[-2], reverse=False)

    results = rm_overlap(results)

    if verbose:
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

        for x in results:
            print(x)
            print('---------------------\n')

        # abort()

    return results

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        text = text.replace('-', ' ')
        text = text.replace('/', ' ')
        text = text.replace('(', ' ')
        text = text.replace(')', ' ')
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def ans_pair_metric(a_pred, a_gt, tokenizer):
    '''
    a_pred: the predicted answer text
    a_gt: the groundtruth answer text
    '''
    # Exclude the special tokens
    a_pred = a_pred.lower()
    a_gt = a_gt.lower()

    pred_ids = normalize_answer(a_pred).split()
    gt_ids = normalize_answer(a_gt).split()

    len_pred = len(pred_ids)
    len_gt = len(gt_ids)
    num_same = 0
    for word_id in pred_ids:
        if word_id in gt_ids:
            num_same += 1.

    em = float(a_pred == a_gt)
    if num_same == 0:
        f1 = 0
    else:
        prec = num_same / len_pred
        recall = num_same / len_gt
        f1 = 2 * prec * recall / (prec + recall)

    return em, f1

def select_qa(ae, qg, tok):
    ae_txt = ae[0]
    ans_pred = qg[-2][0]
    ques_pred = qg[-1]

    if ans_pred in ques_pred:
        return False

    em, f1 = ans_pair_metric(ae_txt, ans_pred, tok)
    if f1 > 0.8:
        return True
    else:
        return False


def get_qa_score(ae_t, qa, tok):
    ae, qg = qa
    ae_recog = ae[0]
    _, f1 = ans_pair_metric(ae_t, ae_recog, tok)
    ques_logp = qg[1]
    ans_loss = qg[3]
    return 0.2 * f1 + 0.2 * ques_logp - ans_loss, f1


def choose(ae, qa_list, tok):
    cur_score = -100000
    cur_qa = None
    ae_t, st_char = ae
    for qa in qa_list:
        score, f1 = get_qa_score(ae_t, qa, tok)
        if score > cur_score:
            cur_score = score
            cur_qa = qa
    return [cur_qa, f1, cur_score]


def choose_middle(ae, qa_list, tok):
    qa_list = sorted(qa_list, key = lambda x: x[1][3], reverse = True)
    mid_idx = len(qa_list) // 2
    cur_qa = qa_list[mid_idx]
    # for x in qa_list:
    #     print(x)
    # print('------------------------------')
    cur_score, f1 = get_qa_score(ae[0], cur_qa, tok)
    return [cur_qa, f1, cur_score]


def choose_lm(ae, qa_list, tok):
    qa_list = sorted(qa_list, key = lambda x: x[1][2], reverse = True)
    cur_score, f1 = get_qa_score(ae[0], qa_list[0], tok)
    return [qa_list[0], f1, cur_score]


def filter_qa_raw(ans2item_list, verbose=False):

    results = []
    tok = ElectraTokenizerFast.from_pretrained('model_file/electra-tokenizer.pt')

    ans2item_list_sorted = sorted(
        ans2item_list,
        key = lambda x: torch.Tensor([
            y[1][1] for y in x[1]
        ]).min(),
        reverse = True
    )

    for i in range(len(ans2item_list)):
        k, v = ans2item_list[i]
        # if max([x[1][0] for x in v]) < 5000:
        #     continue
        if k[0] == '':
            continue
        qa = choose(k, v, tok)
        qa_selected = [k, qa[0][1][-1], qa[0][1][1], qa[0][1][3], qa[1], qa[2]]
        # print(qa_selected)
        # abort()
        results.append(qa_selected)


    results = sorted(results, key = lambda x: x[-1], reverse=False)
    results = [x for x in results if x[-1] > 0]
    # abort()

    if verbose:
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

        for x in results:
            print(x)
            print('---------------------\n')

    return results


def detect_overlap(ae_t, ae_selected):
    for case in ae_selected:
        ae_cand = case[0][0]
        if ae_t != ae_cand and ae_t in ae_cand:
            return True
    return False


def filter_overlap(ae_selected):
    new_ae_selected = [x for x in ae_selected if not detect_overlap(x[0][0], ae_selected)]
    return new_ae_selected


if __name__ == '__main__':
    fn = f'splits_ft/{sys.argv[2]}/squad_aer_reranked_{sys.argv[1]}_play.json'
    fn_out = f'splits_ft/{sys.argv[2]}/new_gen_squad_{sys.argv[1]}.json'

    new_squad = []
    squad_aer = json.load(open(fn, 'r'))
    for i, sq in enumerate(squad_aer):
        # if i != 400:
        #      continue
        # if 'The mechanism of L-DOPA for antinociception was investigated' not in sq[0]['context']:
        #     continue
        ae_selected = filter_qa_raw(sq[1], verbose=True)[:5]
        # for x in ae_selected:
        #     print(x)
        #     print('--------------------------\n')
        print(sq[0]['context'])
        abort()
        # ae_selected = most_relevent(sq[0]['context'], ae_selected)
        # print('##########################')
        # print(ae_selected)
        # abort()
        for aer in ae_selected:
            # if aer[-1] > 1e-4:
            #     continue
            ans_txt, ans_char = aer[0]
            question = aer[1]
            new_sq = deepcopy(sq[0])
            new_sq['answers'] = {}
            new_sq['answers']['text'] = [ans_txt]

            new_sq['answers']['answer_start'] = [ans_char]
            new_sq['question'] = question
            new_sq['ans_ppl'] = aer[-1]
            new_squad.append(new_sq)

    json.dump(new_squad, open(fn_out, 'w'))
    print(f'Finished merging data for split {sys.argv[1]}')
