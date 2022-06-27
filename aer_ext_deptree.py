import sys
import json

import stanza
from stanza.pipeline.core import DownloadMethod

from transformers import (
    ElectraTokenizerFast
)


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


def sentence_aer(context, sent_info, tok, pred_ans, sent_id, question):
    all_valid_pos = set([
        'NOUN', 'PROPN', 'ADV', 'ADJ', 'VERB'
    ])
    sent_st_char = sent_info.words[0].start_char
    sent_ed_char = sent_info.words[-1].end_char
    sentence = context[sent_st_char: sent_ed_char]
    
    spans = []
    if pred_ans is not None:
        pred_ans_sent = [x for x in pred_ans if x['sent_id'] == sent_id]
        for i in range(len(pred_ans_sent)):
            pred_ans_sent[i]['st_char'] -= sent_st_char
            pred_ans_sent[i]['ed_char'] = pred_ans_sent[i]['st_char'] +\
                                            len(pred_ans_sent[i]['text'])
    else:
        pred_ans_sent = []
    num_eval_ans = len(pred_ans_sent)
    
    for i, word in enumerate(sent_info.words):
        if word.upos in all_valid_pos:
            pred_ans_sent.append({
                'text': word.text,
                'st_char': word.start_char - sent_st_char,
                'ed_char': word.end_char - sent_st_char,
                'sent_id': sent_id
            })
        
        head_idx = word.head
        if head_idx == 0:
            continue
        if i <= head_idx - 1:
            word_st_idx = i
            word_ed_idx = head_idx - 1
        else:
            word_st_idx = head_idx - 1
            word_ed_idx = i
        
        span_st_char = sent_info.words[word_st_idx].start_char
        try:
            span_ed_char = sent_info.words[word_ed_idx].end_char
        except:
            print(len(sent_info.words))
            print(i)
            print(sent_info.words[i].head)
            print(word_ed_idx)
            abort()
        
        span_txt = context[span_st_char: span_ed_char]
        
        span_st_char_sent = span_st_char - sent_st_char
        span_ed_char_sent = span_ed_char - sent_st_char

        pred_ans_sent.append({
            'text': span_txt,
            'st_char': span_st_char_sent,
            'ed_char': span_ed_char_sent,
            'sent_id': sent_id
        })
        
    sent_enc = tok(sentence, return_offsets_mapping=True)
    for pred_ans_item in pred_ans_sent:
        span_txt = pred_ans_item['text']
        span_st_char_sent = pred_ans_item['st_char']
        span_ed_char_sent = pred_ans_item['ed_char']

        st_tok, ed_tok, _, _ = ans_loc_in_sent(
            sent_enc['offset_mapping'], span_txt, span_st_char_sent, tok
        )
        
        span_info = [span_txt, 0, st_tok, ed_tok, span_st_char_sent, span_ed_char_sent, 0]
        
        spans.append(span_info)

    return [{'context': sentence, 'question': question}, spans]


def get_sent_id(pred_ans, doc):
    if pred_ans is None:
        return None
    sent_st_list = [x.words[0].start_char for x in doc.sentences]
    
    for i in range(len(pred_ans)):
        st_char = pred_ans[i]['st_char']
        sent_id = len(sent_st_list) - 1
        
        for j, sent_st in enumerate(sent_st_list):
            if sent_st > st_char:
                sent_id = j - 1
                break
        pred_ans[i]['sent_id'] = sent_id
    
    return pred_ans


def aer(dataset, nlp_pipeline, tok, nbest=None, eval_data=None, verbose=True):
    squad_aer_list = []
    psg_info_list = []
    
    cur_psg_offset = 0
    num_psg = len(dataset)
    
    if verbose:
        print(f'Processing {num_psg} passages')

    nbest_dict = {}
    if nbest is not None:
        for i, sq_eval in enumerate(eval_data):
            eval_psg = sq_eval['context']
            ans_pred_txt = nbest[str(i)][0]['text']
            ans_pred_st = nbest[str(i)][0]['offsets'][0]
            ans_pred = {'text': ans_pred_txt, 'st_char': ans_pred_st}

            if eval_psg in nbest_dict:
                nbest_dict[eval_psg].append(ans_pred)
            else:
                nbest_dict[eval_psg] = [ans_pred]
    
    for i, squad in enumerate(dataset):
        context = squad['context']
        pred_answers = nbest_dict[context] if context in nbest_dict else None
        
        doc = nlp_pipeline(context)
        num_sents = len(doc.sentences)

        pred_answers = get_sent_id(pred_answers, doc)
        
        psg_info = {
            'answers': squad['answers'] if 'answers' in squad else None,
            'question': squad['question'] if 'question' in squad else None,
            'psg_offset': [cur_psg_offset, cur_psg_offset + num_sents]
        }
        psg_info_list.append(psg_info)
        cur_psg_offset += num_sents
        
        sent_aer_list = [
            sentence_aer(
                context,
                x, tok,
                pred_answers, j,
                squad['question']
            ) for j, x in enumerate(doc.sentences)
        ]
        squad_aer_list += sent_aer_list
        
        if verbose and i % 1000 == 0:
            print(f'Processed {i} / {num_psg} passages')
    
    if verbose:
        print('AER Finished')
            
    return squad_aer_list, psg_info_list


if __name__ == '__main__':
    domain = sys.argv[1]
    split = sys.argv[2]
    if len(sys.argv) == 4:
        checkpoint = sys.argv[3]
    else:
        checkpoint = None

    dataset = json.load(open(f'splits/{domain}/merged_data_{split}.json'))
    if checkpoint is not None:
        nbest = json.load(
            open(f'coop_model_file/{checkpoint}/nbest_predictions.json')
        )
        eval_data = json.load(
            open(f'data/{domain}/data_proc.json')
        )
    else:
        nbest = None
        eval_data = None
    
    tok = ElectraTokenizerFast.from_pretrained(
        'model_file/electra-tokenizer.pt'
    )

    nlp_pipeline = stanza.Pipeline(
        'en', download_method=DownloadMethod.REUSE_RESOURCES
    )
    
    squad_aer_list, psg_info_list = aer(dataset, nlp_pipeline, tok, nbest, eval_data)
    
    json.dump(squad_aer_list, open(
        f'splits/{domain}/squad_aer_{split}.json', 'w'
    ))
    
    json.dump(psg_info_list, open(
        f'splits/{domain}/psg_info_list_{split}.json', 'w'
    ))