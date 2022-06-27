import sys
import json

import torch

from transformers import (
    ElectraForQuestionAnswering,
)

def mix(sq_model, tgt_model, mix_rate):
    sq_state_dict = sq_model.state_dict()
    target_state_dict = tgt_model.state_dict()

    for k in sq_state_dict:
        # print(k)
        sq_weight = sq_state_dict[k]
        tg_weight = target_state_dict[k]

        sq_state_dict[k] = tg_weight * mix_rate + sq_weight * (1 - mix_rate)

    sq_model.load_state_dict(sq_state_dict)
    return sq_model

if __name__ == '__main__':
    mix_rate = float(sys.argv[0])
    squad_model_path = sys.argv[1]
    rgx_model_path = sys.argv[2]
    '''sq_model = BartForConditionalGeneration.from_pretrained(
        'model_file/ques_gen_squad.pt'
    )

    target_model = BartForConditionalGeneration.from_pretrained(
        sys.argv[3]
    )'''

    sq_model = ElectraForQuestionAnswering.from_pretrained(
        squad_model_path
    )

    target_model = ElectraForQuestionAnswering.from_pretrained(
        rgx_model_path
    )

    sq_state_dict = sq_model.state_dict()
    target_state_dict = target_model.state_dict()

    for k in sq_state_dict:
        sq_weight = sq_state_dict[k]
        tg_weight = target_state_dict[k]

        sq_state_dict[k] = tg_weight * mix_rate + sq_weight * (1 - mix_rate)

    sq_model.load_state_dict(sq_state_dict)
    sq_model.save_pretrained(
        f'model_ft_file/ext_mixed.pt'
    )
