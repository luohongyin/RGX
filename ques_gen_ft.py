import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    BartTokenizer,
    BartTokenizerFast,
    BartForConditionalGeneration,
    ElectraTokenizerFast,
    ElectraForQuestionAnswering,
    get_linear_schedule_with_warmup
)

from ques_pretrain_data import *

def train_ques_gen(model, ctx_set, ques_set, num_epochs=1, batch_size=8, save=False):

    t_total = int(len(ctx_set) / batch_size * num_epochs)
    warmup_steps = int(t_total / 20)

    # model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-6)

    qgen_sche = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # model = model.cuda()
    # model.train()

    model = nn.DataParallel(model)

    step_id = 0

    for epoch in range(num_epochs):
        for i in range(0, len(ctx_set), batch_size):

            ctx_batch = ctx_set[i: i + batch_size]
            ques_batch = ques_set[i: i + batch_size]

            ctx_ids = torch.Tensor([x[0] for x in ctx_batch]).long().cuda()
            ctx_attn_mask = torch.Tensor([x[1] for x in ctx_batch]).long().cuda()

            ques_ids = torch.Tensor([x[0] for x in ques_batch]).long().cuda()
            ques_attn_mask = torch.Tensor([x[1] for x in ques_batch]).long().cuda()

            ctx_len = ctx_attn_mask.sum(1).max().item()
            ques_len = ques_attn_mask.sum(1).max().item()

            ctx_ids = ctx_ids[:, :ctx_len].contiguous()
            ctx_attn_mask = ctx_attn_mask[:, :ctx_len].contiguous()

            ques_ids = ques_ids[:, :ques_len].contiguous()
            ques_attn_mask = ques_attn_mask[:, :ques_len].contiguous()

            result = model(
                input_ids = ctx_ids,
                attention_mask = ctx_attn_mask,
                labels = ques_ids
            )

            loss = result[0].mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            qgen_sche.step()
            model.zero_grad()

            step_id += 1

            if step_id % 100 == 0:
                print('Training step {}, batch mean loss = {}'.format(step_id, loss))

        ctx_set, ques_set = shuffle_training_batch(ctx_set, ques_set)

        if save:
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(f'model_ft_file/qg_nmsl_{sys.argv[1]}_{sys.argv[2]}.pt')

            print('Epoch {} finished'.format(epoch))
            print('Checkpoint saved')
            print('-' * 89)

    return model.module if hasattr(model, 'module') else model


if __name__ == '__main__':
    num_epochs = 2
    batch_size = 32

    model_config_str = 'facebook/bart-large'

    bart_tokenizer = BartTokenizer.from_pretrained(
        # model_config_str
        'model_file/bart-tokenizer.pt'
    )
    model = BartForConditionalGeneration.from_pretrained(
        # model_config_str,
        'model_file/ques_gen_squad.pt'
    ).cuda()
    model.train()

    ctx_set_train, ques_set_train = load_data(
        f'data/{sys.argv[1]}/qg_train_corpus.json',
        # f'data/{sys.argv[1]}/new_gen_ft_with_evi.json',
        bart_tokenizer,
        ctx_max_length=448,
        ques_max_length=64,
    )

    model = train_ques_gen(model, ctx_set_train, ques_set_train, num_epochs, batch_size, save=True)
