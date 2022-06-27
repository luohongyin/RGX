# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

# import logging
import os
import sys
import copy

from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, load_metric, Dataset

datasets.logging.disable_progress_bar()

# datasets.set_progress_bar_enabled(False)
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
    ElectraTokenizerFast,
    ElectraForQuestionAnswering,
    logging
)
from transformers.trainer_utils import is_main_process
from utils_qa import (
    postprocess_qa_predictions,
    new_squad_dataset,
    new_aqa_dataset,
    new_eval_dataset
)

logging.set_verbosity_info()
# logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    ckp_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the ckp to use (via the datasets library)."}
    )
    data_split: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

import copy

def load_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # training_args.should_save = False
    return model_args, data_args, training_args


def new_squad_dataset(new_train):
    titles = ['answers', 'context', 'id', 'question', 'title']

    train_dataset = {x:[] for x in titles}
    for i, data in enumerate(new_train):
        for t in titles:
            if t == 'id' or t == 'title':
                data[t] = str(i)
            train_dataset[t].append(data[t])
    train_dataset = Dataset.from_dict(train_dataset)
    return train_dataset


def paraphrase_ques(question):
    if 'Which' in question:
        question = question.replace('Which', 'What')
    return question

def flip_logits(candidates):
    noans = -100
    hasans = -100
    top_ans = None
    top_ans_offset = [0, 0]
    for cand in candidates:
        if cand['text'] == '' and noans == -100:
            noans = cand['start_logit'] + cand['end_logit']
        elif hasans == -100:
            hasans = cand['start_logit'] + cand['end_logit']
            top_ans = cand['text']
            try:
                top_ans_offset = cand['offsets']
            except:
                print(cand)
                abort()
        if noans > -100 and hasans > -100:
            break
    return abs(noans - hasans), top_ans, top_ans_offset


def run_qa(train_dataset, tokenizer, model, model_args, data_args, training_args):
    set_seed(training_args.seed)

    sq_datasets = {}
    if training_args.do_train:
        sq_datasets['train'] = new_squad_dataset(train_dataset)
    if training_args.do_eval:
        sq_datasets['validation'] = new_squad_dataset(train_dataset)

    if training_args.do_train:
        column_names = sq_datasets["train"].column_names
    else:
        column_names = sq_datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"

    def prepare_train_features(examples):

        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=data_args.max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        '''
        print(tokenized_examples.keys())
        print(len(examples['question']))
        print(tokenized_examples['overflow_to_sample_mapping'])
        print(examples.keys())
        print(len(tokenized_examples['input_ids']))
        # for i in range(10):
        #     print(tokenizer.convert_ids_to_tokens(tokenized_examples['input_ids'][i]))
        abort()
        '''

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    try:
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index + 1)
                    except:
                        print(start_char)
                        print(end_char)
                        print(len(offsets))
                        print(token_end_index)
                        abort()

        return tokenized_examples

    if training_args.do_train:
        train_dataset = sq_datasets["train"].map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Validation preprocessing
    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=data_args.max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if training_args.do_eval:
        # print(datasets.is_progress_bar_enabled())
        # abort()
        if True:
            validation_dataset = sq_datasets["validation"].map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
        # abort()

    # Data collator
    # collator.
    data_collator = default_data_collator if data_args.pad_to_max_length else DataCollatorWithPadding(tokenizer)
    # abort()

    # Post-processing:
    def post_processing_function(examples, features, predictions):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions, nbest = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=None,
            is_world_process_zero=trainer.is_world_process_zero(),
            disable_tqdm = True
        )

        for k in nbest:
            gap, top_ans, top_ans_offset = flip_logits(nbest[k])
            nb_pred = copy.deepcopy(nbest[k])
            nbest[k] = nbest[k][0]
            nbest[k]['gap'] = gap
            nbest[k]['top_ans'] = [top_ans, top_ans_offset[0]]
            nbest[k]['candidates'] = nb_pred

        return nbest

    # training_args.should_save = False

    # TODO: Once the fix lands in a Datasets release, remove the _local here and the squad_v2_local folder.
    current_dir = os.path.sep.join(os.path.join(__file__).split(os.path.sep)[:-1])
    metric = load_metric(os.path.join(current_dir, "squad_v2_local") if data_args.version_2_with_negative else os.path.join(current_dir, "metric/squad.py"))
    # print(metric)
    

    def compute_metrics(p: EvalPrediction):
        return None
        for i in range(len(p.predictions)):
            p.predictions[i]['id'] = str(p.predictions[i]['id'])
        for i in range(len(p.label_ids)):
            p.label_ids[i]['id'] = str(p.label_ids[i]['id'])
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    

    # Initialize our Trainer
    # training_args.disable_tqdm = True
    
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        eval_examples=sq_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        # should_save=False
    )
    
    # print(trainer.args.disable_tqdm)
    # abort()
    

    # Training
    if training_args.do_train:
        train_result = trainer.train()

    # abort()
    # Evaluation
    results = {}
    eval_preds = []
    if training_args.do_eval:
        results, eval_preds = trainer.evaluate()
        
    # print('test')
    # abort()

    return trainer.model, eval_preds


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
