# RGX: question-answer generation for documents

This repo contains the software developed for the paper,

Cooperative Self-training for Machine Reading Comprehension, Luo H., Li S.-W., Gao M., Yu S., Glass J., NAACL 2022.

# Dependency
We run this software using the following packages,
- Python 3.8.13
- NLTK 3.7
- Stanza 1.4.0
- PyTorch 1.11.0 + cu113
- Transformers 4.19.2
- Datasets 2.3.2

The pretrained models are available via this [Google Drive Link](https://drive.google.com/drive/folders/1pREUVN9FSL6RwamhkQBDb5_3oED3Qjlq?usp=sharing). Please download the models and move them under the `model_file/` directory.
- `model_file/ext_sqv2.pt`: Pretrained ELECTRA-large question answering model on SQuAD v2.0.
- `model_file/ques_gen_squad.pt`: Pretrained BART-large question generation model on SQuAD v2.0.
- `model_file/electra-tokenize.pt`: Electra-large tokenizer provided by Huggingface.
- `model_file/bart-tokenizer.pt`: BART-large tokenizer provided by Huggingface.

# Quick (?) Start
Generate question-answer pairs on the example SQuAD passages we provide at `data/squad/doc_data_0.json` by running the following command,

```
python rgx_doc.py \
    --dataset_name squad \
    --data_split 0 \
    --output_dir tmp/rgx \
    --version_2_with_negative
```

The generated data will be stored under `data_gen/squad`, including `rgx_0.json` and `qa_train_corpus_0.json`. We provide the `$DATA_SPLIT` option for distributed data generation, for example, with Slurm. If only generating QA pairs with one process, simply use `--data_split 0`.

## Data & File Locations
All data are stored at the `data/` and `data_gen/` directories.
- `data/{$DATASET_NAME}/doc_data_{$DATA_SPLIT}.json`: unlabeled documents of the target dataset.
- `data_gen/{$DATASET_NAME}/rgx_{$DATA_SPLIT}.json`: generated QA data aligned with each document from the corresponding dataset.
- `data_gen/{$DATASET_NAME}/qa_train_corpus_{$DATA_SPLIT}.json`: generated QA training set of the given dataset. The training examples follows the SQuAD data format and are randomly shuffled.

## Data Format
- The format of the input file, `doc_data_{$DATA_SPLIT}.json` is a list of dictionaries as
```
[
    {"context": INPUT_DOC_TXT__0},
    {"context": INPUT_DOC_TXT__1},
    ...,
    {"context": INPUT_DOC_TXT__N},
]
```
- The format of the output file, `qa_train_corpus_{$DATA_SPLIT}.json`, is a list of dictionaries as
```
[
    {
        "context": INPUT_DOC_TXT_0,
        "question": GEN_QUESTION_TXT_0,
        "answers": {
            "text": [ ANSWER_TXT ], # only one answer per question
            "answer_start": [ ANSWER_ST_CHAR ]
            # index of the starting character of the answer txt
        }
    },
    {
        ...
    },
]
```
- The format of the output file, `rgx_{$DATA_SPLIT}.json` is a list of document-QA mappings,
```
[
    [
        $DOCUMENT_i,
        $ANS2ITEM_LIST_i,
        $GEN_QA_LIST_i
    ],
    ...
]
```
`$DOCUMENT_i` has the save format in the input file.  The `$ANS2ITEM_LIST_i` is the meta-data of recognized answers and all generated questions. Note that one answer can have multiple questions, and the questions can be either correct or not. The final output of the model is `$GEN_QA_LIST_i`, which is a list of dictionaries of generated QA pairs based on the input document,
```
[
    {
        "question": GEN_QUESTION_TXT_0,
        "answers": {
            "text": [ ANSWER_TXT ],
            "answer_start": [ ANSWER_ST_CHAR ]
        }
    }
]
```

# QA Generation for Your Documents
- Run the following command, or manually create directories under the `data/` and `data_gen/` directories,
```
bash new_dataset.sh $NEW_DATASET_NAME
```
- Move the input file containing the target documents as `data/$NEW_DATASET_NAME/doc_data_0.json`. The format is described in the previous section.

- Run the following command
```
python rgx_doc.py \
    --dataset_name $NEW_DATASET_NAME \
    --data_split 0 \
    --output_dir tmp/rgx \
    --version_2_with_negative
```

The generated files will be stored at `data_gen/{$NEW_DATASET_NAME}/`.

# Contact and Citation

Please contact the first author, Hongyin Luo (hyluo at mit dot edu) if there is any questions. If our system is applied in your work, please cite our paper

```
@article{luo2021cooperative,
  title={Cooperative self-training machine reading comprehension},
  author={Luo, Hongyin and Li, Shang-Wen and Mingye Gao, and Yu, Seunghak and Glass, James},
  journal={arXiv preprint arXiv:2103.07449},
  year={2021}
}
```