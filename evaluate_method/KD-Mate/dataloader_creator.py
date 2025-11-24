
import torch
from torch.utils import data
from dataclasses import dataclass, field


def mask_tokens(inputs, tokenizer, mlm_probability):
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels, masked_indices.float()


def create_collate_fn(tokenizer, max_length):
    def collate_fn(data):
        text = tokenizer.batch_encode_plus(
            [(example.text_a, example.text_b) if example.text_b is not None else example.text_a for example in data], max_length=max_length, truncation=True,
            pad_to_max_length=True,
        )
        input_ids = text['input_ids']
        #token_type_ids = text['token_type_ids']
        attention_mask = text['attention_mask']
        guids = [example.guid for example in data]
        labels = [example.label for example in data]
        input_ids = torch.tensor(input_ids)
        #token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        input_ids_permuted, labels_permuted, mask_permuted = mask_tokens(input_ids.clone(), tokenizer, 0.3)
        labels = torch.tensor(labels)
        guids = torch.tensor(guids)

        return guids, input_ids, attention_mask, labels, input_ids_permuted, mask_permuted, labels_permuted #, token_type_ids

    return collate_fn

def create_dataloader(train_dataset, train_batch_size, tokenizer):
    train_sampler = data.RandomSampler(train_dataset)
    train_dataloader = data.DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, collate_fn=create_collate_fn(tokenizer, 200))
    return train_dataloader

