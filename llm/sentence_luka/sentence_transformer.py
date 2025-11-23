
import torch
import os, json
from model import LukeModel, LukeConfig
import sentence_transformers
import common_resource
from torch import nn
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Union, Tuple


class Transformer(nn.Module):
    def __init__(self, model, tokenizer, max_seq_length: Optional[int] = None,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False):
        super(Transformer, self).__init__()
        self.config = model.config
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case
        self.auto_model = model
        self.tokenizer = tokenizer

        #No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length


    def __repr__(self):
        return "Transformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.auto_model.__class__.__name__)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        テキストをトークン化する
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # 文字の調整
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output


    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    def load(input_path: str):
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return Transformer(model_name_or_path=input_path, **config)



def make_sentence_transformer():
    dir_current = os.path.dirname(__file__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(dir_current + '/' + common_resource.dir_luke_made + '/config.json', 'r') as f:
        config = json.load(f)
        config = LukeConfig.from_dict(config)
    model = LukeModel(config).to(device)
    model.load_state_dict(torch.load(dir_current + '/' + common_resource.dir_luke_made + '/model.pth'))

    tokenizer = AutoTokenizer.from_pretrained(dir_current + '/' + common_resource.dir_luke_pretrained, do_lower_case=False
                                              , truncation=True, padding=True, max_length=config.max_position_embeddings)
    model = Transformer(model, tokenizer, max_seq_length=config.max_position_embeddings)
    pooling = sentence_transformers.models.Pooling(model.config.hidden_size, pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
    return sentence_transformers.SentenceTransformer(modules=[model, pooling])

if __name__ == '__main__':
    model = make_sentence_transformer()
    sentence_query = '一番良い方法を調査する'
    sentence_target = 'ベストな方法を調査する'
    sentences_input = [sentence_query, sentence_target]
    embeddings = model.encode(sentences_input)
    score_similarity = sentence_transformers.util.pytorch_cos_sim(embeddings[0]
                                                                  , embeddings[1])
    print(sentence_target, score_similarity)
    sentence_target = '明日の準備を行う'
    sentences_input = [sentence_query, sentence_target]
    embeddings = model.encode(sentences_input)
    score_similarity = sentence_transformers.util.pytorch_cos_sim(embeddings[0]
                                                                  , embeddings[1])
    print(sentence_target, score_similarity)
    sentence_target = 'お腹が空いたのでラーメンを食べに行く'
    sentences_input = [sentence_query, sentence_target]
    embeddings = model.encode(sentences_input)
    score_similarity = sentence_transformers.util.pytorch_cos_sim(embeddings[0]
                                                                  , embeddings[1])
    print(sentence_target, score_similarity)
