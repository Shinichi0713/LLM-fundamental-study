from transformers import LukeModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
import torch.nn as nn
import torch, math, json
import sub_model
import common_resource


# トークン不足の場合に、位置IDを作成する関数
def create_position_ids_from_input_ids(input_ids, padding_idx):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
    return incremental_indices.long() + padding_idx

class LukeConfig(PretrainedConfig):
    model_type = "luke"
    def __init__(
        self,
        vocab_size=50267,
        entity_vocab_size=500000,
        hidden_size=768,
        entity_emb_size=256,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_entity_aware_attention=True,
        classifier_dropout=None,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.hidden_size = hidden_size
        self.entity_emb_size = entity_emb_size  # entity_embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_entity_aware_attention = use_entity_aware_attention
        self.classifier_dropout = classifier_dropout

class LukePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BaseLukeModelOutput(BaseModelOutput):
    entity_last_hidden_state: torch.FloatTensor = None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# セルフアテンション
class LukeSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.use_entity_aware_attention = config.use_entity_aware_attention

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # entity-aware attentionを使う場合,e2w, w2e, e2e, w2wのクエリを作成
        if self.use_entity_aware_attention:
            self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # マルチヘッドアテンション用に変換
    def transpose_for_scores(self, x):
        # サイズの計算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # マルチヘッドアテンション用に変換
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        word_size = word_hidden_states.size(1)

        if entity_hidden_states is None:
            concat_hidden_states = word_hidden_states
        else:
            concat_hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)

        key_layer = self.transpose_for_scores(self.key(concat_hidden_states))
        value_layer = self.transpose_for_scores(self.value(concat_hidden_states))
        
        # エンティティを使う場合
        if self.use_entity_aware_attention and entity_hidden_states is not None:
            # compute query vectors using word-word (w2w), word-entity (w2e), entity-word (e2w), entity-entity (e2e)
            # query layers
            w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
            w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
            e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
            e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

            # compute w2w, w2e, e2w, and e2e key vectors used with the query vectors computed above
            w2w_key_layer = key_layer[:, :, :word_size, :]
            e2w_key_layer = key_layer[:, :, :word_size, :]
            w2e_key_layer = key_layer[:, :, word_size:, :]
            e2e_key_layer = key_layer[:, :, word_size:, :]

            # compute attention scores based on the dot product between the query and key vectors
            w2w_attention_scores = torch.matmul(w2w_query_layer, w2w_key_layer.transpose(-1, -2))
            w2e_attention_scores = torch.matmul(w2e_query_layer, w2e_key_layer.transpose(-1, -2))
            e2w_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2))
            e2e_attention_scores = torch.matmul(e2e_query_layer, e2e_key_layer.transpose(-1, -2))

            # combine attention scores to create the final attention score matrix
            word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=3)
            entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=3)
            attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2)

        else:
            query_layer = self.transpose_for_scores(self.query(concat_hidden_states))
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in LukeModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # アテンションをvalueに適用
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output_word_hidden_states = context_layer[:, :word_size, :]
        if entity_hidden_states is None:
            output_entity_hidden_states = None
        else:
            output_entity_hidden_states = context_layer[:, word_size:, :]

        if output_attentions:
            outputs = (output_word_hidden_states, output_entity_hidden_states, attention_probs)
        else:
            outputs = (output_word_hidden_states, output_entity_hidden_states)

        return outputs

class BaseLukeModelOutputWithPooling(BaseModelOutputWithPooling):
    entity_last_hidden_state: torch.FloatTensor = None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class LukeSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LukeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 自己注意機構
        self.self = LukeSelfAttention(config)
        # 出力を解釈するレイヤ(FFN+正規化)
        self.output = LukeSelfOutput(config)

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        word_size = word_hidden_states.size(1)
        self_outputs = self.self(
            word_hidden_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        if entity_hidden_states is None:
            concat_self_outputs = self_outputs[0]
            concat_hidden_states = word_hidden_states
        else:
            concat_self_outputs = torch.cat(self_outputs[:2], dim=1)
            concat_hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)
        
        attention_output = self.output(concat_self_outputs, concat_hidden_states)

        word_attention_output = attention_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_attention_output = None
        else:
            entity_attention_output = attention_output[:, word_size:, :]

        # add attentions if we output them
        outputs = (word_attention_output, entity_attention_output) + self_outputs[2:]

        return outputs


# transformerからの出力を整形
# 基本は出力の表現力向上のためのレイヤ
class LukeIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = sub_model.ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class LukeOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# Lukeのレイヤ
class LukeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # アテンション
        self.attention = LukeAttention(config)
        # 中間解釈層
        self.intermediate = LukeIntermediate(config)
        # 出力層
        self.output = LukeOutput(config)

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        word_size = word_hidden_states.size(1)

        self_attention_outputs = self.attention(
            word_hidden_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        if entity_hidden_states is None:
            concat_attention_output = self_attention_outputs[0]
        else:
            concat_attention_output = torch.cat(self_attention_outputs[:2], dim=1)

        outputs = self_attention_outputs[2:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, concat_attention_output
        )
        word_layer_output = layer_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_layer_output = None
        else:
            entity_layer_output = layer_output[:, word_size:, :]

        outputs = (word_layer_output, entity_layer_output) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class LukeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Lukeレイヤの積層
        self.layer = nn.ModuleList([LukeLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_word_hidden_states = () if output_hidden_states else None
        all_entity_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
                all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    word_hidden_states,
                    entity_hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    word_hidden_states,
                    entity_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            word_hidden_states = layer_outputs[0]

            if entity_hidden_states is not None:
                entity_hidden_states = layer_outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
            all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    word_hidden_states,
                    all_word_hidden_states,
                    all_self_attentions,
                    entity_hidden_states,
                    all_entity_hidden_states,
                ]
                if v is not None
            )
        else:
            return BaseLukeModelOutput(
                last_hidden_state=word_hidden_states,
                hidden_states=all_word_hidden_states,
                attentions=all_self_attentions,
                entity_last_hidden_state=entity_hidden_states,
                entity_hidden_states=all_entity_hidden_states,
            )

# Lukeのベース(基底)モデル
class LukeBaseModel(PreTrainedModel):
    def __init__(self, config: LukeConfig):
        super().__init__(config)
        self.config = config
        self.supports_gradient_checkpointing = True
        self._no_split_modules = ["LukeAttention", "LukeEntityEmbeddings"]
    
    # 重みの初期化
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if module.embedding_dim == 1:  # embedding for bias parameters
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LukeEncoder):
            module.gradient_checkpointing = value


class LukeEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

# エンティティの埋め込み
class LukeEntityEmbeddings(nn.Module):
    def __init__(self, config: LukeConfig):
        super().__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, entity_ids: torch.LongTensor, position_ids: torch.LongTensor, token_type_ids: torch.LongTensor = None
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = torch.sum(position_embeddings, dim=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

# Lukeモデル
class LukeModel(LukeBaseModel):
    def __init__(self, config: LukeConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        self.embeddings = LukeEmbeddings(config)
        self.entity_embeddings = LukeEntityEmbeddings(config)
        self.encoder = LukeEncoder(config)

        self.pooler = LukePooler(config) if add_pooling_layer else None

        # 重み初期化(ないと学習収束しなくなる)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        entity_ids: Optional[torch.LongTensor] = None,
        entity_attention_mask: Optional[torch.FloatTensor] = None,
        entity_token_type_ids: Optional[torch.LongTensor] = None,
        entity_position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseLukeModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 入力のサイズを保持
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if entity_ids is not None:
            entity_seq_length = entity_ids.size(1)
            if entity_attention_mask is None:
                entity_attention_mask = torch.ones((batch_size, entity_seq_length), device=device)
            if entity_token_type_ids is None:
                entity_token_type_ids = torch.zeros((batch_size, entity_seq_length), dtype=torch.long, device=device)

        # ヘッドマスク
        # 1.0 = ヘッドの影響あり, 0.0 = ヘッドの影響なし
        # head_mask  [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # エンベディング計算(入力層)
        word_embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # maskアテンション再計算
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, entity_attention_mask)

        # エンティティブロックの計算
        if entity_ids is None:
            entity_embedding_output = None
        else:
            entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_token_type_ids)

        # 埋め込み表現をエンコーダに渡す
        encoder_outputs = self.encoder(
            word_embedding_output,
            entity_embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        # プーリングレイヤーの計算
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        # レイヤーの出力を返す
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        # レイヤーの出力を返す
        return BaseLukeModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            entity_last_hidden_state=encoder_outputs.entity_last_hidden_state,
            entity_hidden_states=encoder_outputs.entity_hidden_states,
        )

    def get_extended_attention_mask(
        self, word_attention_mask: torch.LongTensor, entity_attention_mask: Optional[torch.LongTensor]
    ):
        """
        マスクアテンションをtokenに合わせて拡張
        """
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=-1)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        return extended_attention_mask


# 事前学習のパラメータの置き換え
def replace_network_parameter():
    dir_current = os.path.dirname(__file__)
    # モデルの読み込み
    model_pretrained = SentenceTransformer(dir_current + "/" + common_resource.dir_luke_pretrained)
    model_pretrained = model_pretrained._modules['0'].auto_model.base_model
    # torch.save(model_pretrained, dir_current + "/encoder.pth")
    # 自作モデルの作成
    config = LukeConfig()
    # 埋め込み表現のサイズを揃える
    config.vocab_size = model_pretrained.config.vocab_size
    # トークン数を揃える
    config.max_position_embeddings = model_pretrained.config.max_position_embeddings

    config = model_pretrained.config
    model = LukeModel(config)
    model.load_state_dict(model_pretrained.state_dict())
    torch.save(model.state_dict(), dir_current + "/model.pth")

    with open(dir_current + '/config.json', 'w') as f:
        config = config.to_dict()
        json.dump(config, f, indent=4)

                                    

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    import os
    # replace_network_parameter()
    make_sentence_transformer()
    