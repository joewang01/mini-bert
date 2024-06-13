import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *
import math

class BertModelPAL(BertPreTrainedModel):
    """
    The BERT model with PAL returns the final embeddings for each token in a sentence.
    """
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        # Embedding layers.
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Register position_ids (1, len position emb) to buffer because it is a constant.
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # PAL Apparatus
        self.v_e_s = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_dim_pal)
                                          for task in range(config.nb_tasks)])
        self.v_d_s = nn.ModuleList([nn.Linear(config.hidden_dim_pal, config.hidden_size)
                                          for task in range(config.nb_tasks)])

        # BERT encoder.
        self.bert_layers = nn.ModuleList([BertLayerPAL(config, v_e_s=self.v_e_s, v_d_s=self.v_d_s)
                                                                            for _ in range(config.num_hidden_layers)])

        # [CLS] token transformations.
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Get word embedding from self.word_embedding into input_embeds.
        inputs_embeds = self.word_embedding(input_ids)

        # Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)

        # Get token type ids. Since we are not considering token type, this embedding is
        # just a placeholder.
        tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        # Add three embeddings together; then apply embed_layer_norm and dropout and return.
        my_embeddings = inputs_embeds + pos_embeds + tk_type_embeds
        emb_n = self.embed_layer_norm(my_embeddings)
        emb_n_drop = self.embed_dropout(emb_n)

        return emb_n_drop

    def encode(self, hidden_states, attention_mask, task):

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
        for i, layer_module in enumerate(self.bert_layers):
            # Feed the encoding from the last bert_layer to the next.
            hidden_states = layer_module(hidden_states, extended_attention_mask, task)

        return hidden_states

    def forward(self, input_ids, attention_mask, task):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # Get the embedding for each input token.
        embedding_output = self.embed(input_ids=input_ids)

        # Feed to a transformer (a stack of BertLayers).
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask, task=task)

        # Get cls token hidden state.
        if self.config.agg_method == 'cls':
            first_tk = sequence_output[:, 0]
        elif self.config.agg_method == 'mean':
            first_tk = torch.mean(sequence_output, dim=1)
        elif self.config.agg_method == 'max':
            first_tk = sequence_output.max(dim=1).values

        return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}

class BertLayerPAL(nn.Module):
    def __init__(self, config, v_e_s=None, v_d_s=None):
        super().__init__()
        # Multi-head attention.
        self.self_attention = BertSelfAttention(config)
        # Add-norm for multi-head attention.
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Feed forward.
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # Add-norm for feed forward.
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.TS = nn.ModuleList([TS_Attention(config, v_e_s[task], v_d_s[task]) for task in range(config.nb_tasks)])

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        """
        This function is applied after the multi-head attention layer or the feed forward layer.
        input: the input of the previous layer
        output: the output of the previous layer
        dense_layer: used to transform the output
        dropout: the dropout to be applied
        ln_layer: the layer norm to be applied
        """
        # Hint: Remember that BERT applies dropout to the transformed output of each sub-layer,
        # before it is added to the sub-layer input and normalized with a layer norm.
        out_dense = dense_layer(output)
        out_dense_dropped = dropout(out_dense)
        out_add = input + out_dense_dropped
        out = ln_layer(out_add)
        return out

    def forward(self, hidden_states, attention_mask, task):
        """
        hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
        as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
        Each block consists of:
        1. A multi-head attention layer (BertSelfAttention).
        2. An add-norm operation that takes the input and output of the multi-head attention layer.
        3. A feed forward layer.
        4. An add-norm operation that takes the input and output of the feed forward layer.
        """
        MH = self.self_attention(hidden_states, attention_mask) + self.TS[task](hidden_states, attention_mask)
        LN = self.add_norm(hidden_states, MH, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
        SA = self.interm_af(self.interm_dense(LN))
        out = self.add_norm(LN, SA, self.out_dense, self.out_dropout, self.out_layer_norm)
        return out


class TS_Attention(nn.Module):
    # Notation of python variables inspired by Bert and Pals paper
    def __init__(self, config, v_e = None, v_d = None):
        super().__init__()
        self.v_e = v_e
        self.v_d = v_d
        config_att = copy.deepcopy(config)
        config_att.hidden_size = config.hidden_dim_pal
        self.g = BertSelfAttention(config_att)
    def forward(self, hidden_states, attention_mask):
        ts = self.v_d(self.g(self.v_e(hidden_states), attention_mask))
        return ts


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # JPL
        # Mapping with the DFP guidelines
        # config.hidden_size = d
        # config.num_attention_heads = n
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize the linear transformation layers for key, value, query.
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # This dropout is applied to normalized attention scores following the original
        # implementation of transformer. Although it is a bit unusual, we empirically
        # observe that it yields better performance.
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # Next, we need to produce multiple heads for the proj. This is done by spliting the
        # hidden state to self.num_attention_heads, each of size self.attention_head_size.
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        # Each attention is calculated following eq. (1) of https://arxiv.org/pdf/1706.03762.pdf.
        # Attention scores are calculated by multiplying the key and query to obtain
        # a score matrix S of size [bs, num_attention_heads, seq_len, seq_len].
        # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th
        # token, given by i-th attention head.
        # Before normalizing the scores, use the attention mask to mask out the padding token scores.
        # Note that the attention mask distinguishes between non-padding tokens (with a value of 0)
        # and padding tokens (with a value of a large negative number).

        # Make sure to:
        # - Normalize the scores with softmax.
        # - Multiply the attention scores with the value to get back weighted values.
        # - Before returning, concatenate multi-heads to recover the original shape:
        #   [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].

        # Argument to softmax: multiplying query and key and scaling it
        bs, n, seq_len, d_n = key.shape
        # Create the softmax argument for each sequence element t
        # We effect a matrix product on [SeqL X d_n] X [d_n X SeqL]
        # which corresponds to W_i^q h_j dot W_i^k h_t / sqrt(d / n)
        e = torch.matmul(query, key.transpose(-1, -2)) / (math.sqrt(d_n))
        # Apply attention_mask
        e += attention_mask
        # Normalize with softmax along the last dimension
        a = F.softmax(e, dim=-1)  # This keeps same dimensions
        # Multiply attention scores with the value to get back weighted values
        c = torch.matmul(a, value)
        # Concatenate along the head dimension to obtain desired dimensions
        c = c.transpose(1, 2)
        c = c.contiguous().view(bs, seq_len, int(n * d_n))
        return c

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # First, we have to generate the key, value, query for each token for multi-head attention
        # using self.transform (more details inside the function).
        # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        # Calculate the multi-head attention.
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value