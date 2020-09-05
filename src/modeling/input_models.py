import torch
import torch.nn as nn
from transformers import BertModel


# B: batch size
# T: max sequence length
# E: word embedding size
# C: conn embeddings size
# H: hidden size
# Y: output size
# N_dir: num directions
# N_layer: num layers
# L_i: length of sequence i


class BasicWordEmbedLayer(torch.nn.Module):
    def __init__(self, vecs, static_embeds=True, use_cuda=False):
        super(BasicWordEmbedLayer, self).__init__()
        vec_tensor = torch.tensor(vecs)
        self.static_embeds=static_embeds

        self.embeds = nn.Embedding.from_pretrained(vec_tensor, freeze=self.static_embeds)

        self.dim = vecs.shape[1]
        self.vocab_size = float(vecs.shape[0])
        self.use_cuda = use_cuda

    def forward(self, **kwargs):
        embed_args = {'txt_E': self.embeds(kwargs['text']).type(torch.FloatTensor), # (B, T, E)
                      'top_E': self.embeds(kwargs['topic']).type(torch.FloatTensor)} #(B, C, E)
        return embed_args


class BERTLayer(torch.nn.Module):
    def __init__(self, mode='text-level', use_cuda=False, device=None):
        super(BERTLayer, self).__init__()

        self.mode = mode
        self.use_cuda = use_cuda
        self.static_embeds = True
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.dim = 768
        if self.use_cuda:
            self.device = 'cuda' if device is None else 'cuda:' + device
            self.bert_layer = self.bert_layer.to('cuda')

    #Takes in a list of tokenized strings (b_item_ids). Creates masks, feeds tokens + masks to BERT. Returns CLS, SEP and averaged embedding for each item in b_item_ids.
    def apply_bert_layer(self, b_item_ids):                                 #b_item_ids - [[tokens for string1],[tokens for string2],...], len(b_items) = B
        item_ids = b_item_ids.type(torch.LongTensor)                        # (B, max_len)
        item_masks = (item_ids != 0)
        item_masks = item_masks.type(torch.LongTensor)                      # (B, max_len)

        if self.use_cuda:
            item_ids = item_ids.to('cuda')
            item_masks = item_masks.to('cuda')

        outputs = self.bert_layer(item_ids, attention_mask=item_masks)
        last_hidden_state = outputs[0]                                              #Sequence of hidden states at last layer of model - (B, max_len, hidden_size). Includes CLS, SEP and PAD token embeddings
        sep_indices_inverse = (item_ids != 102)
        item_masks = item_masks * sep_indices_inverse                               #Making sep index 0 in item mask
        cls_masks = item_masks[:, 0]                                                #Will be 1 if element has CLS token, 0 otherwise. The only case where an element in batch does not have CLS is in the sentences added for padding.
        cls_masks = torch.unsqueeze(cls_masks, 1).repeat(1, self.dim)
        item_masks = item_masks[:, 1:]                                              #Removing CLS mask from item_masks. item_masks is now 1 only for tokens apart from CLS, SEP and PAD.
        token_length = item_masks.sum(dim=1)                                        # (B). No of items that are not CLS, SEP or PAD. An element in this tensor will be 0 only in case of padding sentences.
        item_masks = torch.unsqueeze(item_masks, 2).repeat(1, 1, self.dim)          #Making item masks of dimension - (B, max_len, 768)
        cls_item_embeddings = last_hidden_state[:, 0, :]                            #Extracting embeddings of CLS token for every batch element
        cls_item_embeddings = cls_item_embeddings * cls_masks                       #Makes CLS embedding 0 for the sentences which are added as padding. (B,768)
        last_hidden_state = last_hidden_state[:, 1:, :]                             #last_hidden_state now contains all embeddings except CLS token embeddings for all elements in batch
        token_embeddings = last_hidden_state * item_masks                           #token_embeddings contains embeddings of all tokens. For SEP and PAD tokens - embeddings are made to be all 0s. (B, max_len - 1, hidden_size)
        sum_item_embeddings = (token_embeddings).sum(dim=1)
        denom = item_masks.sum(dim=1)
        denom[denom==0] = 1                                                         #Accounts for sentences which are added as padding as the sum of item_masks (denom) will be 0s for these sentences.
        averaged_item_embeddings = sum_item_embeddings / denom                      # (B, 768)

        return cls_item_embeddings, averaged_item_embeddings, token_embeddings, token_length

    def forward(self, **kwargs):
        b_text_ids = kwargs['text']         #[[[sent1],[sent1],...] ...] if mode is sentence-level. [[text1],[text2]...] if mode is text-level. Is a tensor.
        b_topic_ids = kwargs['topic']       #[[topic1],[topic2],...]. Is a tensor.
        # Use BERT to get text embedding (Truncating text to 200 tokens)
        #Here token refers to tokens in text formed due to BERT's tokenization. text_length is the number of tokens in each text example in batch.
        cls_text_embeddings, averaged_text_embeddings, tokenlevel_text_embeddings, text_length = self.apply_bert_layer(b_text_ids)              #(B,768), (B,768), (B,max_top_len,768), (B)

        #Use BERT to get topic embedding
        cls_topic_embeddings, averaged_topic_embeddings, tokenlevel_topic_embeddings, topic_length = self.apply_bert_layer(b_topic_ids)             #(B,768), (B,768), (B,max_top_len,768), (B)

        embed_args = {'avg_txt_E': averaged_text_embeddings, 'avg_top_E': averaged_topic_embeddings, 'txt_E': tokenlevel_text_embeddings,'top_E': tokenlevel_topic_embeddings, 'txt_l': text_length, 'top_l': topic_length}
        return embed_args


class JointBERTLayer(torch.nn.Module):
    def __init__(self, use_cuda=False):
        super(JointBERTLayer, self).__init__()

        self.use_cuda = use_cuda
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.dim = 768
        self.static_embeds=True
        if self.use_cuda:
            self.bert_layer = self.bert_layer.to('cuda')

    def forward(self, **kwargs):
        text_topic = kwargs['text_topic_batch']
        token_type_ids = kwargs['token_type_ids']

        item_ids = text_topic.type(torch.LongTensor)
        item_masks = (item_ids != 0).type(torch.LongTensor)

        if self.use_cuda:
            item_ids = item_ids.to('cuda')
            item_masks = item_masks.to('cuda')
            token_type_ids = token_type_ids.to('cuda')

        last_hidden, _ = self.bert_layer(input_ids=item_ids,attention_mask=item_masks,
                                         token_type_ids=token_type_ids)
        full_masks = item_masks.unsqueeze(2).repeat(1, 1, self.dim)
        masked_last_hidden = torch.einsum('blh,blh->blh', last_hidden, full_masks)

        max_tok_len = token_type_ids.sum(1)[0].item()
        text_no_cls_sep = masked_last_hidden[:, 1:-max_tok_len - 1, :]
        topic_no_sep = masked_last_hidden[:, -max_tok_len:, :]

        txt_l = (kwargs['text'] != 0).sum(1)
        topic_l = (kwargs['topic'] != 0).sum(1)

        if self.use_cuda:
            txt_l = txt_l.to('cuda')
            topic_l = topic_l.to('cuda')

        avg_txt = text_no_cls_sep.sum(1) / txt_l.unsqueeze(1)
        avg_top = topic_no_sep.sum(1) / topic_l.unsqueeze(1)

        embed_args = {'avg_txt_E': avg_txt, 'avg_top_E': avg_top,
                      'txt_E': text_no_cls_sep, 'top_E': topic_no_sep,
                      'txt_l': txt_l,'top_l': topic_l}
        return embed_args


class JointBERTLayerWithExtra(torch.nn.Module):
    def __init__(self,vecs, use_both=True, static_vecs=True, use_cuda=False):
        super(JointBERTLayerWithExtra, self).__init__()

        self.use_cuda = use_cuda
        self.static_embeds = static_vecs
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        topic_tensor = torch.tensor(vecs)
        self.topic_embeds = nn.Embedding.from_pretrained(topic_tensor, freeze=static_vecs)

        self.use_both = use_both
        if self.use_both:
            self.dim = 768 + topic_tensor.shape[1]
        else:
            self.dim = 768

        if self.use_cuda:
            self.bert_layer = self.bert_layer.to('cuda')

    def forward(self, **kwargs):
        text_topic = kwargs['text_topic_batch']
        token_type_ids = kwargs['token_type_ids']

        item_ids = text_topic.type(torch.LongTensor)
        item_masks = (item_ids != 0).type(torch.LongTensor)

        if self.use_cuda:
            item_ids = item_ids.to('cuda')
            item_masks = item_masks.to('cuda')
            token_type_ids = token_type_ids.to('cuda')

        last_hidden, _ = self.bert_layer(input_ids=item_ids,attention_mask=item_masks,
                                         token_type_ids=token_type_ids)
        full_masks = item_masks.unsqueeze(2).repeat(1, 1, last_hidden.shape[2])
        masked_last_hidden = torch.einsum('blh,blh->blh', last_hidden, full_masks)

        max_tok_len = token_type_ids.sum(1)[0].item()
        text_no_cls_sep = masked_last_hidden[:, 1:-max_tok_len - 1, :]
        topic_no_sep = masked_last_hidden[:, -max_tok_len:, :]

        txt_l = (kwargs['text'] != 0).sum(1)
        topic_l = (kwargs['topic'] != 0).sum(1)

        top_v = self.topic_embeds(kwargs['topic_rep_ids']).type(torch.FloatTensor)  # (B, L, E)

        if self.use_cuda:
            txt_l = txt_l.to('cuda')
            topic_l = topic_l.to('cuda')
            top_v = top_v.to('cuda')

        avg_txt = text_no_cls_sep.sum(1) / txt_l.unsqueeze(1)
        avg_top = topic_no_sep.sum(1) / topic_l.unsqueeze(1)

        if self.use_both:
            top_in = torch.cat([avg_top, top_v], dim=1)
        else:
            top_in = top_v

        embed_args = {'avg_txt_E': avg_txt, 'avg_top_E': top_in,
                      'txt_E': text_no_cls_sep, 'top_E': topic_no_sep,
                      'txt_l': txt_l,'top_l': topic_l,
                      'ori_avg_top_E': avg_top}

        return embed_args