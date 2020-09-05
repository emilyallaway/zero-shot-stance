import torch
import torch.nn as nn

import model_layers as bml


class BiCondLSTMModel(torch.nn.Module):
    '''
    Bidirectional Coniditional Encoding LSTM (Augenstein et al, 2016, EMNLP)
    Single layer bidirectional LSTM where initial states are from the topic encoding.
    Topic is also with a bidirectional LSTM. Prediction done with a single layer FFNN with
    tanh then softmax, to use cross-entropy loss.
    '''
    def __init__(self, **kwargs):
        super(BiCondLSTMModel, self).__init__()
        self.use_cuda = kwargs['use_cuda']
        self.num_labels = kwargs.get('num_labels', 3)
        self.sentence_level = kwargs.get('keep_sentences', False)
        self.doc_m = kwargs.get('doc_method', 'maxpool')

        self.num_layers = kwargs.get('num_layers', 1)

        self.hidden_dim = kwargs['hidden_dim']
        self.drop_prob = kwargs['drop_prob']

        self.bilstm = bml.BiCondLSTMLayer(hidden_dim=self.hidden_dim, embed_dim=kwargs['embed_dim'],
                                          input_dim=kwargs['input_dim'], drop_prob=self.drop_prob,
                                          num_layers=self.num_layers, use_cuda=self.use_cuda,
                                          sentence_level=self.sentence_level, doc_method=self.doc_m,
                                          premade_topic=kwargs.get('premade_topic', False))
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.pred_layer = bml.PredictionLayer(input_size=2 * self.num_layers * self.hidden_dim,
                                          output_size=self.num_labels,
                                          pred_fn=nn.Tanh(), use_cuda=self.use_cuda)



    def forward(self, text, topic, text_l, topic_l):
        text = text.transpose(0, 1)  # (T, B, E)
        topic = topic.transpose(0, 1)  # (C,B,E)

        _, combo_fb_hn, _ = self.bilstm(text, topic, topic_l, text_l)  # (B, H*N_dir*N_layer)

        combo_fb_hn = self.dropout(combo_fb_hn) #(B, H*N, dir*N_layers)

        y_pred = self.pred_layer(combo_fb_hn)  # (B, 2)
        return y_pred


class CTSAN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(CTSAN, self).__init__()

        self.num_labels = kwargs.get('out_dim', 3)
        self.hidden_dim = kwargs['hidden_dim']
        self.use_cuda = kwargs['use_cuda']
        self.sentence_level = kwargs.get('keep_sentences', False)
        self.sentence_v = kwargs.get('sentence_version', 'default')
        self.doc_method = kwargs.get('doc_method', 'maxpool')

        self.ctsan_layer = bml.CTSANLayer(hidden_dim=self.hidden_dim, embed_dim=kwargs['embed_dim'],
                                          att_dim=kwargs['att_dim'], drop_prob=kwargs['drop_prob'],
                                          use_cuda=kwargs['use_cuda'],
                                          sentence_level=(self.sentence_v == 'bicondonly'),
                                          premade_topic=kwargs['premade_topic'],
                                          topic_trans=kwargs.get('topic_trans', False),
                                          topic_dim=kwargs.get('topic_dim', None))

        if self.sentence_level and self.sentence_v == 'default' and self.doc_method == 'topicatt':
            self.topic_att = bml.ScaledDotProductAttention(input_dim=2*self.hidden_dim,
                                                           use_cuda=self.use_cuda)

        self.lin = bml.TwoLayerFFNNLayer(input_dim=2 * self.hidden_dim, hidden_dim=kwargs['lin_size'],
                                     out_dim=self.num_labels, nonlinear_fn=nn.Tanh())

    def forward(self, text, topic, text_l, topic_l):
        if not self.sentence_level or (self.sentence_level and self.sentence_v == 'bicondonly'):
            _, att_vec, _ = self.ctsan_layer(text, topic, text_l, topic_l)
        else:
            # text: (B, S, T, E), topic: (B, C, E), text_l: (B, S)
            text = text.transpose(0, 1) #(S, B, T, E)
            text_l = text_l.transpose(0, 1) #(S, B)
            a_lst = []
            for si in range(text.shape[0]):
                _, avec, topic_rep = self.ctsan_layer(text[si], topic, text_l[si], topic_l)
                a_lst.append(avec)
            vec_lst = torch.stack(a_lst) #(S, B, 2H)

            if self.doc_method == 'topicatt':
                ## topic-attention
                # topic_rep: (2, B, H)
                topic_q = topic_rep.transpose(0, 1).reshape(-1, 2 * self.hidden_dim) #(B, 2H)
                vec_lst = vec_lst.transpose(0, 1) #(B, S, 2H)
                att_vec = self.topic_att(vec_lst, topic_q)
            else:
                ## max-pooling
                att_vec = vec_lst.max(0)[0] #(B, 2H)

        preds = self.lin(att_vec)
        return preds


class FFNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(FFNN, self).__init__()
        self.use_cuda = kwargs['use_cuda']
        self.num_labels = kwargs.get('num_labels', 3)
        self.use_topic = kwargs['add_topic']

        if 'input_dim' in kwargs:
            if self.use_topic:
                in_dim = 2 * kwargs['input_dim']
            else:
                in_dim = kwargs['input_dim']
        else:
            in_dim = kwargs['topic_dim'] + kwargs['text_dim']

        self.model = nn.Sequential(nn.Dropout(p=kwargs['in_dropout_prob']),
                                   nn.Linear(in_dim, kwargs['hidden_size']),
                                   kwargs.get('nonlinear_fn',nn.Tanh()),
                                   nn.Linear(kwargs['hidden_size'], 3,
                                             bias=kwargs.get('bias', True)))


    def forward(self, text, topic):
        if self.use_topic:
            combined_input = torch.cat((text, topic), 1)
        else:
            combined_input = text
        y_pred = self.model(combined_input)
        return y_pred


class TGANet(torch.nn.Module):
    ## Topic-grouped Attention Network
    def __init__(self, **kwargs):
        super(TGANet, self).__init__()
        self.use_cuda = kwargs['use_cuda']
        self.hidden_dim = kwargs['hidden_size']
        self.input_dim = kwargs['text_dim']
        self.num_labels = 3

        self.attention_mode = kwargs.get('att_mode', 'text_only')

        self.learned = kwargs['learned']

        self.att = bml.ScaledDotProductAttention(self.input_dim)

        self.topic_trans = torch.empty((kwargs['topic_dim'], self.input_dim),
                                       device=('cuda' if self.use_cuda else 'cpu'))
        self.topic_trans = nn.Parameter(nn.init.xavier_normal_(self.topic_trans))

        self.ffnn = FFNN(use_cuda=kwargs['use_cuda'], add_topic=True,
                         input_dim=self.input_dim,
                         in_dropout_prob=kwargs['in_dropout_prob'],
                         hidden_size=self.hidden_dim)

    def forward(self, text, topic, topic_rep, text_l):
        avg_text = text.sum(1) / text_l.unsqueeze(1)

        if topic_rep.shape[1] != topic.shape[2]:
            topic_in = torch.mm(topic_rep, self.topic_trans)  # (B, I)
        else:
            topic_in = topic_rep
        # topic_in = topic_rep
        gen_rep = self.att(topic, topic_in)
        preds = self.ffnn(avg_text, gen_rep)

        return preds


class RepFFNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(RepFFNN, self).__init__()
        self.use_cuda = kwargs['use_cuda']
        self.num_labels = kwargs.get('num_labels', 3)

        self.hidden_size = kwargs['hidden_size']
        self.model = nn.Sequential(nn.Dropout(p=kwargs['in_dropout_prob']),
                                   nn.Linear(kwargs['input_dim'], self.hidden_size),
                                   kwargs.get('nonlinear_fn', nn.Tanh()),
                                   nn.Linear(self.hidden_size, self.num_labels))

    def forward(self, text, topic, topic_rep, text_l):
        y_pred = self.model(topic_rep)
        return y_pred
