import torch, math
import torch.nn as nn
import torch.nn.utils.rnn as rnn

# B: batch size
# T: max sequence length
# E: word embedding size
# C: conn embeddings size
# H: hidden size
# Y: output size
# N_dir: num directions
# N_layer: num layers
# L_i: length of sequence i
# S: number of sentences

class TwoLayerFFNNLayer(torch.nn.Module):
    '''
    2-layer FFNN with specified nonlinear function
    must be followed with some kind of prediction layer for actual prediction
    '''
    def __init__(self, input_dim, hidden_dim, out_dim, nonlinear_fn):
        super(TwoLayerFFNNLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nonlinear_fn,
                                   nn.Linear(hidden_dim, out_dim))

    def forward(self, input):
        # return self.model(input.type(torch.FloatTensor)) #-> (B, Y)
        return self.model(input)

class PredictionLayer(torch.nn.Module):
    '''
    Predicition layer. linear projection followed by the specified functions
    ex: pass pred_fn=nn.Tanh()
    '''
    def __init__(self, input_size, output_size, pred_fn, use_cuda=False):
        super(PredictionLayer, self).__init__()

        self.use_cuda = use_cuda

        self.input_dim = input_size
        self.output_dim = output_size
        self.pred_fn = pred_fn

        # self.model = nn.Sequential(nn.Linear(self.input_dim, int(self.input_dim / 2)),
        #                            self.pred_fn, nn.Linear(int(self.input_dim / 2), self.output_dim))
        self.model = nn.Sequential(nn.Linear(self.input_dim, self.output_dim, bias=False), nn.Tanh())
        # self.model = nn.Linear(self.input_dim, self.output_dim)

        if self.use_cuda:
            self.model = self.model.to('cuda')#cuda()

    def forward(self, input_data):
        return self.model(input_data)


class BiCondLSTMLayer(torch.nn.Module):
    '''
        Bidirection Conditional Encoding (Augenstein et al. 2016 EMNLP).
        Bidirectional LSTM with initial states from topic encoding.
        Topic encoding is also a bidirectional LSTM.
        '''
    def __init__(self, **kwargs):
        super(BiCondLSTMLayer, self).__init__()

        self.hidden_dim = kwargs['hidden_dim']
        self.embed_dim = kwargs['embed_dim']
        self.num_layers = kwargs['num_layers']
        self.use_cuda =kwargs.get('use_cuda', False)
        self.sentence_level = kwargs.get('sentence_level', False)
        self.doc_method = kwargs.get('doc_method', 'maxpool')
        self.premade_topic = kwargs.get('premade_topic', False)

        self.topic_lstm = nn.LSTM(kwargs['input_dim'], self.hidden_dim, bidirectional=True) #LSTM
        self.text_lstm = nn.LSTM(self.embed_dim, self.hidden_dim, bidirectional=True)


        if self.sentence_level and self.doc_method == 'topicatt':
            self.topic_att = ScaledDotProductAttention(input_dim=2*self.hidden_dim, use_cuda=self.use_cuda)
        # self.W_h = nn.Parameter(nn.init.xavier_normal_(torch.empty((embed_dim, input_dim))))
        # self.W_c = nn.Parameter(nn.init.xavier_normal_(torch.empty((embed_dim, input_dim))))

        self.trans_topic = kwargs.get('topic_trans', False)
        if self.premade_topic and self.trans_topic:
            self.topic_W = nn.Parameter(nn.init.xavier_normal_(torch.empty((kwargs['topic_dim'] + self.embed_dim, self.hidden_dim),
                                                               device=('cuda' if self.use_cuda else 'cpu'))))

    def forward(self, txt_e, top_e, top_l, txt_l):
        ####################
        # txt_e = (Lx, B, E), top_e = (Lt, B, E), top_l=(B), txt_l=(B)
        ########################
        if not self.premade_topic:
            p_top_embeds = rnn.pack_padded_sequence(top_e, top_l, enforce_sorted=False)  # LSTM
            self.topic_lstm.flatten_parameters()
            # feed topic
            _, last_top_hn_cn = self.topic_lstm(p_top_embeds)  # ((2, B, H), (2, B, H)) #LSTM

        else:
            if self.trans_topic:
                #top_e: (B, Et)
                top_e_temp = torch.mm(top_e, self.topic_W) #(B, H)
                top_e_rep = top_e_temp.unsqueeze(0).repeat(2, 1, 1) #(2,  B, H)
                last_top_hn_cn = (top_e_rep, top_e_rep)
            else:
                last_top_hn_cn = (top_e[0], top_e[1]) # rep from AE

        last_top_hn = last_top_hn_cn[0]  # LSTM

        if not self.sentence_level:
            p_text_embeds = rnn.pack_padded_sequence(txt_e, txt_l, enforce_sorted=False) # these are sorted
            self.text_lstm.flatten_parameters()

            # feed text conditioned on topic
            output, (txt_last_hn, _)  = self.text_lstm(p_text_embeds, last_top_hn_cn) # (2, B, H)
            txt_fw_bw_hn = txt_last_hn.transpose(0, 1).reshape((-1, 2 * self.hidden_dim))
            padded_output, _ = rnn.pad_packed_sequence(output, total_length=txt_e.shape[0])
        else:
            #txt_e: (S, B, T, E)
            txt_e = txt_e.transpose(1, 2)  # (S, T, B, E)
            txt_l = txt_l.transpose(0, 1)  # (S, B)
            last_lst = []

            for si in range(txt_e.shape[0]):
                p_text_embeds = rnn.pack_padded_sequence(txt_e[si], txt_l[si], enforce_sorted=False)
                _, (last_hn, _) = self.text_lstm(p_text_embeds, last_top_hn_cn)
                last_lst.append(last_hn.transpose(0, 1).reshape(-1, 2 * self.hidden_dim))

            padded_output = torch.stack(last_lst) #(S, B, H)

            if self.doc_method == 'topicatt':
                topic_q = last_top_hn.transpose(0, 1).reshape(-1, 2*self.hidden_dim) #(B, H)
                out = padded_output.transpose(0, 1) #(B, S, H)
                txt_fw_bw_hn = self.topic_att(out, topic_q)
            else:
                ## max-pooling
                txt_fw_bw_hn = padded_output.max(0)[0] #(B, H)

        return padded_output, txt_fw_bw_hn, last_top_hn


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, input_dim, use_cuda=False):
        super(ScaledDotProductAttention, self).__init__()
        self.input_dim = input_dim

        self.scale = math.sqrt(2 * self.input_dim)

    def forward(self, inputs, query):
        # inputs = (B, L, 2*H), query = (B, 2*H), last_hidden=(B, 2*H)
        sim = torch.einsum('blh,bh->bl', inputs, query) / self.scale  # (B, L)
        att_weights = nn.functional.softmax(sim, dim=1)  # (B, L)
        context_vec = torch.einsum('blh,bl->bh', inputs, att_weights)  # (B, 2*H)
        return context_vec


class CTSANLayer(torch.nn.Module):
    def __init__(self, **kwargs):
        super(CTSANLayer, self).__init__()
        self.sentence_level= kwargs['sentence_level']
        self.premade_topic = kwargs['premade_topic']
        self.hidden_dim = kwargs['hidden_dim']
        self.att_dim = kwargs['att_dim']
        self.use_cuda = kwargs['use_cuda']
        self.drop_prob = kwargs['drop_prob']


        self.bicond = BiCondLSTMLayer(hidden_dim=self.hidden_dim, embed_dim=kwargs['embed_dim'],
                                      input_dim=kwargs['embed_dim'],
                                      drop_prob=self.drop_prob, num_layers=1,
                                      use_cuda=self.use_cuda,
                                      sentence_level=self.sentence_level,
                                      premade_topic=self.premade_topic,
                                      topic_trans=kwargs.get('topic_trans', False),
                                      topic_dim=kwargs.get('topic_dim', None))

        self.device = 'cuda' if self.use_cuda else 'cpu'

        self.W1 = torch.empty((2* self.hidden_dim, self.att_dim), device=self.device)
        self.W1 = nn.Parameter(nn.init.xavier_normal_(self.W1))

        self.w2 = torch.empty((self.att_dim, 1), device=self.device)
        self.w2 = nn.Parameter(nn.init.xavier_normal_(self.w2))

        self.b1 = torch.empty((self.att_dim, 1), device=self.device)
        self.b1 = nn.Parameter(nn.init.xavier_normal_(self.b1)).squeeze(1)

        self.b2 = torch.rand([1],  device=self.device)
        self.b2 = nn.Parameter(self.b2)

        self.dropout = nn.Dropout(p=self.drop_prob)

    def forward(self, text, topic, text_l, topic_l):
        text_trans = text.transpose(0, 1)  # (T, B, E)
        if not self.premade_topic:
            # text: (B, T, E), topic: (B, C, E)
            topic_trans = topic.transpose(0, 1)  # (C,B,E)
        else:
            # topic: (2, 2, B, H)
            topic_trans =  topic

        ### bicond-lstm
        padded_output, _, last_top_hn = self.bicond(text_trans, topic_trans, topic_l, text_l)

        padded_output = self.dropout(padded_output)
        # padded_output: (L, B, 2H), txt_fw_bw_hn: (B, 2H), last_top_hn: (2, B, H)
        output = padded_output.transpose(0, 1) #(B, L, 2H)

        ### self-attnetion
        temp_c = torch.sigmoid(torch.einsum('blh,hd->bld', output, self.W1) + self.b1) #(B, L, D)
        c = torch.einsum('bld,ds->bls', temp_c, self.w2).squeeze(-1) + self.b2 #(B, L)
        a = nn.functional.softmax(c, dim=1)

        att_vec = torch.einsum('blh,bl->bh', output, a) #(B, 2H)

        return output, att_vec, last_top_hn