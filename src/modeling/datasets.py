import pickle, json
from torch.utils.data import Dataset
import pandas as pd
from functools import reduce
from transformers import BertTokenizer


class StanceData(Dataset):
    '''
    Holds the stance dataset.
    '''
    def __init__(self, data_name, vocab_name, topic_name=None, name='CFpersp-train-full',
                 max_sen_len=10, max_tok_len=200, max_top_len=5, binary=False, keep_sen=False,
                 pad_val=0, truncate_data=None, is_bert=False, add_special_tokens=True,
                 **kwargs):
        self.data_name = data_name
        self.data_file = pd.read_csv(data_name)
        if vocab_name != None:
            self.word2i = pickle.load(open(vocab_name, 'rb'))
        self.name = name
        self.max_sen_len = max_sen_len
        self.max_tok_len = max_tok_len
        self.max_top_len = max_top_len
        self.binary = binary
        self.keep_sen = keep_sen
        self.pad_value = pad_val
        self.is_bert = is_bert
        self.add_special_tokens = add_special_tokens

        if kwargs.get('topic_rep_dict', None) is not None:
            self.topic_rep_dict = pickle.load(open(kwargs['topic_rep_dict'], 'rb'))
        else:
            self.topic_rep_dict = None

        self.preprocess_data()

    def preprocess_data(self):
        print('preprocessing data {} ...'.format(self.data_name))

        self.data_file['text_idx'] = [[] for _ in range(len(self.data_file))]
        self.data_file['topic_idx'] = [[] for _ in range(len(self.data_file))]
        self.data_file['text_topic_idx'] = [[] for _ in range(len(self.data_file))]
        self.data_file['token_type_ids'] = [[] for _ in range(len(self.data_file))]
        if self.keep_sen:
            self.data_file['text_l'] = [[] for _ in range(len(self.data_file))]
            self.data_file['ori_text'] = [[] for _ in range(len(self.data_file))]
        else:
            self.data_file['text_l'] = 0
            self.data_file['ori_text'] = ''
        self.data_file['topic_l'] = 0
        self.data_file['num_sens'] = 0
        self.data_file['text_mask'] = [[] for _ in range(len(self.data_file))]

        if self.is_bert:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            print("processing BERT")
            for i in self.data_file.index:
                row = self.data_file.iloc[i]
                text = json.loads(row['text'])
                num_sens = len(text)
                ori_topic = json.loads(row['topic'])
                ori_text = [' '.join(ti) for ti in text] if self.keep_sen else row['text_s']
                text = self.tokenizer.encode(ori_text, add_special_tokens=self.add_special_tokens,
                                             max_length=int(self.max_tok_len), pad_to_max_length=True)
                topic = self.tokenizer.encode(ori_topic, add_special_tokens=self.add_special_tokens,
                                              max_length=int(self.max_top_len), pad_to_max_length=True)
                self.data_file.at[i, 'text_idx'] = text
                self.data_file.at[i, 'ori_text'] = ori_text
                self.data_file.at[i, 'topic_idx'] = topic
                self.data_file.at[i, 'num_sens'] = num_sens
                if not self.add_special_tokens:
                    self.data_file.at[i, 'text_topic_idx'] = self.tokenizer.build_inputs_with_special_tokens(text, topic)
                    self.data_file.at[i, 'token_type_ids'] = self.tokenizer.create_token_type_ids_from_sequences(text, topic)
            print("...finished pre-processing for BERT")
            return

        for i in self.data_file.index:
            row = self.data_file.iloc[i]

            ori_text = json.loads(row['text'])

            # load topic
            ori_topic = json.loads(row['topic'])

            # index text & topic
            text = [[self.get_index(w) for w in s] for s in ori_text]  # [get_index(w) for w in text]
            topic = [self.get_index(w) for w in ori_topic][:self.max_top_len]

            # truncate text
            if self.keep_sen:
                text = text[:self.max_sen_len]
                text = [t[:self.max_tok_len] for t in text]  # truncates the length of each sentence, by # tokens
                text_lens = [len(t) for t in text]  # compute lens (before padding)
                num_sens = len(text)
                text_mask = [[1] * n for n in text_lens]
            else:
                text = reduce(lambda x, y: x + y, text)
                text = text[:self.max_tok_len]
                text_lens = len(text)  # compute combined text len
                num_sens = 1
                text_mask = [1] * text_lens

            # pad text
            if self.keep_sen:
                for t, tm in zip(text, text_mask):
                    while len(t) < self.max_tok_len:
                        t.append(self.pad_value)
                        tm.append(0)
                while len(text_lens) < self.max_sen_len:
                    text_lens.append(1)
            else:
                while len(text) < self.max_tok_len:
                    text.append(self.pad_value)
                    text_mask.append(0)

            # compute topic len
            topic_lens = len(topic)  # get len (before padding)

            # pad topic
            while len(topic) < self.max_top_len:
                topic.append(self.pad_value)

            if 'text_s' in self.data_file.columns:
                ori_text = [' '.join(ti) for ti in row['text']] if self.keep_sen else row['text_s']
            else:
                ori_text = [' '.join(ti) for ti in row['text']] if self.keep_sen else ' '.join([' '.join(ti) for ti in row['text']])

            if 'topic_str' not in self.data_file.columns:
                self.data_file.at[i, 'topic_str'] = ' '.join(row['topic'])

            self.data_file.at[i, 'text_idx'] = text
            self.data_file.at[i, 'topic_idx'] = topic
            self.data_file.at[i, 'text_l'] = text_lens
            self.data_file.at[i, 'topic_l'] = topic_lens
            self.data_file.at[i, 'ori_text'] = ori_text
            self.data_file.at[i, 'num_sens'] = num_sens
            self.data_file.at[i, 'text_mask'] = text_mask
        print("... finished preprocessing")

    def get_index(self, word):
        return self.word2i[word] if word in self.word2i else len(self.word2i)

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx, corpus=None):
        row = self.data_file.iloc[idx]

        l = int(row['label'])

        sample = {'text': row['text_idx'], 'topic': row['topic_idx'], 'label': l,
                  'txt_l': row['text_l'], 'top_l': row['topic_l'],
                  'ori_topic': row['topic_str'],
                  'ori_text': row['ori_text'],
                  'num_s': row['num_sens'],
                  'text_mask': row['text_mask'],
                  'seen': row['seen?'],
                  'id': row['new_id'],
                  'contains_topic?': row['contains_topic?']} # HERE
        if not self.add_special_tokens:
            sample['text_topic'] = row['text_topic_idx']
            sample['token_type_ids'] = row['token_type_ids']
        if self.topic_rep_dict is not None:
            sample['topic_rep_id'] = self.topic_rep_dict[row['new_id']]
        return sample


class StanceDataBoW(Dataset):
    '''
    Holds and loads stance data sets with vectors instead of word indices and for
    BoWV model. Does NOT actually store the vectors.
    '''
    def __init__(self, data_name, text_vocab_file, topic_vocab_file):
        self.data_file = pd.read_csv(data_name)
        self.text_vocab2i = dict()
        self.topic_vocab2i = dict()
        self.data_name = data_name

        self.unk_index = 0

        self.__load_vocab(self.text_vocab2i, text_vocab_file)
        self.__load_vocab(self.topic_vocab2i, topic_vocab_file)

    def __load_vocab(self, vocab2i, vocab_file):
        f = open(vocab_file, 'r')
        lines = f.readlines()
        i = 1
        print(len(lines))
        for l in lines:
            w = str(l.strip())
            vocab2i[w] = i
            i += 1

    def __len__(self):
        return len(self.data_file)

    def get_vocab_size(self):
        '''
        Gets the size of the vocabulary.
        :return: The vocabulary size
        '''
        return len(self.vocab2i)

    def __convert_text(self, input_data, col):
        '''
        Converts text data to BoW
        :param input_data: tokenized text, as a list
        :return: BoW version of input using the stored vocabulary
        '''
        # make BoW representation
        if col == 'topic':
            vocab2i = self.topic_vocab2i
        else:
            vocab2i = self.text_vocab2i

        text_bow = [0 for _ in range(len(vocab2i) + 1)]
        for w in input_data:
            if w in vocab2i:
                text_bow[vocab2i[w]] = 1
            else:
                text_bow[self.unk_index] = 1
        return text_bow

    def __getitem__(self, idx):
        row = self.data_file.iloc[idx]

        text = reduce(lambda x,y: x + y, json.loads(row['text'])) # collapse the sentences
        topic = json.loads(row['topic'])

        text = self.__convert_text(text, 'text')
        topic = self.__convert_text(topic, 'topic')

        if float(row['label']) == 0:
            l = [1, 0, 0]
        elif float(row['label']) == 1:
            l = [0, 1, 0]
        else:
            l = [0, 0, 1]

        sample = {'text': text, 'topic': topic, 'label': l,
                  'seen': row['seen?'], 'id': row['new_id']}
        return sample
