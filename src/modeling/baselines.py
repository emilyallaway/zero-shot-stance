import sys, os, time, argparse
sys.path.append('..')
import data_utils, model_utils, datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

REP_PATH = '../../resources/'
DATA_PATH= '../../data/VAST/'

def load_setup(model_type, trn_name, dev_name):
    print("Loading data")
    trn_data = datasets.StanceDataBoW(DATA_PATH + trn_name,
                                      text_vocab_file=REP_PATH + 'text_vocab_top10000.txt',
                                      topic_vocab_file=REP_PATH + 'topic_vocab.txt')

    trn_datasampler = data_utils.DataSampler(trn_data, batch_size=len(trn_data))

    dev_data = datasets.StanceDataBoW(DATA_PATH + dev_name,
                                      text_vocab_file=REP_PATH + 'text_vocab_top10000.txt',
                                      topic_vocab_file=REP_PATH + 'topic_vocab.txt')

    dev_datasampler = data_utils.DataSampler(dev_data, batch_size=len(dev_data))

    print("Initializing model")
    #########
    # MODEL #
    #########
    if model_type == 'bowv':
        model = LogisticRegression(solver='lbfgs', class_weight='balanced',
                                   multi_class='multinomial',
                                   max_iter=600)
    elif model_type == 'cmaj':
        model = MajorityClusterBaseline(trn_data)

    return model, trn_datasampler, dev_datasampler


def baseline_BoWV(model_type, trn_name, dev_name):
    '''
    Loads, trains, and evaluates a logistics regression model
    on the training and dev data. Currently does BINARY classification.
    Prints the scores to the console. Saves the trained model
    :param trn_name: the name of the training data file
    :param dev_name: the name of the dev data file
    '''
    model, trn_datasampler, dev_datasampler = load_setup(model_type, trn_name, dev_name)

    model_handler = model_utils.ModelHandler(model=model, name=model_type,
                                             dataloader=trn_datasampler)

    print("Training model")
    st = time.time()
    model_handler.train_step()
    et = time.time()
    print("   took: {:.1f} minutes".format((et - st) / 60.))

    print("Evaluating model on train data")
    eval_helper(model_handler, trn_datasampler, model_type, is_train=True)

    print("Evaluating model on dev data")
    eval_helper(model_handler, dev_datasampler, model_type)

    print("Saving model")
    model_handler.save('../../checkpoints/')


def eval_only(model_type, trn_name, dev_name):
    model, trn_datasampler, dev_datasampler = load_setup(model_type, trn_name, dev_name)
    model_handler = model_utils.ModelHandler(model=model, name=model_type,
                                             dataloader=trn_datasampler)
    if model_type == 'bowv':
        print("Loading model")
        st = time.time()
        model_handler.load('../../checkpoints/')
        et = time.time()
        print("   took: {:.1f} minutes".format((et - st) / 60.))

    print("Evaluating model on data")
    eval_helper(model_handler, dev_datasampler, model_type)


def eval_helper(model_handler,datasampler, model_type, is_train=False):
    print("Evaluating model on dev data")

    if model_type != 'cmaj':
        dev_scores = model_handler.eval_model(data=datasampler, class_wise=True)
    else:
        dev_scores = model_handler.eval_model(data=datasampler, class_wise=True, pass_ids=True)
    for s in dev_scores:
        print('{}: {}'.format(s, dev_scores[s]))

    if not is_train:
        if model_type != 'cmaj':
            dev_scores_unseen = model_handler.eval_model(data=datasampler, class_wise=True, type_lst=[0])
        else:
            dev_scores_unseen = model_handler.eval_model(data=datasampler, class_wise=True,
                                                         type_lst=[0], pass_ids=True)
        for s in dev_scores_unseen:
            print('{}: {}'.format(s, dev_scores_unseen[s]))

        if model_type != 'cmaj':
            dev_scores_seen = model_handler.eval_model(data=datasampler, class_wise=True, type_lst=[1])
        else:
            dev_scores_seen = model_handler.eval_model(data=datasampler, class_wise=True,
                                                       type_lst=[1], pass_ids=True)
        for s in dev_scores_seen:
            print('{}: {}'.format(s, dev_scores_seen[s]))


class MajorityClusterBaseline():
    def __init__(self, trn_data, topic_name='bert_tfidfW_ward_euclidean_197'):
        self.trn_cluster_labels = pickle.load(open(REP_PATH + 'topicreps/' + topic_name + '-train.labels.pkl', 'rb'))
        dev_cluster_labels = pickle.load(open(REP_PATH + 'topicreps/' + topic_name + '-dev.labels.pkl', 'rb'))
        test_cluster_labels = pickle.load(open(REP_PATH + 'topicreps/' + topic_name + '-test.labels.pkl', 'rb'))
        self.id2cluster_label = dict()
        for i,cid in self.trn_cluster_labels.items(): self.id2cluster_label[i] = cid
        for i, cid in dev_cluster_labels.items(): self.id2cluster_label[i] = cid
        for i, cid in test_cluster_labels.items(): self.id2cluster_label[i] = cid

        self.setup(trn_data)

    def setup(self, data):
        c2labels_lst = dict()
        for i in data.data_file.index:
            row = data.data_file.iloc[i]
            cid = self.trn_cluster_labels[row['new_id']]
            if cid not in c2labels_lst:
                c2labels_lst[cid] = []
            c2labels_lst[cid].append(row['label'])
        self.c2l = dict()
        for cid in c2labels_lst:
            self.c2l[cid] = np.argmax(np.bincount(c2labels_lst[cid]))

    def fit(self, data, labels):
        pass

    def predict(self, id_lst):

        labels = []
        for i in id_lst:
            cid = self.id2cluster_label[i]
            labels.append(self.c2l[cid])
        return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='What to do', required=True)
    parser.add_argument('-i', '--trn_data', help='Training data file name', required=True)
    parser.add_argument('-d', '--dev_data', help='Dev data file name')
    parser.add_argument('-t', '--model_type', help='Type of model to evaluate', required=False)
    args = vars(parser.parse_args())

    if args['mode'] == 'bow':
        print("training {} model".format(args['model_type']))
        baseline_BoWV('bowv', args['trn_data'], args['dev_data'])

    elif args['mode'] == 'eval':
        print("evaluating {} model".format(args['model_type']))
        eval_only(args['model_type'], args['trn_data'], args['dev_data'])

    else:
        print("ERROR: doing nothing")

