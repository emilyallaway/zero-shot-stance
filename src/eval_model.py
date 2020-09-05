import numpy as np
import torch, os, sys, argparse
sys.path.append('./modeling')
import models as bm
import data_utils, model_utils, datasets
import input_models as im
import torch.nn as nn
import torch.optim as optim
import pandas as pd

VECTOR_NAME = 'glove.6B.100d'
SEED = 0
NUM_GPUS = None
use_cuda = torch.cuda.is_available()

def eval(model_handler, dev_data, class_wise=False, is_test=False, correct_preds=False):
    '''
    Evaluates the given model on the given data, by computing
    macro-averaged F1, precision, and recall scores. Can also
    compute class-wise scores. Prints the resulting scores
    :param class_wise: whether to return class-wise scores. Default(False):
                        does not return class-wise scores.
    :return: a dictionary from score names to the score values.
    '''
    model_handler.eval_and_print(data_name='TRAIN', class_wise=class_wise,
                                 correct_preds=correct_preds)

    if is_test:
        dev_name = 'test'
    else:
        dev_name = 'dev'

    model_handler.eval_and_print(data=dev_data, data_name=dev_name,
                                 class_wise=class_wise, correct_preds=correct_preds)


def save_predictions(model_handler, dev_data, out_name, is_test=False, correct_preds=False):
    trn_preds, _, _, _ = model_handler.predict()
    dev_preds, _, _, _ = model_handler.predict(data=dev_data, correct_preds=correct_preds)
    if is_test:
        dev_name = 'test'
    else:
        dev_name = 'dev'

    predict_helper(trn_preds, model_handler.dataloader.data).to_csv(out_name + '-train.csv', index=False)
    print("saved to {}-train.csv".format(out_name))
    predict_helper(dev_preds, dev_data.data).to_csv(out_name + '-{}.csv'.format(dev_name), index=False)
    print("saved to {}-{}.csv".format(out_name, dev_name))


def predict_helper(pred_lst, pred_data):
    out_data = []
    cols = list(pred_data.data_file.columns)
    for i in pred_data.data_file.index:
        row = pred_data.data_file.iloc[i]
        temp = [row[c] for c in cols]
        temp.append(pred_lst[i])
        out_data.append(temp)
    cols += ['pred label']
    return pd.DataFrame(out_data, columns=cols)


if __name__ == '__main__':
    '''
    first arg: config file name
    second arg: data file name
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-i', '--trn_data', help='Name of the training data file', required=False)
    parser.add_argument('-d', '--dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('-k', '--ckp_name', help='Checkpoint name', required=False)
    parser.add_argument('-m', '--mode', help='What to do', required=True)
    parser.add_argument('-n', '--name', help='something to add to the saved model name',
                        required=False, default='')
    parser.add_argument('-o', '--out', help='Ouput file name', default='')
    parser.add_argument('-v', '--score_key', help='What optimized for', required=False, default='f_macro')
    args = vars(parser.parse_args())

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    ####################
    # load config file #
    ####################
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]

    trn_data_kwargs = {}
    dev_data_kwargs = {}

    if 'topic_name' in config:
        topic_vecs = np.load('{}/{}.{}.npy'.format(config['topic_path'],
                                                   config['topic_name'],
                                                   config.get('rep_v', 'centroids')))
        trn_data_kwargs['topic_rep_dict'] = '{}/{}-train.labels.pkl'.format(config['topic_path'],
                                                                            config['topic_name'])

        if 'test' in args['dev_data']:
            dev_s = 'test'
        else:
            dev_s = 'dev'
        dev_data_kwargs['topic_rep_dict'] = '{}/{}-{}.labels.pkl'.format(config['topic_path'],
                                                                          config['topic_name'],
                                                                         dev_s)

    #############
    # LOAD DATA #
    #############
    # load training data

    if 'bert' not in config and 'bert' not in config['name']:
        ################
        # load vectors #
        ################


        vec_name = config['vec_name']
        vec_dim = int(config['vec_dim'])

        vecs = data_utils.load_vectors('../resources/{}.vectors.npy'.format(vec_name),
                                       dim=vec_dim, seed=SEED)
        vocab_name = '../resources/{}.vocab.pkl'.format(vec_name)
        data = datasets.StanceData(args['trn_data'], vocab_name, pad_val=len(vecs) - 1,
                                   max_tok_len=int(config.get('max_tok_len', '200')),
                                   max_sen_len=int(config.get('max_sen_len', '10')),
                                   keep_sen=('keep_sen' in config),
                                   **trn_data_kwargs)
    else:
        data = datasets.StanceData(args['trn_data'], None, max_tok_len=config['max_tok_len'],
                                   max_top_len=config['max_top_len'], is_bert=True,
                                   add_special_tokens=(config.get('together_in', '0') == '0'),
                                   **trn_data_kwargs)

    dataloader = data_utils.DataSampler(data, batch_size=int(config['b']))

    if 'bert' not in config and 'bert' not in config['name']:
        dev_data = datasets.StanceData(args['dev_data'], vocab_name, pad_val=len(vecs) - 1,
                                       max_tok_len=int(config.get('max_tok_len', '200')),
                                       max_sen_len=int(config.get('max_sen_len', '10')),
                                       keep_sen=('keep_sen' in config),
                                       **dev_data_kwargs)
    else:
        dev_data = datasets.StanceData(args['dev_data'], None, max_tok_len=config['max_tok_len'],
                                       max_top_len=config['max_top_len'], is_bert=True,
                                       add_special_tokens=(config.get('together_in', '0') == '0'),
                                       **dev_data_kwargs)

    dev_dataloader = data_utils.DataSampler(dev_data, batch_size=int(config['b']), shuffle=False)

    lr = float(config.get('lr', '0.001'))

    if 'tganet' in config['name']:
        batch_args = {'keep_sen': False}
        input_layer = im.JointBERTLayerWithExtra(vecs=topic_vecs, use_cuda=use_cuda,
                                                 use_both=(config.get('use_ori_topic', '1') == '1'),
                                                 static_vecs=(config.get('static_topics', '1') == '1'))

        setup_fn = data_utils.setup_helper_bert_attffnn

        loss_fn = nn.CrossEntropyLoss()

        model = bm.TGANet(in_dropout_prob=float(config['in_dropout']),
                                 hidden_size=int(config['hidden_size']),
                                 text_dim=int(config['text_dim']),
                                 add_topic=(config.get('add_resid', '0') == '1'),
                                 att_mode=config.get('att_mode', 'text_only'),
                                 topic_dim=int(config['topic_dim']),
                                 learned=(config.get('learned', '0') == '1'),
                                 use_cuda=use_cuda)

        optimizer = optim.Adam(model.parameters())

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': data_utils.prepare_batch,
                  'batching_kwargs': batch_args, 'name': config['name'],  # + args['name'],
                  'loss_function': loss_fn,
                  'optimizer': optimizer,
                  'setup_fn': setup_fn}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda,
                                                      checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                      result_path=config.get('res_path', 'data/gen-stance/'),
                                                      use_score=args['score_key'],
                                                      **kwargs)


    elif 'ffnn-bert' in config['name']:
        if config.get('together_in', '0') == '1':
            batch_args = {'keep_sen': False}
            if 'topic_name' in config:
                input_layer = im.JointBERTLayerWithExtra(vecs=topic_vecs, use_cuda=use_cuda,
                                                         use_both=(config.get('use_ori_topic', '1') == '1'),
                                                         static_vecs=(config.get('static_topics', '1') == '1'))
            else:
                input_layer = im.JointBERTLayer(use_cuda=use_cuda)

        else:
            batch_args = {'keep_sen': False}
            input_layer = im.BERTLayer(mode='text-level', use_cuda=use_cuda)

        setup_fn =data_utils.setup_helper_bert_ffnn

        loss_fn = nn.CrossEntropyLoss()
        model = bm.FFNN(input_dim=input_layer.dim, in_dropout_prob=float(config['in_dropout']),
                        hidden_size=int(config['hidden_size']), bias=False,
                        add_topic=(config.get('add_resid', '1') == '1'), use_cuda=use_cuda)
        optimizer = optim.Adam(model.parameters())

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': data_utils.prepare_batch,
                  'batching_kwargs': batch_args, 'name': config['name'],
                  'loss_function': loss_fn,
                  'optimizer': optimizer,
                  'setup_fn': setup_fn,
                  'fine_tune': (config.get('fine-tune', 'no') == 'yes')}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda,
                                                      checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                      result_path=config.get('res_path', 'data/gen-stance/'),
                                                      use_score=args['score_key'],
                                                      **kwargs)

    elif 'BiCond' in config['name']:
        batch_args = {}
        input_layer = im.BasicWordEmbedLayer(vecs=vecs, use_cuda=use_cuda,
                                             static_embeds=(config.get('tune_embeds', '0') == '0'))

        setup_fn = data_utils.setup_helper_bicond
        loss_fn = nn.CrossEntropyLoss()

        model = bm.BiCondLSTMModel(hidden_dim=int(config['h']), embed_dim=input_layer.dim,
                                   input_dim=(int(config['in_dim']) if 'in_dim' in config['name'] else input_layer.dim),
                                   drop_prob=float(config['dropout']), use_cuda=use_cuda,
                                   num_labels=3, keep_sentences=('keep_sen' in config),
                                   doc_method=config.get('doc_m', 'maxpool'))
        o = optim.Adam(model.parameters(), lr=lr)

        bf = data_utils.prepare_batch

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': bf,
                  'batching_kwargs': batch_args, 'name': config['name'] + args['name'],
                  'loss_function': loss_fn,
                  'optimizer': o,
                  'setup_fn': setup_fn}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda, num_gpus=NUM_GPUS,
                                                      checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                      result_path=config.get('res_path', 'data/gen-stance/'),
                                                      use_score=args['score_key'],
                                                      **kwargs)

    elif 'CTSAN' in config['name']:
        batch_args = {}
        input_layer = im.BasicWordEmbedLayer(vecs=vecs, use_cuda=use_cuda)
        setup_fn = data_utils.setup_helper_bicond

        loss_fn = nn.CrossEntropyLoss()

        bf = data_utils.prepare_batch

        model = bm.CTSAN(hidden_dim=int(config['h']), embed_dim=input_layer.dim, att_dim=int(config['a']),
                         lin_size=int(config['lh']), drop_prob=float(config['dropout']),
                         use_cuda=use_cuda, out_dim=3, keep_sentences=('keep_sen' in config),
                         sentence_version=config.get('sen_v', 'default'),
                         doc_method=config.get('doc_m', 'maxpool'),
                         premade_topic=('topic_name' in config),
                         topic_trans=('topic_name' in config),
                         topic_dim=(int(config.get('topic_dim')) if 'topic_dim' in config else None))

        o = optim.Adam(model.parameters(), lr=lr)

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': bf,
                  'batching_kwargs': batch_args, 'name': config['name'] + args['name'],
                  'loss_function': loss_fn,
                  'optimizer': o,
                  'setup_fn': setup_fn}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda, num_gpus=NUM_GPUS,
                                                      checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                      result_path=config.get('res_path', 'data/gen-stance/'),
                                                      use_score=args['score_key'],
                                                      **kwargs)

    elif 'repffnn' in config['name']:
        batch_args = {'keep_sen': False}
        input_layer = im.JointBERTLayerWithExtra(vecs=topic_vecs, use_cuda=use_cuda,
                                                 use_both=(config.get('use_ori_topic', '1') == '1'),
                                                 static_vecs=(config.get('static_topics', '1') == '1'))

        setup_fn = data_utils.setup_helper_bert_attffnn

        loss_fn = nn.CrossEntropyLoss()

        model = bm.RepFFNN(in_dropout_prob=float(config['in_dropout']),
                           hidden_size=int(config['hidden_size']),
                           input_dim=int(config['topic_dim']),
                           use_cuda=use_cuda)

        optimizer = optim.Adam(model.parameters())

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': data_utils.prepare_batch,
                  'batching_kwargs': batch_args, 'name': config['name'],  # + args['name'],
                  'loss_function': loss_fn,
                  'optimizer': optimizer,
                  'setup_fn': setup_fn}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda,
                                                      checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                      result_path=config.get('res_path', 'data/gen-stance/'),
                                                      use_score=args['score_key'],
                                                      **kwargs)



    cname = '{}ckp-[NAME]-{}.tar'.format(config.get('ckp_path', 'data/checkpoints/'), args['ckp_name'])
    model_handler.load(filename=cname)

    if args['mode'] == 'eval':
        eval(model_handler, dev_dataloader, class_wise=True, is_test=('test' in args['dev_data']))
    elif args['mode'] == 'predict':
        save_predictions(model_handler, dev_dataloader, out_name=args['out'], is_test=('test' in args['dev_data']))
    else:
        print("doing nothing")