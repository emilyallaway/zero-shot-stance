import torch
import numpy as np
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt

from IPython import embed

import sys
import pickle

sys.path.append('../modeling')
import input_models as im
import datasets, data_utils

use_cuda = torch.cuda.is_available()
SEED = 4783


def load_data(train_data, id_col='ori_id'):
    df = pd.read_csv(train_data)

    seen_ids = set()
    corpus = []
    for i in df.index:
        row = df.iloc[i]
        if row[id_col] in seen_ids: continue
        corpus.append(row['text_s'])
        seen_ids.add(row[id_col])

    return corpus


def get_features(corpus):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    word2idf = dict()
    for w,i in vectorizer.vocabulary_.items():
        word2idf[w] = vectorizer.idf_[i]
    return word2idf

def combine_word_piece_tokens(word_toks, word2tfidf):
    # join the BERT word-piece tokens
    new_word_toks = []
    for tok_lst in word_toks:
        word2pieces = dict()
        i = 0
        new_tok_lst = []
        while i < len(tok_lst):
            w = tok_lst[i]
            if not w.startswith('##'):
                new_tok_lst.append(w)
            else:
                old_word = new_tok_lst.pop(-1)
                new_w = old_word + w.strip("##")
                new_tok_lst.append(new_w)

                word2pieces[new_w] = [old_word, w]
            i += 1
        new_word_toks.append(new_tok_lst)

        for w, p_lst in word2pieces.items():
            if w not in word2tfidf:
                continue

            all_pieces = [p_lst[1]]
            wp = p_lst[0]
            while wp in word2pieces:
                all_pieces.append(word2pieces[wp][1])
                wp = word2pieces[wp][0]
            all_pieces.append(wp)

            for wp in all_pieces:
                if wp not in word2tfidf:
                    word2tfidf[wp] = word2tfidf[w]

    return new_word_toks


def get_tfidf_weights(new_word_toks, vecs, word2tfidf):
    tfidf_lst = []
    for toklst in new_word_toks:  # word_toks:
        temp = []
        for w in toklst:
            temp.append(word2tfidf.get(w, 0.))
        # temp = [word2tfidf[w] for w in toklst if w in word2tfidf]
        while len(temp) < vecs.shape[1]:  # padding to maxlen
            temp.append(0)
        tfidf_lst.append(temp)
    return tfidf_lst


def save_bert_vectors(embed_model, dataloader, batching_fn, batching_kwargs, word2tfidf, dataname):
    doc_matrix = []
    topic_matrix = []
    doc2i = dict()
    topic2i = dict()
    didx = 0
    tidx = 0
    for sample_batched in dataloader:
        args = batching_fn(sample_batched, **batching_kwargs)
        with torch.no_grad():
            embed_args = embed_model(**args)
            args.update(embed_args)

            vecs = args['txt_E']  # (B, L, 768)
            word_toks = [dataloader.data.tokenizer.convert_ids_to_tokens(args['text'][i],
                                                                         skip_special_tokens=True)
                         for i in range(args['text'].shape[0])]

            # join the BERT word-piece tokens
            new_word_toks = combine_word_piece_tokens(word_toks, word2tfidf)

            tfidf_lst = get_tfidf_weights(new_word_toks, vecs, word2tfidf)

            tfidf_weights = torch.tensor(tfidf_lst, device=('cuda' if use_cuda else 'cpu'))  # (B, L)
            tfidf_weights = tfidf_weights.unsqueeze(2).repeat(1, 1, vecs.shape[2])
            weighted_vecs = torch.einsum('blh,blh->blh', vecs, tfidf_weights)

            avg_vecs = weighted_vecs.sum(1) / args['txt_l'].unsqueeze(1)

            doc_vecs = avg_vecs.detach().cpu().numpy()
            topic_vecs = args['avg_top_E'].detach().cpu().numpy()

        for bi, b in enumerate(sample_batched):
            if b['ori_text'] not in doc2i:
                doc2i[b['ori_text']] = didx
                didx += 1
                doc_matrix.append(doc_vecs[bi])
            if b['ori_topic'] not in topic2i:
                topic2i[b['ori_topic']] = tidx
                tidx += 1
                topic_matrix.append(topic_vecs[bi])
    docm = np.array(doc_matrix)
    np.save('../../resources/topicreps/bert_tfidfW_doc-{}.vecs.npy'.format(dataname), docm)
    del docm
    topicm = np.array(topic_matrix)
    np.save('../../resources/topicreps/bert_topic-{}.vecs.npy'.format(dataname), topicm)
    del topicm
    print("[{}] saved to ../../resources/topicreps/bert_[tfidfW_doc/topic]-{}.vecs.npy".format(dataname, dataname))

    pickle.dump(doc2i, open('../../resources/topicreps/bert_tfidfW_doc-{}.vocab.pkl'.format(dataname), 'wb'))
    pickle.dump(topic2i, open('../../resources/topicreps/bert_topic-{}.vocab.pkl'.format(dataname), 'wb'))
    print("[{}] saved to ../../resources/topicreps/bert_[tfidfW_doc/topic]-{}.vocab.pkl".format(dataname, dataname))


def load_vector_data(p, docname, topicname, dataname, dataloader, mode='concat'):
    docm = np.load('{}{}-{}.vecs.npy'.format(p, docname, dataname))
    topicm = np.load('{}{}-{}.vecs.npy'.format(p, topicname, dataname))
    doc2i = pickle.load(open('{}{}-{}.vocab.pkl'.format(p, docname, dataname), 'rb'))
    topic2i = pickle.load(open('{}{}-{}.vocab.pkl'.format(p, topicname, dataname), 'rb'))

    doc2topics = dict()
    unique_topics = set()

    dataY = dict()
    dataX = []
    idx = -1
    for sample_batched in dataloader:
        for bi, b in enumerate(sample_batched):
            x = b['ori_text']
            t = b['ori_topic']

            if x not in doc2topics:
                doc2topics[x] = set()
            if t not in doc2topics[x]:
                doc2topics[x].add(t)
                docv = docm[doc2i[x]]
                topicv = topicm[topic2i[t]]

                if mode == 'concat':
                    dataX.append(np.concatenate((docv, topicv)))
                elif mode == 'avg':
                    dataX.append(np.mean((docv, topicv), 0))
                else:
                    print("ERROR")
                    sys.exit(1)
                idx += 1
            dataY[b['id']] = idx
    assert len(dataX) - 1 == idx
    return np.array(dataX), dataY


def cluster(dataname, trn_X, trn_Y, dev_X, dev_Y, k, trial_num, link_type='ward', m='euclidean'):
    print("[{}] clustering with: linkage={}, m={}, n_clusters={}...".format(trial_num, link_type, m, k))
    clustering = AgglomerativeClustering(n_clusters=k, linkage=link_type, affinity=m)
    # clustering = KMeans(n_clusters=k)
    clustering.fit(trn_X)
    labels = clustering.labels_
    print('[{}] finished clustering.'.format(trial_num))

    ## labels: new_id -> cluster_number
    trn_id2i = dict()
    for rid, eid in trn_Y.items():
        trn_id2i[rid] = labels[eid]
    trn_oname = '../../resources/topicreps/{}_{}_{}_{}-train.labels.pkl'.format(dataname, link_type, m, k)
    pickle.dump(trn_id2i, open(trn_oname, 'wb'))
    print("[{}] saved to {}".format(trial_num, trn_oname))

    print("[{}] fitting centroid classifier ...".format(trial_num))
    clf  = NearestCentroid()
    clf.fit(trn_X, labels)
    print("[{}] finished fitting classifier.".format(trial_num))
    cen_oname = '../../resources/topicreps/{}_{}_{}_{}.centroids.npy'.format(dataname, link_type, m, k)
    np.save(cen_oname, clf.centroids_)
    print("[{}] saved to {}".format(trial_num, cen_oname))

    dev_labels = clf.predict(dev_X)
    sse = calculate_sse(clf.centroids_, dev_X, dev_labels)
    print("[{}] Sum Squared Error: {}".format(trial_num, sse))

    dev_id2i = dict()
    for rid, eid in dev_Y.items():
        dev_id2i[rid] = dev_labels[eid]
    dev_oname = '../../resources/topicreps/{}_{}_{}_{}-dev.labels.pkl'.format(dataname, link_type, m, k)
    pickle.dump(dev_id2i, open(dev_oname, 'wb'))
    print("[{}] saved to {}".format(trial_num, dev_oname))
    print()
    return sse


def calculate_sse(centroids, dev_X, dev_labels):
    temp = euclidean_distances(dev_X, centroids)
    sse = 0
    for i, l in enumerate(dev_labels):
        sse += temp[i, l]
    return sse


def get_cluster_labels(dataname, k, X, Y, s):
    trn_centroids = np.load('../../resources/topicreps/{}_ward_euclidean_{}.centroids.npy'.format(dataname, k))
    classes = np.array([i for i in range(len(trn_centroids))])

    clf = NearestCentroid()
    clf.centroids_ = trn_centroids
    clf.classes_ = classes

    labels = clf.predict(X)
    id2i = dict()
    for rid, eid in Y.items():
        id2i[rid] = labels[eid]
    oname = '../../resources/topicreps/{}_ward_euclidean_{}-{}.labels.pkl'.format(dataname, k, s)
    pickle.dump(id2i, open(oname, 'wb'))
    print("saved to {}".format(oname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='What to do', required=True)
    parser.add_argument('-i', '--trn_data', help='Name of the training data file', required=False)
    parser.add_argument('-d', '--dev_data', help='Name of the dev data file', required=False)
    parser.add_argument('-e', '--test_data', help='Name of the test data file', required=False,
                        default=None)
    parser.add_argument('-p', '--data_path', required=False, help='Data path to directory for topic reps')
    parser.add_argument('-t', '--topic_name', required=False, default='bert_topic')
    parser.add_argument('-c', '--doc_name', required=False, default='bert_tfidfW_doc')
    parser.add_argument('-k', '--k', required=False)
    parser.add_argument('-v', '--value_range', required=False, help='Range of values for search')
    parser.add_argument('-n', '--n', help='Num neigbors', required=False)
    parser.add_argument('-f', '--file_name', help='Name for files', required=False,
                        default='bert_tfidfW')
    parser.add_argument('-r', '--num_trials', help='Number of trials for search')
    args = vars(parser.parse_args())

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    data = datasets.StanceData(args['trn_data'], None, max_tok_len=200, max_top_len=5, is_bert=True,
                               add_special_tokens=True)
    dataloader = data_utils.DataSampler(data, batch_size=64, shuffle=False)

    dev_data = datasets.StanceData(args['dev_data'], None, max_tok_len=200,
                                   max_top_len=5, is_bert=True, add_special_tokens=True)
    dev_dataloader = data_utils.DataSampler(dev_data, batch_size=64, shuffle=False)

    if args['test_data'] is not None:
        test_data = datasets.StanceData(args['test_data'],None, max_tok_len=200,
                                       max_top_len=5, is_bert=True, add_special_tokens=True)
        test_dataloader = data_utils.DataSampler(test_data, batch_size=64, shuffle=False)

    if args['mode'] == '1':
        print("Saving vectors")

        input_layer = im.BERTLayer(mode='text-level', use_cuda=use_cuda)
        setup_fn = data_utils.setup_helper_bert_ffnn
        batching_fn = data_utils.prepare_batch
        batch_args = {'keep_sen': False}

        corpus = load_data(args['trn_data'])
        word2tfidf = get_features(corpus)

        save_bert_vectors(input_layer, dataloader, batching_fn, batch_args, word2tfidf, 'train')
        save_bert_vectors(input_layer, dev_dataloader, batching_fn, batch_args, word2tfidf, 'dev')

        if args['test_data'] is not None:
            save_bert_vectors(input_layer, test_dataloader, batching_fn, batch_args, word2tfidf, 'test')

    elif args['mode'] == '2':
        print("Clustering")
        trn_X, trn_Y = load_vector_data(args['data_path'], docname=args['doc_name'], topicname=args['topic_name'],
                                        dataname='train', dataloader=dataloader, mode='concat')
        dev_X, dev_Y = load_vector_data(args['data_path'], docname=args['doc_name'], topicname=args['topic_name'],
                                        dataname='dev', dataloader=dev_dataloader, mode='concat')
        if args['k'] is None:
            min_v, max_v = args['value_range'].split('-')

            tried_v = set()
            sse_lst = []
            k_lst = []
            for trial_num in range(int(args['num_trials'])):
                k = np.random.randint(int(min_v), int(max_v) + 1)
                while k in tried_v:
                    k = np.random.randint(int(min_v), int(max_v) + 1)

                sse = cluster(args['file_name'], trn_X, trn_Y, dev_X, dev_Y, k, trial_num)

                sse_lst.append(sse)
                k_lst.append(k)

                tried_v.add(k)
            sort_k_indices = np.argsort(k_lst)
            sorted_k = [k_lst[i] for i in sort_k_indices]
            sorted_sse = [sse_lst[i] for i in sort_k_indices]
            plt.plot(sorted_k, sorted_sse, 'go--')
            plt.savefig('../../resources/topicreps/SSE_clusters_{}.png'.format(args['file_name']))
        else:
            cluster(args['file_name'], trn_X, trn_Y, dev_X, dev_Y, int(args['k']), 0)

    elif args['mode'] == '3':
        print("Getting cluster assignments")
        X, Y = load_vector_data(args['data_path'], docname=args['doc_name'], topicname=args['topic_name'],
                                dataname='train', dataloader=dataloader, mode='concat')
        get_cluster_labels('bert_tfidfW', int(args['k']), X, Y, 'train')

        X, Y = load_vector_data(args['data_path'], docname=args['doc_name'], topicname=args['topic_name'],
                                dataname='dev', dataloader=dev_dataloader, mode='concat')
        get_cluster_labels('bert_tfidfW', int(args['k']), X, Y, 'dev')

        X, Y = load_vector_data(args['data_path'], docname=args['doc_name'], topicname=args['topic_name'],
                                dataname='test', dataloader=test_dataloader, mode='concat')
        get_cluster_labels('bert_tfidfW', int(args['k']), X, Y, 'test')



    else:
        print("doing nothing.")


