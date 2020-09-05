import torch, data_utils, pickle, time, json, copy
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch.nn as nn


class ModelHandler:
    '''
    Class that holds a model and provides the functionality to train it,
    save it, load it, and evaluate it. The model used here is assumed to be an
    sklearn model. Use TorchModelHandler for a model written in pytorch.
    '''
    def __init__(self, model, dataloader, name):
        self.model = model
        self.dataloader = dataloader

        self.name = name

        self.score_dict = dict()

    def prepare_data(self, data=None, type_lst=None):
        '''
        Prepares data to be used for training or dev by formatting it
        correctly.
        :param data: the data to be formatted, in a DataHandler. If this is None (default) then
                        the data used is in self.dataloader.
        :return: the formatted input data as a numpy array,
                the formatted labels as a numpy array
                list of ids
        '''
        if data == None:
            data = self.dataloader

        data = data.get()

        concat_data = []
        labels = []
        id_lst = []
        for s in data:
            if type_lst is not None and s.get('seen', -1) not in type_lst: continue
            concat_data.append(s['text'] + s['topic'])
            labels.append(np.argmax(s['label']))
            id_lst.append(s['id'])

        input_data = np.array(concat_data)
        input_labels = np.array(labels)

        return input_data, input_labels, id_lst


    def train_step(self):
        print("   preparing data")
        input_data, input_labels, _ = self.prepare_data()

        print("   training")
        self.model.fit(input_data, input_labels)

    def save(self, out_prefix):
        pickle.dump(self.model, open('{}{}.pkl'.format(out_prefix, self.name), 'wb'))
        print("model saved to {}".format('{}{}.pkl'.format(out_prefix, self.name)))

    def load(self, name_prefix):
        self.model = pickle.load(open('{}{}.pkl'.format(name_prefix, self.name), 'rb'))
        print("model loaded from {}".format('{}{}.pkl'.format(name_prefix, self.name)))

    def compute_scores(self, score_fn, true_labels, pred_labels, class_wise, name):
        vals = score_fn(true_labels, pred_labels, labels=[0, 1, 2], average=None)
        self.score_dict['{}_macro'.format(name)] = sum(vals) / 3.

        if class_wise:
            self.score_dict['{}_anti'.format(name)] = vals[0]
            self.score_dict['{}_pro'.format(name)] = vals[1]
            self.score_dict['{}_none'.format(name)] = vals[2]

    def eval_model(self, data, class_wise=False, type_lst=None, pass_ids=False):
        print("   preparing data")
        input_data, true_labels, id_lst = self.prepare_data(data, type_lst=type_lst)
        print("   making predictions")

        if pass_ids:
            pred_labels = self.model.predict(id_lst)
        else:
            pred_labels = self.model.predict(input_data)

        print("   computing scores")
        if type_lst is not None:
            print("type list {}".format(type_lst))
        self.compute_scores(f1_score, true_labels, pred_labels, class_wise, 'f')
        # calculate class-wise and macro-average precision
        self.compute_scores(precision_score, true_labels, pred_labels, class_wise, 'p')
        # calculate class-wise and macro-average recall
        self.compute_scores(recall_score, true_labels, pred_labels, class_wise, 'r')

        return self.score_dict


class TorchModelHandler:
    '''
    Class that holds a model and provides the functionality to train it,
    save it, load it, and evaluate it. The model used here is assumed to be
    written in pytorch.
    '''
    def __init__(self, num_ckps=10, use_score='f_macro', use_cuda=False, use_last_batch=True,
                 num_gpus=None, checkpoint_path='data/checkpoints/',
                 result_path='data/', **params):
        super(TorchModelHandler, self).__init__()
        # data fields
        self.model = params['model']
        self.embed_model = params['embed_model']
        self.dataloader = params['dataloader']
        self.batching_fn = params['batching_fn']
        self.batching_kwargs = params['batching_kwargs']
        self.setup_fn = params['setup_fn']
        self.fine_tune=params.get('fine_tune', False)
        self.save_checkpoints=params.get('save_ckp', False)


        self.num_labels = self.model.num_labels
        self.labels = params.get('labels', None)
        self.name = params['name']
        self.use_last_batch = use_last_batch

        # optimization fields
        self.loss_function = params['loss_function']
        self.optimizer = params['optimizer']

        # stats fields
        self.checkpoint_path = checkpoint_path
        self.checkpoint_num = 0
        self.num_ckps = num_ckps
        self.epoch = 0

        self.result_path = result_path

        # evaluation fields
        self.score_dict = dict()
        self.max_score = 0.
        self.max_lst = []  # to keep top 5 scores
        self.score_key = use_score

        # GPU support
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.device = 'cuda' if params.get('device') is None else 'cuda:'+ params['device']
            # move model and loss function to GPU, NOT the embedder
            self.model = self.model.to(self.device)
            self.loss_function = self.loss_function.to(self.device)

        if num_gpus is not None:
            self.model = nn.DataParallel(self.model, device_ids=[0,1])

    def save_best(self, data=None, scores=None, data_name=None, class_wise=False):
        '''
        Evaluates the model on data and then updates the best scores and saves the best model.
        :param data: data to evaluate and update based on. Default (None) will evaluate on the internally
                        saved data. Otherwise, should be a DataSampler. Only used if scores is not None.
        :param scores: a dictionary of precomputed scores. Default (None) will compute a list of scores
                        using the given data, name and class_wise flag.
        :param data_name: the name of the data evaluating and updating on. Only used if scores is not None.
        :param class_wise: lag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores. Only used if scores is not None.
        '''
        if scores is None:
            # evaluate and print
            scores = self.eval_and_print(data=data, data_name=data_name,
                                         class_wise=class_wise)
        scores = copy.deepcopy(scores)  # copy the scores, otherwise storing a pointer which won't track properly

        # update list of top scores
        curr_score = scores[self.score_key]
        score_updated = False
        if len(self.max_lst) < 5:
            score_updated = True
            if len(self.max_lst) > 0:
                prev_max = self.max_lst[-1][0][self.score_key] # last thing in the list
            else:
                prev_max = curr_score
            self.max_lst.append((scores, self.epoch - 1))
        elif curr_score > self.max_lst[0][0][self.score_key]: # if bigger than the smallest score
            score_updated = True
            prev_max = self.max_lst[-1][0][self.score_key] # last thing in the list
            self.max_lst[0] = (scores, self.epoch - 1) #  replace smallest score

        # update best saved model and file with top scores
        if score_updated:
            # sort the scores
            self.max_lst = sorted(self.max_lst, key=lambda p: p[0][self.score_key]) # lowest first
            # write top 5 scores
            f = open('{}{}.top5_{}.txt'.format(self.result_path, self.name, self.score_key), 'w') # overrides
            for p in self.max_lst:
                f.write('Epoch: {}\nScore: {}\nAll Scores: {}\n'.format(p[1], p[0][self.score_key],
                                                                      json.dumps(p[0])))
            # save best model step, if its this one
            print(curr_score, prev_max)
            if curr_score > prev_max:
                if self.save_checkpoints:
                    self.save(num='BEST')

    def save(self, num=None):
        '''
        Saves the pytorch model in a checkpoint file.
        :param num: The number to associate with the checkpoint. By default uses
                    the internally tracked checkpoint number but this can be changed.
        '''
        if num is None:
            check_num = self.checkpoint_num
        else: check_num = num

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
        }, '{}ckp-{}-{}.tar'.format(self.checkpoint_path, self.name, check_num))

        if not self.embed_model.static_embeds:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.embed_model.state_dict(),
            }, '{}ckp-{}-{}.embeddings.tar'.format(self.checkpoint_path, self.name,
                                                   check_num))

        if num is None:
            self.checkpoint_num = (self.checkpoint_num + 1) % self.num_ckps

    def load(self, filename='data/checkpoints/ckp-[NAME]-FINAL.tar', use_cpu=False):#filename='data/checkpoints/ckp-[NAME]-FINAL.tar'):
        '''
        Loads a saved pytorch model from a checkpoint file.
        :param filename: the name of the file to load from. By default uses
                        the final checkpoint for the model of this' name.
        '''
        filename = filename.replace('[NAME]', self.name)
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def train_step(self):
        '''
        Runs one epoch of training on this model.
        '''
        print("[{}] epoch {}".format(self.name, self.epoch))
        self.model.train()
        self.loss = 0.
        start_time = time.time()
        for i_batch, sample_batched in enumerate(self.dataloader):
            self.model.zero_grad()

            y_pred, labels = self.get_pred_with_grad(sample_batched)

            label_tensor = torch.tensor(labels)
            if self.use_cuda:
                label_tensor = label_tensor.to(self.device)

            graph_loss = self.loss_function(y_pred, label_tensor)

            self.loss += graph_loss.item()

            graph_loss.backward()

            self.optimizer.step()

        end_time = time.time()
        print("   took: {:.1f} min".format((end_time - start_time)/60.))
        self.epoch += 1

    def compute_scores(self, score_fn, true_labels, pred_labels, class_wise, name):
        '''
        Computes scores using the given scoring function of the given name. The scores
        are stored in the internal score dictionary.
        :param score_fn: the scoring function to use.
        :param true_labels: the true labels.
        :param pred_labels: the predicted labels.
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        :param name: the name of this score function, to be used in storing the scores.
        '''
        if self.labels is None:
            labels = [i for i in range(self.num_labels)]
        else:
            labels = self.labels
        n = float(len(labels))

        vals = score_fn(true_labels, pred_labels, labels=labels, average=None)
        self.score_dict['{}_macro'.format(name)] = sum(vals) / n
        if class_wise:
            self.score_dict['{}_anti'.format(name)] = vals[0]
            self.score_dict['{}_pro'.format(name)] = vals[1]
            if n > 2:
                self.score_dict['{}_none'.format(name)] = vals[2]

    def eval_model(self, data=None, class_wise=False, correct_preds=False):
        '''
        Evaluates this model on the given data. Stores computed
        scores in the field "score_dict". Currently computes macro-averaged
        F1 scores, precision and recall. Can also compute scores on a class-wise basis.
        :param data: the data to use for evaluation. By default uses the internally stored data
                    (should be a DataSampler if passed as a parameter).
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        :return: a map from score names to values
        '''
        pred_labels, true_labels, t2pred, marks = self.predict(data, correct_preds=correct_preds)
        self.score(pred_labels, true_labels, class_wise, t2pred, marks)

        return self.score_dict

    def predict(self, data=None, correct_preds=False):
        all_y_pred = None
        all_labels = None
        all_marks = None
        all_ct = None

        self.model.eval()
        self.loss = 0.

        if data is None:
            data = self.dataloader

        t2pred = dict()
        for sample_batched in data:

            with torch.no_grad():
                y_pred, labels = self.get_pred_noupdate(sample_batched)

                label_tensor = torch.tensor(labels)
                if self.use_cuda:
                    label_tensor = label_tensor.to(self.device)
                self.loss += self.loss_function(y_pred, label_tensor).item()

                if isinstance(y_pred, dict):
                    y_pred_arr = y_pred['preds'].detach().cpu().numpy()
                else:
                    y_pred_arr = y_pred.detach().cpu().numpy()
                ls = np.array(labels)

                m = [b['seen'] for b in sample_batched]
                if correct_preds:
                    ct = [b['contains_topic?'] for b in sample_batched]

                for bi, b in enumerate(sample_batched):
                    t = b['ori_topic']
                    t2pred[t] = t2pred.get(t, ([], []))
                    t2pred[t][0].append(y_pred_arr[bi, :])
                    t2pred[t][1].append(ls[bi])

                if all_y_pred is None:
                    all_y_pred = y_pred_arr
                    all_labels = ls
                    all_marks = m
                    if correct_preds: all_ct = ct
                else:
                    all_y_pred = np.concatenate((all_y_pred, y_pred_arr), 0)
                    all_labels = np.concatenate((all_labels, ls), 0)
                    all_marks = np.concatenate((all_marks, m), 0)
                    if correct_preds: all_ct = np.concatenate((all_ct, ct), 0)

        for t in t2pred:
            t2pred[t] = (np.argmax(t2pred[t][0], axis=1), t2pred[t][1])

        pred_labels = all_y_pred.argmax(axis=1)
        if correct_preds:
            pred_labels = self.retroactive_correct(all_y_pred, pred_labels, all_ct)
        true_labels = all_labels
        return pred_labels, true_labels, t2pred, all_marks

    def eval_and_print(self, data=None, data_name=None, class_wise=False, correct_preds=False):
        '''
        Evaluates this model on the given data. Stores computed
        scores in the field "score_dict". Currently computes macro-averaged.
        Prints the results to the console.
        F1 scores, precision and recall. Can also compute scores on a class-wise basis.
        :param data: the data to use for evaluation. By default uses the internally stored data
                    (should be a DataSampler if passed as a parameter).
        :param data_name: the name of the data evaluating.
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        :return: a map from score names to values
        '''
        scores = self.eval_model(data=data, class_wise=class_wise, correct_preds=correct_preds)
        print("Evaling on \"{}\" data".format(data_name))
        for s_name, s_val in scores.items():
            print("{}: {}".format(s_name, s_val))
        return scores

    def score(self, pred_labels, true_labels, class_wise, t2pred, marks, topic_wise=False):
        '''
        Helper Function to compute scores. Stores updated scores in
        the field "score_dict".
        :param pred_labels: the predicted labels
        :param true_labels: the correct labels
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        '''
        self.compute_scores(f1_score, true_labels, pred_labels, class_wise, 'f')
        self.compute_scores(precision_score, true_labels, pred_labels, class_wise, 'p')
        self.compute_scores(recall_score, true_labels, pred_labels, class_wise, 'r')

        for v in [1, 0]:
            tl_lst = []
            pl_lst = []
            for m, tl, pl in zip(marks, true_labels, pred_labels):
                if m != v: continue
                tl_lst.append(tl)
                pl_lst.append(pl)
            self.compute_scores(f1_score, tl_lst, pl_lst, class_wise, 'f-{}'.format(v))
            self.compute_scores(precision_score, tl_lst, pl_lst, class_wise, 'p-{}'.format(v))
            self.compute_scores(recall_score, tl_lst, pl_lst, class_wise, 'r-{}'.format(v))

        if topic_wise:
            for t in t2pred:
                self.compute_scores(f1_score, t2pred[t][1], t2pred[t][0], class_wise,
                                    '{}-f'.format(t))

    def get_pred_with_grad(self, sample_batched):
        '''
        Helper function for getting predictions while tracking gradients.
        Used for training the model.
        OVERRIDES: super method.
        :param sample_batched: the batch of data samples
        :return: the predictions for the batch (as a tensor) and the true
                    labels for the batch (as a numpy array)
        '''
        args = self.batching_fn(sample_batched, **self.batching_kwargs)

        if not self.fine_tune:
            # EMBEDDING
            embed_args = self.embed_model(**args)
            args.update(embed_args)

            # PREDICTION
            y_pred = self.model(*self.setup_fn(args, self.use_cuda))
        else:
            y_pred = self.model(**args)
        labels = args['labels']

        return y_pred, labels

    def get_pred_noupdate(self, sample_batched):
        '''
        Helper function for getting predictions without tracking gradients.
        Used for evaluating the model or getting predictions for other reasons.
        OVERRIDES: super method.
        :param sample_batched: the batch of data samples
        :return: the predictions for the batch (as a tensor) and the true labels
                    for the batch (as a numpy array)
        '''
        args = self.batching_fn(sample_batched, **self.batching_kwargs)

        with torch.no_grad():
            # EMBEDDING
            if not self.fine_tune:
                embed_args = self.embed_model(**args)
                args.update(embed_args)

                # PREDICTION
                y_pred = self.model(*self.setup_fn(args, self.use_cuda))
            else:
                y_pred = self.model(**args)

            labels = args['labels']

        return y_pred, labels