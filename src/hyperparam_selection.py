import numpy as np
from scipy.stats import loguniform, uniform
import argparse
import os
import sys
import time
import json
import pandas as pd


def select_hyperparams(config, output_name, remove_checkpoints=False, score_key='f_macro'):
    ### make directories
    config_path, checkpoint_path, result_path = make_dirs(config)

    setup_params = ['tune_params', 'num_search_trials', 'dir_name']
    model_params = set()
    for p in config:
        if p in setup_params or ('range' in p or 'algo' in p or 'type' in p or p.startswith('CON')): continue
        model_params.add(p)
    print("[model params] {}".format(model_params))

    score_lst = []
    time_lst = []
    tn2vals = dict()
    for trial_num in range(int(config['num_search_trials'])):
        ### sample values
        print("[trial {}] Starting...".format(trial_num))
        print("[trial {}] sampling parameters in {}".format(trial_num, config['tune_params']))

        constraints_OK = False
        while not constraints_OK:
            p2v = sample_values()
            constraints_OK = check_constraints(config, p2v)
        tn2vals[trial_num] = p2v


        ### construct the appropriate config file
        config_file_name = config_path + 'config-{}.txt'.format(trial_num)
        print("[trial {}] writing configuration to {}".format(trial_num, config_file_name))
        print("[trial {}] checkpoints to {}".format(trial_num, checkpoint_path))
        print("[trial {}] results to {}".format(trial_num, result_path))
        f = open(config_file_name, 'w')
        model_name = '{}_t{}'.format(config['name'], trial_num)
        f.write('name:{}\n'.format(model_name)) # include trial number in name
        f.write('ckp_path:{}\n'.format(checkpoint_path)) # checkpoint save location
        f.write('res_path:{}\n'.format(result_path)) # results save location
        for p in model_params:
            if p == 'name': continue
            f.write('{}:{}\n'.format(p, config[p]))
        for p in p2v:
            f.write('{}:{}\n'.format(p, p2v[p]))
        f.flush()

        ### run the script
        start_time = time.time()
        print("[trial {}] running".format(trial_num))
        os.system("./train.sh {} 0 {}".format(config_file_name, score_key))
        script_time = (time.time() - start_time) / 60.
        print("[trial {}] running took {:.4f} minutes".format(trial_num, script_time))

        ### process the result and update information on best
        if 'enc' in config:
            res_f = open('{}{}_t{}-{}.top5_{}.txt'.format(result_path, config['name'], trial_num, config['enc'],
                                                          score_key), 'r')
        else:
            res_f = open('{}{}_t{}.top5_{}.txt'.format(result_path, config['name'], trial_num,
                                                       score_key), 'r')
        res_lines = res_f.readlines()
        score_lst.append(res_lines[-2].strip().split(':')[1])
        time_lst.append(script_time)

        if remove_checkpoints:
            checkpoint_path = '../checkpoints/{}-{}trials/'.format(config['dir_name'],
                                                                     config['num_search_trials'])
            for fname in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, fname))
            print("[trial {}] removing checkpoints from {}".format(trial_num, checkpoint_path))
        print("[trial {}] Done.".format(trial_num))
        print()

    ### save the resulting scores and times, for calculating the expected validation f1
    data = []
    for ti in tn2vals:
        data.append([ti, score_lst[ti], time_lst[ti], json.dumps(tn2vals[ti])])
    df = pd.DataFrame(data, columns=['trial_num', 'avg_score', 'time', 'param_vals'])
    df.to_csv('../data/{}-{}trials/{}'.format(config['dir_name'], config['num_search_trials'],
                                                      output_name), index=False)
    print("results to {}".format(output_name))


def parse_config(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    n2info = dict()
    for l in lines:
        n, info = l.strip().split(':')
        n2info[n] = info

    n2info['tune_params'] = n2info['tune_params'].split(',')
    for p in n2info['tune_params']:
        t = n2info['{}_type'.format(p)]
        n2info['{}_range'.format(p)] = list(map(lambda x: int(x) if t == 'int' else
                                                    float(x) if t == 'float' else x,
                                                    n2info['{}_range'.format(p)].split('-')))
    return n2info


def sample_values():
    p2v = dict()
    for p in config['tune_params']:
        a = config['{}_algo'.format(p)]
        if a != 'choice':
            min_v, max_v = config['{}_range'.format(p)]
            if a == 'loguniform':
                p2v[p] = loguniform.rvs(min_v, max_v)
            elif a == 'uniform-integer':
                p2v[p] = np.random.randint(min_v, max_v + 1)
            elif a == 'uniform-float':
                p2v[p] = uniform.rvs(min_v, max_v)
            else:
                print("ERROR: sampling method specified as {}".format(a))
        else:  # a == 'choice'
            p2v[p] = np.random.choice(config['{}_range'.format(p)])
    return p2v


def check_constraints(n2info, p2v):
    constraints_OK = True
    for n in n2info:
        if not n.startswith('CON'): continue
        eq = n2info[n].split('.') # equations should be in format param1.symbol.param2
        i = 0
        vlst = [None, None]
        vidx = 0
        s = None
        prev_s = None
        while i < len(eq):
            if eq[i] in {'<', '<=', '>', '>=', '==', '!='}:
                s = eq[i]
                vidx += 1
            else:
                if eq[i] in p2v:
                    v = p2v[eq[i]]
                elif eq[i] in n2info:
                    v = float(n2info[eq[i]])
                elif eq[i] in {'*', '+'}:
                    v = None
                    prev_s = eq[i]
                else:
                    v = float(eq[i])
                if v is not None and prev_s is not None:
                    if prev_s == '+':

                        vlst[vidx] = vlst[vidx] + v
                    else: # prev_s == '*':
                        vlst[vidx] = vlst[vidx] * v
                    prev_s = None
                elif v is not None:
                    vlst[vidx] = v
            i += 1
        con_res = parse_equation(vlst[0], s, vlst[1])

        constraints_OK = con_res and constraints_OK
    return constraints_OK


def parse_equation(v1, s, v2):
    if s == '<': return v1 < v2
    elif s == '<=': return v1 <= v2
    elif s == '=': return v1 == v2
    elif s == '!=': return v1 != v2
    elif s == '>': return v2 > v2
    elif s == '>=': return v1 >= v2
    else:
        print("ERROR: symbol {} not recognized".format(s))
        sys.exit(1)


def make_dirs(config):
    config_path = '../config/{}-{}trials/'.format(config['dir_name'],
                                                    config['num_search_trials'])
    checkpoint_path = '../checkpoints/{}-{}trials/'.format(config['dir_name'],
                                                             config['num_search_trials'])
    result_path = '../data/{}-{}trials/'.format(config['dir_name'],
                                             config['num_search_trials'])
    for p_name, p_path in [('config_path', config_path), ('ckp_path', checkpoint_path),
                           ('result_path', result_path)]:
        if not os.path.exists(p_path):
            os.makedirs(p_path)
        else:
            print("[{}] Directory {} already exists!".format(p_name, p_path))
            sys.exit(1)
    return config_path, checkpoint_path, result_path


def remove_dirs(config):
    config_path = '../config/{}-{}trials/'.format(config['dir_name'],
                                                    config['num_search_trials'])
    checkpoint_path = '../checkpoints/{}-{}trials/'.format(config['dir_name'],
                                                             config['num_search_trials'])
    result_path = '../data/{}-{}trials/'.format(config['dir_name'],
                                             config['num_search_trials'])
    for p_name, p_path in [('config_path', config_path), ('ckp_path', checkpoint_path),
                           ('result_path', result_path)]:
        if not os.path.exists(p_path):
            print("[{}] directory {} doesn't exist".format(p_name, p_path))
            continue
        else:
            print("[{}] removing all files from {}".format(p_name, p_path))
            for fname in os.listdir(p_path):
                os.remove(os.path.join(p_path, fname))
            print("[{}] removing empty directory".format(p_name))
            os.rmdir(p_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='What to do', required=True)
    parser.add_argument('-s', '--settings', help='Name of the file containing hyperparam info', required=True)
    parser.add_argument('-o', '--output', help='Name of the output file (full path)', required=False,
                        default='trial_results.csv')
    parser.add_argument('-r', '--remove_ckp', help='Remove checkpoints after running trial', required=False,
                        default=0, type=int)
    parser.add_argument('-k', '--score_key', help='Score used in optimization', required=False,
                        default='f_macro')
    args = vars(parser.parse_args())

    config = parse_config(args['settings'])

    if args['mode'] == '1':
        ## run hyperparam serarch
        remove_dirs(config)
        select_hyperparams(config, args['output'],
                           remove_checkpoints=(args['remove_ckp'] == 1), score_key=args['score_key'])
    elif args['mode'] == '2':
        ## remove directories
        remove_dirs(config)
    else:
        print("ERROR. exiting")