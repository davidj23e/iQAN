import argparse
import os
import shutil
import yaml
import json
import click
from pprint import pprint
from data_loader import DatasetLoader, collate_fn
from torch.utils.data import DataLoader
from samplers import ValCategoriesSampler, CategoriesSampler
from vocab import load_vocab
from torchvision import transforms
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import vqa.lib.utils as utils
import vqa.lib.logger as logger
import vqa.datasets as datasets

# task specific package
import models.dual_model as models
import dual_model.lib.engine_v2 as engine
from vqg.lib.utils import set_trainable

import pdb
# torch.cuda.set_device(1)
model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(
    description='Train/Evaluate models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
##################################################
# yaml options file contains all default choices #
parser.add_argument('--path_opt', default='options/dual_model/default.yaml', type=str, 
                    help='path to a yaml options file')
################################################
# change cli options to modify default choices #
# logs options
parser.add_argument('--dir_logs', type=str, help='dir logs')
# data options
parser.add_argument('--vqa_trainsplit', type=str, choices=['train','trainval'])
# model options
parser.add_argument('--arch', choices=model_names,
                    help='vqa model architecture: ' +
                        ' | '.join(model_names))
# parser.add_argument('--st_type',
#                     help='skipthoughts type')
# parser.add_argument('--emb_drop', type=float,
#                     help='embedding dropout')
# parser.add_argument('--st_dropout', type=float)
# parser.add_argument('--st_fixed_emb', default=None, type=utils.str2bool,
#                     help='backprop on embedding')
# optim options
parser.add_argument('-lr', '--learning_rate', type=float,
                    help='initial learning rate')
parser.add_argument('-b', '--batch_size', type=int,
                    help='mini-batch size')
parser.add_argument('--epochs', type=int,
                    help='number of total epochs to run')
parser.add_argument('--eval_epochs', type=int, default=10,
                    help='Number of epochs to evaluate the model')
# options not in yaml file          
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint')
parser.add_argument('--save_model', default=True, type=utils.str2bool,
                    help='able or disable save model and optim state')
parser.add_argument('--save_all_from', type=int,
                    help='''delete the preceding checkpoint until an epoch,'''
                         ''' then keep all (useful to save disk space)')''')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation and test set')
parser.add_argument('--print_freq', '-p', default=1000, type=int,
                    help='print frequency')
################################################
parser.add_argument('-ho', '--help_opt', dest='help_opt', action='store_true',
                    help='show selected options before running')
parser.add_argument('--beam_search', action='store_true', help='whether to use beam search, the batch_size will be set to 1 automatically')

parser.add_argument('--dual_training', action='store_true', help='Whether to use additional loss')

parser.add_argument('--share_embeddings', action='store_true', help='Whether to share the embeddings')
parser.add_argument('--partial', type=float, default=-1., 
    help='Only use part of the VQA dataset. Valid range is (0, 1). [default: -1.]')

parser.add_argument('--alternative_train', type=float, default=-1., 
    help='The sample rate for QG training. if [alternative_train] > 1 or < 0, then jointly train.')

parser.add_argument('--train-dataset', type=str,
                        default='data/processed/latest_train_iq_dataset.hdf5',
                        help='Path for train annotation json file.')
parser.add_argument('--val-dataset', type=str,
                        default='data/processed/latest_val_iq_dataset.hdf5',
                        help='Path for train annotation json file.')
parser.add_argument('--finetune-cats', type=str, default='finetune_task_cats.json')
parser.add_argument('--alpha-c', type=float, default=1, metavar='A',
                        help='regularization constant (default: 1)')
parser.add_argument('--num-batch', type=int, default=32, metavar='N',
                        help='batch size for training (default: 64)')
parser.add_argument('--way', type=int, default=3) # Way number, how many classes in a task
parser.add_argument('--train_query', type=int, default=10) # (Shot) The number of meta train samples for each class in a task
parser.add_argument('--test_query', type=int, default=10) # The nu
parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_iq.json',
                        help='Path for vocabulary wrapper.')
best_acc1 = 0.
best_acc5 = 0.
best_acc10 = 0.
best_loss_q = 1000.


def catname2list(label_dict):
    with open('data/processed/cat2name.json', 'r') as fid:
        cats = json.load(fid)
    label_list = []
    for task_labels in label_dict:
        temp_list = []
        nameList = list(task_labels.values())[1]
        # print(nameList)
        for name in nameList:
            temp_list.append(cats.index(name))
        label_list.append(temp_list)
    # print(label_list)
    # exit()
    return label_list


def main():
    to_set_trainable = True
    global args, best_acc1, best_acc5, best_acc10, best_loss_q
    args = parser.parse_args()

    # Set options
    options = {
        'vqa' : {
            'trainsplit': args.vqa_trainsplit,
            'partial': args.partial, 
        },
        'logs': {
            'dir_logs': args.dir_logs
        },
        'optim': {
            'lr': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'eval_epochs': args.eval_epochs,
        }
    }
    if args.path_opt is not None:
        with open(args.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        options = utils.update_values(options, options_yaml)

    if 'model' not in options.keys():
        options['model'] = {}

    
    if args.dual_training:
        options['logs']['dir_logs'] += '_dual_training'
    print('## args'); pprint(vars(args))
    print('## options'); pprint(options)

    if args.help_opt:
        return

    # Set datasets
    print('Loading dataset....',)
    word_dict = load_vocab(args.vocab_path)
    new_transform =  transforms.Compose([
        transforms.ToTensor()])
    vocabulary_size = len(word_dict)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224,
                                     scale=(1.00, 1.2),
                                     ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    trainset = DatasetLoader(options['vqa'], args.train_dataset, 
                             transform=new_transform,
                             vocab=word_dict, 
                             max_examples=None)
    # trainset.wid_to_word = word_dict.idx2word.copy()
    # trainset.word_to_wid = word_dict.word2idx.copy()
    valset = DatasetLoader(options['vqa'], args.train_dataset, 
                             transform=new_transform,
                             vocab=word_dict, 
                             max_examples=None)

    with open(args.finetune_cats, 'r') as fid:
        label_dict = json.load(fid)
    label_combos = catname2list(label_dict)
    # train_sampler = CategoriesSampler(trainset.labeln, trainset.unique_labels,
    #                                 label_combos,
    #                                 args.way,
    #                                 args.train_query,
    #                                 args.test_query)
    
    train_loader = DataLoader(trainset,
                            batch_size=200,
                            num_workers=8,
                            collate_fn=collate_fn)  

    # valset.wid_to_word = word_dict.idx2word.copy()
    # valset.word_to_wid = word_dict.word2idx.copy()
    # val_sampler = CategoriesSampler(valset.labeln, valset.unique_labels,
    #                                 label_combos,
    #                                 # args.way,
    #                                 args.train_query,
    #                                 args.test_query)
    
    val_loader = DataLoader(valset,
                            batch_size=200,
                            num_workers=8,
                            collate_fn=collate_fn)  

    print('Done.')
    print('Setting up the model...')

    # Set model, criterion and optimizer
    # assert options['model']['arch_resnet'] == options['coco']['arch'], 'Two [arch] should be set the same.'
    model = getattr(models, options['model']['arch'])(
        options['model'], trainset.vocab_words(), trainset.vocab_answers())

    if args.share_embeddings:
        model.set_share_parameters()
 
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), 
        lr=options['optim']['lr'], weight_decay=options['optim']['weight_decay'])

    # Optionally resume from a checkpoint
    exp_logger = None
    # print('Loading saved model...')
    # args.start_epoch, best_acc1, exp_logger = load_checkpoint(model, optimizer,#model.module, optimizer,
    #       os.path.join(options['logs']['dir_logs'], args.resume))
        
    if exp_logger is None:
        # Set loggers
        exp_name = os.path.basename(options['logs']['dir_logs']) # add timestamp
        exp_logger = logger.Experiment(exp_name, options)
        exp_logger.add_meters('train', make_meters())
        exp_logger.add_meters('test', make_meters())
        if options['vqa']['trainsplit'] == 'train':
            exp_logger.add_meters('val', make_meters())
        exp_logger.info['model_params'] = utils.params_count(model)
        print('Model has {} parameters'.format(exp_logger.info['model_params']))

    # Begin evaluation and training
    model = model.cuda()
    # if args.evaluate:
    #     print('Start evaluating...')
    #     path_logger_json = os.path.join(options['logs']['dir_logs'], 'logger.json')

    #     evaluate_result = engine.evaluate(test_loader, model, exp_logger, args.print_freq)
    
    #     pdb.set_trace()
    #     save_results(evaluate_result, args.start_epoch, valset.split_name(),
    #                      options['logs']['dir_logs'], options['vqa']['dir'])

    #     return

    print('Start training')
    for epoch in range(args.start_epoch, options['optim']['epochs']):
        engine.new_train(train_loader, model, optimizer,
                      exp_logger, epoch, args.print_freq, 
                      dual_training=args.dual_training, 
                      alternative_train=args.alternative_train)

        if options['vqa']['trainsplit'] == 'train':
            # evaluate on validation set
            acc1, acc5, acc10, loss_q = engine.new_validate(val_loader, model,
                                                exp_logger, epoch, args.print_freq)
            # if (epoch + 1) % options['optim']['eval_epochs'] == 0:
            #     #print('[epoch {}] evaluation:'.format(epoch))
            #     evaluate_result = engine.evaluate(test_loader, model, exp_logger, args.print_freq)   #model.module, exp_logger, args.print_freq)
            #     save_results(evaluate_result, epoch, valset.split_name(),
            #              options['logs']['dir_logs'], options['vqa']['dir'], is_testing=False)

            # remember best prec@1 and save checkpoint
            is_best = acc1 > best_acc1
            is_best_q = loss_q < best_loss_q
            best_acc5 = acc5 if is_best else best_acc5
            best_acc10 = acc10 if is_best else best_acc10
            best_acc1 = acc1 if is_best else best_acc1
            best_loss_q = loss_q if is_best_q else best_loss_q

            print('** [Best]\tAcc@1: {0:.2f}%\tAcc@5: {1:.2f}%\tAcc@10: {2:.2f}% \tQ_Loss: {3:.4f}'.format(
                best_acc1, best_acc5, best_acc10, best_loss_q))
            save_checkpoint({
                    'epoch': epoch,
                    'arch': options['model']['arch'],
                    'best_acc1': best_acc1,
                    'best_acc5': best_acc5,
                    'best_acc10': best_acc10,
                    'exp_logger': exp_logger
                },
                model.state_dict(), #model.module.state_dict(),
                optimizer.state_dict(),
                options['logs']['dir_logs'],
                args.save_model,
                args.save_all_from,
                is_best, is_best_q)
        else:
            raise NotImplementedError
    

def make_meters():  
    meters_dict = {
        'loss': logger.AvgMeter(),
        'loss_a': logger.AvgMeter(),
        'loss_q': logger.AvgMeter(),
        'batch_time': logger.AvgMeter(),
        'data_time': logger.AvgMeter(),
        'epoch_time': logger.SumMeter(), 
        'bleu_score': logger.AvgMeter(), 
        'acc1': logger.AvgMeter(),
        'acc5': logger.AvgMeter(),
        'acc10': logger.AvgMeter(),
        'dual_loss': logger.AvgMeter(),
    }
    return meters_dict

def save_results(results, epoch, split_name, dir_logs, dir_vqa, is_testing=True):
    if is_testing:
        subfolder_name = 'evaluate'
    else:
        subfolder_name = 'epoch_' + str(epoch)
    dir_epoch = os.path.join(dir_logs, subfolder_name)
    name_json = 'OpenEnded_mscoco_{}_vqg_results.json'.format(split_name)
    # TODO: simplify formating
    if 'test' in split_name:
        name_json = 'vqa_' + name_json
    path_rslt = os.path.join(dir_epoch, name_json)
    os.system('mkdir -p ' + dir_epoch)
    with open(path_rslt, 'w') as handle:
        json.dump(results, handle)

def save_checkpoint(info, model, optim, dir_logs, save_model, save_all_from=None, is_best=True, is_best_q=True):
    os.system('mkdir -p ' + dir_logs)
    if save_all_from is None:
        path_ckpt_info  = os.path.join(dir_logs, 'ckpt_info.pth.tar')
        path_ckpt_model = os.path.join(dir_logs, 'ckpt_model.pth.tar')
        path_ckpt_optim = os.path.join(dir_logs, 'ckpt_optim.pth.tar')
        path_best_info  = os.path.join(dir_logs, 'best_info.pth.tar')
        path_best_model = os.path.join(dir_logs, 'best_model.pth.tar')
        path_best_optim = os.path.join(dir_logs, 'best_optim.pth.tar')
        path_best_info_q  = os.path.join(dir_logs, 'best_info_VQG.pth.tar')
        path_best_model_q = os.path.join(dir_logs, 'best_model_VQG.pth.tar')
        path_best_optim_q = os.path.join(dir_logs, 'best_optim_VQG.pth.tar')
        # save info & logger
        path_logger = os.path.join(dir_logs, 'logger.json')
        info['exp_logger'].to_json(path_logger)
        torch.save(info, path_ckpt_info)
        if is_best:
            shutil.copyfile(path_ckpt_info, path_best_info)
        if is_best_q:
            shutil.copyfile(path_ckpt_info, path_best_info_q)
        # save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model)
            torch.save(optim, path_ckpt_optim)
            if is_best:
                shutil.copyfile(path_ckpt_model, path_best_model)
                shutil.copyfile(path_ckpt_optim, path_best_optim)
            if is_best_q:
                shutil.copyfile(path_ckpt_model, path_best_model_q)
                shutil.copyfile(path_ckpt_optim, path_best_optim_q)
    else:
        is_best = False # because we don't know the test accuracy
        path_ckpt_info  = os.path.join(dir_logs, 'ckpt_epoch,{}_info.pth.tar')
        path_ckpt_model = os.path.join(dir_logs, 'ckpt_epoch,{}_model.pth.tar')
        path_ckpt_optim = os.path.join(dir_logs, 'ckpt_epoch,{}_optim.pth.tar')
        # save info & logger
        path_logger = os.path.join(dir_logs, 'logger.json')
        info['exp_logger'].to_json(path_logger)
        torch.save(info, path_ckpt_info.format(info['epoch']))
        # save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model.format(info['epoch']))
            torch.save(optim, path_ckpt_optim.format(info['epoch']))
        if  info['epoch'] > 1 and info['epoch'] < save_all_from + 1:
            os.system('rm ' + path_ckpt_info.format(info['epoch'] - 1))
            os.system('rm ' + path_ckpt_model.format(info['epoch'] - 1))
            os.system('rm ' + path_ckpt_optim.format(info['epoch'] - 1))
    if not save_model:
        print('Warning train.py: checkpoint not saved')
def load_checkpoint(model, optimizer, path_ckpt):
    path_ckpt_info  = path_ckpt + '_info.pth.tar'
    path_ckpt_model = path_ckpt + '_model.pth.tar'
    path_ckpt_optim = path_ckpt + '_optim.pth.tar'
    if os.path.isfile(path_ckpt_info):
        info = torch.load(path_ckpt_info)
        start_epoch = 0
        best_acc1   = 0
        exp_logger  = None
        if 'epoch' in info:
            start_epoch = info['epoch']
        else:
            print('Warning train.py: no epoch to resume')
        if 'best_acc1' in info:
            best_acc1 = info['best_acc1']
        else:
            print('Warning train.py: no best_acc1 to resume')
        if 'exp_logger' in info:
            exp_logger = info['exp_logger']
        else:
            print('Warning train.py: no exp_logger to resume')
    else:
        print("Warning train.py: no info checkpoint found at '{}'".format(path_ckpt_info))
    if os.path.isfile(path_ckpt_model):
        model_state = torch.load(path_ckpt_model)
        model.load_state_dict(model_state)
    else:
        print("Warning train.py: no model checkpoint found at '{}'".format(path_ckpt_model))
    #  if os.path.isfile(path_ckpt_optim):
    #      optim_state = torch.load(path_ckpt_optim)
    #      optimizer.load_state_dict(optim_state)
    #  else:
    #      print("Warning train.py: no optim checkpoint found at '{}'".format(path_ckpt_optim))
    print("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
              .format(path_ckpt, start_epoch, best_acc1))
    return start_epoch, best_acc1, exp_logger

if __name__ == '__main__':
    main()
    # parser.add_argument('--vocab-path', type=str,
    #                     default='data/processed/vocab_iq.json',
    #                     help='Path for vocabulary wrapper.')