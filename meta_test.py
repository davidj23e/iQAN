import argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
# from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import os
from utils import AverageMeter, accuracy, calculate_caption_lengths

from utils import Vocabulary
from utils import get_glove_embedding
from utils import DatasetLoader, CategoriesSampler, collate_fn, ValCategoriesSampler
from utils import load_vocab
from utils import process_lengths
from utils import NLGEval
from torch.utils.data import DataLoader
import learn2learn as l2l
import numpy as np

from models import VQGNet, recast_category_embeds

from models import Autoencoder, CategoryEncoder, Encoder, Decoder, DecoderWithAttention
from tqdm import tqdm
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

torch.cuda.set_device(1)
torch.manual_seed(7)
np.random.seed(7)

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

def calc_loss_and_acc(loss_fn, alpha_c, preds, alphas, targets):
    
    decode_lengths = process_lengths(targets)

    preds = pack_padded_sequence(preds, decode_lengths, batch_first=True, enforce_sorted=False)
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)

        # some kind of regularization used by Show Attend and tell repo
    att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()
        
        # loss
    loss = loss_fn(preds.data, targets.data)
    loss += att_regularization
    acc1 = accuracy(preds.data, targets.data, 1)
    acc5 = accuracy(preds.data, targets.data, 5)
    return loss, acc1, acc5


def fast_adapt(train_batch, test_batch, learner, alpha_c, loss_fn, adaptation_steps):
    # data, labels = batch
    # data, labels = data.to(device), labels.to(device)

    # # Separate data into adaptation/evalutation sets
    # adaptation_indices = np.zeros(data.size(0), dtype=bool)
    # adaptation_indices[np.arange(shots*ways) * 2] = True
    # evaluation_indices = torch.from_numpy(~adaptation_indices)
    # adaptation_indices = torch.from_numpy(adaptation_indices)
    # adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    # evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]


    img_train, cap_train, cat_train = train_batch
    img_test, cap_test, cat_test = test_batch
    # Adapt the model>
    train_targets = cap_train[:, 1:]
    eval_targets = cap_test[:, 1:]
    # for p in learner.parameters():
    #         p.retain_grad()
    for step in range(adaptation_steps):
        preds, alphas = learner(img_train, cat_train, cap_train)
        train_error,_,_ = calc_loss_and_acc(loss_fn, alpha_c, preds, alphas, train_targets)
        train_error /= preds.size(0)
        learner.adapt(train_error)

    # Evaluate the adapted model
    preds, alphas = learner(img_test, cat_test, cap_test)
    valid_error, acc1, acc5 = calc_loss_and_acc(loss_fn, alpha_c, preds, alphas, eval_targets)
    valid_error /= preds.size(0)
    # valid_accuracy, acc1, acc5 = calc_loss_and_acc(predictions, evaluation_labels)
    return valid_error, acc1, acc5





def adapt_and_eval(train_batch, test_batch, word_dict, nlge, learner, alpha_c, loss_fn, adaptation_steps):
    # data, labels = batch
    # data, labels = data.to(device), labels.to(device)

    # # Separate data into adaptation/evalutation sets
    # adaptation_indices = np.zeros(data.size(0), dtype=bool)
    # adaptation_indices[np.arange(shots*ways) * 2] = True
    # evaluation_indices = torch.from_numpy(~adaptation_indices)
    # adaptation_indices = torch.from_numpy(adaptation_indices)
    # adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    # evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]


    img_train, cap_train, cat_train = train_batch
    img_test, cap_test, cat_test = test_batch
    # Adapt the model
    train_targets = cap_train[:, 1:]
    eval_targets = cap_test[:, 1:]
    # for p in learner.parameters():
    #         p.retain_grad()
    for step in range(adaptation_steps):
        preds, alphas = learner(img_train, cat_train, cap_train)
        train_error,_,_ = calc_loss_and_acc(loss_fn, alpha_c, preds, alphas, train_targets)
        train_error /= preds.size(0)
        learner.adapt(train_error)

    # Evaluate the adapted model
    # category_embeds = learner.module.category_encoder(cat_test)
    # recasted_embeds = recast_category_embeds(category_embeds, img_test.size(2))

    # n_img_features = img_test.mul(recasted_embeds).permute(0, 2, 1)
    cat_embeds, cat_bias = learner.module.category_encoder(cat_test)
    recast_embeds, recast_bias = recast_category_embeds(cat_embeds, cat_bias, img_test.size(2))

    n_img_features = img_test.mul(recast_embeds) + recast_bias
    n_img_features = n_img_features.permute(0, 2, 1)
    beam_size = 3
    predictions = []
    gts = []
    for i in range(img_test.size(0)):
        new_img_features = n_img_features[i].unsqueeze(0)
        new_img_features = new_img_features.expand(beam_size, new_img_features.size(1), new_img_features.size(2))
        outputs, alphas = learner.module.decoder.caption(new_img_features, word_dict, beam_size)
        output = word_dict.tokens_to_words(outputs)                    
        predictions.append(output)


        question = word_dict.tokens_to_words(cap_test[i])
        gts.append(question)
    # print('='*80)
    # print('GROUND TRUTH')
    # print(gts[:10])
    # print('-'*80)
    # print('PREDICTIONS')
    # print(predictions[:10])
    # print('='*80)
    scores = nlge.compute_metrics(ref_list=[gts], hyp_list=predictions)
    return scores

    # preds, alphas = learner(img_test, cat_test, cap_test)
    # valid_error, acc1, acc5 = calc_loss_and_acc(loss_fn, alpha_c, preds, alphas, eval_targets)
    # valid_error /= preds.size(0)
    # # valid_accuracy, acc1, acc5 = calc_loss_and_acc(predictions, evaluation_labels)
    # return valid_error, acc1, acc5

def main(args):
    writer = SummaryWriter()

    word_dict = load_vocab(args.vocab_path)
    vocabulary_size = len(word_dict)
    nlge = NLGEval(no_glove=True, no_skipthoughts=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(args.crop_size,
                                     scale=(1.00, 1.2),
                                     ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    new_transform =  transforms.Compose([
        transforms.ToTensor()])

    # trainset = DatasetLoader(args.dataset, 
    #                          transform=new_transform, 
    #                          max_examples=None)
    
    # train_sampler = CategoriesSampler(trainset.labeln,
    #                                   trainset.unique_labels,
    #                                   args.num_batch,
    #                                   args.way,
    #                                   args.train_query,
    #                                   args.test_query)
    
    # train_loader = DataLoader(trainset,
    #                         #   batch_size = args.batch_size,
    #                           batch_sampler=train_sampler,
    #                           num_workers=8,
    #                           collate_fn=collate_fn)
    #                         #   pin_memory=True)


    
    valset = DatasetLoader(args.val_dataset, 
                             transform=new_transform, 
                             max_examples=None)

    with open(args.finetune_cats, 'r') as fid:
        label_dict = json.load(fid)
    label_combos = catname2list(label_dict)
    val_sampler = ValCategoriesSampler(valset.labeln, valset.unique_labels,
                                    label_combos,
                                    # args.way,
                                    args.train_query,
                                    args.test_query)
    
    val_loader = DataLoader(valset,
                            batch_sampler=val_sampler,
                            num_workers=8,
                            collate_fn=collate_fn)                   
 
 
    vqg_net = VQGNet(args.num_categories, 
                    args.hidden_size, 
                    vocabulary_size,
                    cat_model=args.cat_model,
                    decoder_model=args.decoder_model)
    # vqg_net.retain_grad()
    maml = l2l.algorithms.MAML(vqg_net, lr=args.lr, first_order=False)
 

    
    # cross_entropy_loss = nn.CrossEntropyLoss().cuda()




    optimizer = optim.Adam(maml.parameters(), args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').cuda()
    # for epoch in range(43, args.epochs):
    #     scheduler.step()
        # meta_train(100, maml, optimizer, cross_entropy_loss, train_loader, word_dict, args.alpha_c, args.way, args.train_query, None)
        # torch.save(vqg_net.decoder.state_dict(),
        #             os.path.join(args.model_path, 'meta/','decoder/',
        #             'Decoder-epoch-%d.pkl'% (epoch+1)))
        # torch.save(vqg_net.category_encoder.state_dict(),
        #             os.path.join(args.model_path, 'meta/','category_encoder/',
        #             'cat_encoder-epoch-%d.pkl'% (epoch+1)))
    meta_test(100,nlge, maml, cross_entropy_loss, val_loader, word_dict, args.alpha_c, args.way, args.train_query, None)

# def meta_train(epochs, maml, optimizer, cross_entropy_loss,
#                 train_loader, word_dict, alpha_c, way, shot, log_interval):


#     meta_batch_size = 32    
#     tqdm_gen = tqdm(range(epochs))
        

#         # meta_valid_error = 0.0
#         # meta_valid_accuracy = 0.0
#     K = way
#     N = shot
#     p = K*N
#         # for p in maml.parameters():
#         #     p.retain_grad()
#     for epoch in tqdm_gen:
#         meta_train_error = AverageMeter()
#         meta_train_accuracy1 = AverageMeter()
#         meta_train_accuracy5 = AverageMeter()
#         optimizer.zero_grad()
#         for i, (imgs, captions, _,  _, cat, _) in enumerate(train_loader):

            
#             imgs = imgs.cuda().squeeze()

#             captions = captions.cuda()
#             cat = cat.cuda()
#             # imgs, captions, qlengths, ans, cat, qindices = [_.cuda() for _ in batch]
#             img_train, img_test = imgs[:p], imgs[p:]
#             cap_train, cap_test = captions[:p], captions[p:]
#             # qlengths_train, qlengths_test = qlengths[:p], qlengths[p:]
#             # ans_train, ans_test = ans[:p], ans[p:]
#             cat_train, cat_test = cat[:p], cat[p:]
#             # qindices_train, qindices_test = qindices[:p], qindices[p:]

#             train_batch = (img_train, cap_train, cat_train)
#             test_batch = (img_test, cap_test, cat_test)
#             # Compute meta-training loss
#             learner = maml.clone()
#             # batch = tasksets.train.sample()
#             evaluation_error, evaluation_accuracy1, evaluation_accuracy5 = fast_adapt(train_batch, test_batch,
#                                                                learner, alpha_c,
#                                                                cross_entropy_loss,
#                                                                2)

#             evaluation_error.backward()
#             meta_train_error.update(evaluation_error.item())
#             meta_train_accuracy1.update(evaluation_accuracy1)
#             meta_train_accuracy5.update(evaluation_accuracy5)


#         # meta_train_error /= meta_batch_size
#         # meta_train_accuracy1 /= meta_batch_size
#         # meta_train_accuracy5 /= meta_batch_size
#         tqdm_gen.set_description('Epoch {},'
#                                  ' Train Loss={:.4f} Train Acc1={:.4f}'
#                                  ' train Acc5={:.4f}'.format(epoch, 
#                                                              meta_train_error.avg, 
#                                                              meta_train_accuracy1.avg,
#                                                              meta_train_accuracy5.avg))

 

#         # Average the accumulated gradients and optimize
#         for param in maml.parameters():
#             param.grad.data.mul_(1.0 / meta_batch_size)
#         optimizer.step()


def meta_test(runs, nlge, maml, cross_entropy_loss, val_loader, word_dict, alpha_c, way, train_query, log_interval):
    K = way
    N = train_query
    p = K*N
    # meta_batch_size = 32
    bleu_1 = AverageMeter()
    bleu_2 = AverageMeter()
    bleu_3 = AverageMeter()
    bleu_4 = AverageMeter()
    meteor = AverageMeter()
    rouge_l = AverageMeter()
    cider = AverageMeter()
    tqdm_gen = tqdm(range(runs))
    for epoch in tqdm_gen:        
        for i, (imgs, captions, _,  _, cat, _) in enumerate(val_loader):

            imgs = imgs.cuda().squeeze()

            captions = captions.cuda()
            cat = cat.cuda()

            img_train, img_test = imgs[:p], imgs[p:]
            cap_train, cap_test = captions[:p], captions[p:]
            # qlengths_train, qlengths_test = qlengths[:p], qlengths[p:]
            # ans_train, ans_test = ans[:p], ans[p:]
            cat_train, cat_test = cat[:p], cat[p:]
            # qindices_train, qindices_test = qindices[:p], qindices[p:]

            train_batch = (img_train, cap_train, cat_train)
            test_batch = (img_test, cap_test, cat_test)
            
            learner = maml.clone()
            
            scores = adapt_and_eval(train_batch, test_batch,
                                    word_dict, nlge,
                                    learner, alpha_c,
                                    cross_entropy_loss,
                                    2)

            bleu_1.update(scores['Bleu_1'])
            bleu_2.update(scores['Bleu_2'])
            bleu_3.update(scores['Bleu_3'])
            bleu_4.update(scores['Bleu_4'])
            meteor.update(scores['METEOR'])
            rouge_l.update(scores['ROUGE_L'])
            cider.update(scores['CIDEr'])


        # meta_test_error /= meta_batch_size
        # meta_test_accuracy1 /= meta_batch_size
        # meta_test_accuracy5 /= meta_batch_size
        tqdm_gen.set_description('RUN {},'
                                 ' Bleu 1={:.4f} Bleu 2={:.4f}'
                                 ' Bleu 3={:.4f} Bleu 4={:.4f}'
                                 ' METEOR={:.4f} ROUGE_L={:.4f}'
                                 ' CIDEr={:.4f}'.format(epoch, 
                                                        bleu_1.avg, bleu_2.avg,
                                                        bleu_3.avg, bleu_4.avg,
                                                        meteor.avg, rouge_l.avg,
                                                        cider.avg))
                                                        # meta_test_accuracy1.avg,
                                                            #  meta_test_accuracy5.avg))




    print('Bleu 1={:.4f} Bleu 2={:.4f}'
          ' Bleu 3={:.4f} Bleu 4={:.4f}'
          ' METEOR={:.4f} ROUGE_L={:.4f}'
          ' CIDEr={:.4f}'.format(bleu_1.avg, bleu_2.avg,
                                 bleu_3.avg, bleu_4.avg,
                                 meteor.avg, rouge_l.avg,
                                 cider.avg))


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--num-batch', type=int, default=32, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='E',
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate of the decoder (default: 1e-4)')
    parser.add_argument('--step-size', type=int, default=5,
                        help='step size for learning rate annealing (default: 5)')
    parser.add_argument('--alpha-c', type=float, default=1, metavar='A',
                        help='regularization constant (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                        help='number of batches to wait before logging training stats (default: 100)')
    parser.add_argument('--data', type=str, default='data/coco',
                        help='path to data images (default: data/coco)')
    parser.add_argument('--network', choices=['vgg19', 'resnet152', 'densenet161'], default='resnet152',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--decoder-model', type=str, 
                        default='model/meta/nbn_decoder/Decoder-epoch-413.pkl',
                        help='path to model')
    parser.add_argument('--cat-model', type=str,
                        default='model/meta/nbn_category_encoder/cat-encoder-epoch-413.pkl', 
                        help='path to model')
    parser.add_argument('--tf', action='store_true', default=False,
                        help='Use teacher forcing when training LSTM (default: False)')

    parser.add_argument('--finetune-cats', type=str, default='data/processed/finetune_task_cats.json')
    
    # Data parameters.
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_iq.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,
                        default='data/processed/new_train_iq_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val-dataset', type=str,
                        default='data/processed/latest_val_iq_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--train-dataset-weights', type=str,
                        default='data/processed/iq_train_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--val-dataset-weights', type=str,
                        default='data/processed/iq_val_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--cat2name', type=str,
                        default='data/processed/cat2name.json',
                        help='Location of mapping from category to type name.')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Location of where the model weights are.')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='Size for randomly cropping images')

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-examples', type=int, default=1024,
                        help='For debugging. Limit examples in database.')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--num-categories', type=int, default=16,
                        help='Number of answer types we use.')
    
    parser.add_argument('--model-path', type=str, default='model',
                        help='path to model')
    # parser.add_argument('--num_batch', type=int, default=100) # The number for different tasks used for meta-train
    parser.add_argument('--way', type=int, default=3) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=10) # (Shot) The number of meta train samples for each class in a task
    parser.add_argument('--test_query', type=int, default=10) # The number of meta test samples for each class in a task
    main(parser.parse_args())
