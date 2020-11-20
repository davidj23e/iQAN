import time
import torch
import pdb
import numpy as np
from torch.autograd import Variable
import vqa.lib.utils as utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from models.vec2seq import process_lengths_sort, process_lengths
from nltk.translate.bleu_score import modified_precision as bleu_score
from models.criterions import *
from models.utils import translate_tokens, calculate_bleu_score
from tqdm import tqdm
import vqa.lib.logger as logger2
import torch
# def train(loader, model, optimizer, logger, epoch, print_freq=10, dual_training=False, alternative_train = -1.):
#     # switch to train mode
#     model.train()
#     model.set_testing(False)

#     meters = logger.reset_meters('train')
#     end = time.time()
#     for i, sample in enumerate(loader):

#         batch_size = sample['visual'].size(0)

#         # measure data loading time
#         meters['data_time'].update(time.time() - end, n=batch_size)
#         target_question = sample['question']
#         # To arrange the length of mini-batch by the descending order of question length
#         new_ids, lengths = process_lengths_sort(target_question) 
#         new_ids = Variable(new_ids).detach()
#         target_question = Variable(target_question.cuda())
#         input_visual = Variable(sample['visual'].cuda())
#         target_answer = Variable(sample['answer'].cuda( ))
        
#         # compute output
#         output =  model(input_visual, target_question, target_answer)
#         generated_a = output[0]
#         generated_q = output[1]
#         additional_loss =output[2].mean()
#         torch.cuda.synchronize()
        
#         # Hack for the compatability of reinforce() and DataParallel()
#         target_question = pack_padded_sequence(target_question.index_select(0, new_ids)[:, 1:], lengths, batch_first=True)[0]
#         output = pack_padded_sequence(generated_q.index_select(0, new_ids), lengths, batch_first=True)[0] 
#         loss_q = F.cross_entropy(output, target_question)
#         loss_a = F.cross_entropy(generated_a, target_answer)
#         if alternative_train > 1. or alternative_train < 0.:
#           loss = loss_a + loss_q 
#           if dual_training:
#             loss += additional_loss
#         else:
#           if torch.rand(1)[0] > alternative_train:
#             loss = loss_a
#           else:
#             loss = loss_q
#         # print(generated_a.data)
#         # measure accuracy 
#         acc1, acc5, acc10 = utils.accuracy(generated_a.data, target_answer.data, topk=(1, 5, 10))
#         # bleu_score = calculate_bleu_score(generated_q.cpu().data, sample['question'], loader.dataset.wid_to_word)
#         meters['acc1'].update(acc1.item(), n=batch_size)
#         meters['acc5'].update(acc5.item(), n=batch_size)
#         meters['acc10'].update(acc10.item(), n=batch_size)
#         meters['loss_a'].update(loss_a.data.item(), n=batch_size)
#         meters['loss_q'].update(loss_q.data.item(), n=batch_size)
#         meters['dual_loss'].update(additional_loss.item(), n=batch_size)
#         # meters['bleu_score'].update(bleu_score, n=batch_size)

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         torch.cuda.synchronize()
#         optimizer.step()
#         torch.cuda.synchronize()

#         # measure elapsed time
#         meters['batch_time'].update(time.time() - end, n=batch_size)
#         end = time.time()

#         if (i + 1) % print_freq == 0:
#             print('[Train]\tEpoch: [{0}][{1}/{2}] '
#                   'Time {batch_time.avg:.3f}\t'
#                   'Data {data_time.avg:.3f}\t'
#                   'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}, Dual Loss: {loss_d.avg:.3f}\t'
#                   'Acc@1 {acc1.avg:.3f}\t'
#                   'Acc@5 {acc5.avg:.3f}\t'
#                   'Acc@10 {acc10.avg:.3f}\t'.format(
#                    epoch, i + 1, len(loader),
#                    batch_time=meters['batch_time'], data_time=meters['data_time'],
#                    acc1=meters['acc1'], acc5=meters['acc5'], 
#                    acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q'], 
#                    loss_d=meters['dual_loss']))

#     print('[Train]\tEpoch: [{0}]'
#                   'Time {batch_time.avg:.3f}\t'
#                   'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}, Dual Loss: {loss_d.avg:.3f}\t'
#                   'Acc@1 {acc1.avg:.3f}\t'
#                   'Acc@5 {acc5.avg:.3f}\t'
#                   'Acc@10 {acc10.avg:.3f}\t'.format(
#                    epoch, 
#                    batch_time=meters['batch_time'], 
#                    acc1=meters['acc1'], acc5=meters['acc5'], 
#                    acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q'], 
#                    loss_d=meters['dual_loss']))

#     logger.log_meters('train', n=epoch)

def new_train(loader, model, optimizer, logger, epoch, print_freq=10, dual_training=False, alternative_train = -1.):
    # switch to train mode
    model.train()
    model.set_testing(False)

    meters = logger.reset_meters('train')
    end = time.time()
    tqdm_gen = tqdm(loader)
    for i,  (imgs, captions, qlengths,  answers, cat, qindices) in enumerate(tqdm_gen):
        # print(imgs.shape)
        batch_size = imgs.size(0) 
        # imgs = imgs.cuda()
        # captions = captions.cuda()
        # answers = answers.cuda()
        # cat = cat.cuda()
        num_classes = model.num_classes
        # print(num_classes)
        new_ans = torch.zeros((batch_size, 1)).long()
        for s in range(batch_size):
            ans = answers[s][1:]
            for a in ans:
                texta = loader.dataset.wid_to_word[str(a.item())]
                try:
                    aid = loader.dataset.ans_to_aid[texta]
                    new_ans[s, 0] = aid
                except KeyError:
                    pass
                
                

        imgs = imgs.permute(0, 2, 1, 3)
        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)
        target_question = captions
        # To arrange the length of mini-batch by the descending order of question length
        # new_ids, lengths = process_lengths_sort(target_question) 
        new_ids = qindices.cuda()
        target_question = Variable(target_question.cuda())
        input_visual = Variable(imgs.cuda())
        target_answer = Variable(new_ans.cuda())
        
        # compute output
        output =  model(input_visual, target_question, target_answer)
        generated_a = output[0]
        generated_q = output[1]
        additional_loss =output[2].mean()
        torch.cuda.synchronize()
        
        # Hack for the compatability of reinforce() and DataParallel()
        target_question = pack_padded_sequence(target_question.index_select(0, new_ids)[:, 1:], qlengths, batch_first=True, enforce_sorted=False)
        output = pack_padded_sequence(generated_q.index_select(0, new_ids), qlengths, batch_first=True, enforce_sorted=False) 
        loss_q = F.cross_entropy(output.data, target_question.data)
        # print(target_answer)
        # _, target_answer = torch.max(target_answer, 1)
        target_answer = target_answer.squeeze()
        loss_a = F.cross_entropy(generated_a, target_answer)
        if alternative_train > 1. or alternative_train < 0.:
          loss = loss_a + loss_q 
          if dual_training:
            loss += additional_loss
        else:
          if torch.rand(1)[0] > alternative_train:
            loss = loss_a
          else:
            loss = loss_q
        # print(generated_a.data)
        # measure accuracy 
        acc1, acc5, acc10 = utils.accuracy(generated_a.data, target_answer.data, topk=(1, 5, 10))
        # bleu_score = calculate_bleu_score(generated_q.cpu().data, sample['question'], loader.dataset.wid_to_word)
        meters['acc1'].update(acc1.item(), n=batch_size)
        meters['acc5'].update(acc5.item(), n=batch_size)
        meters['acc10'].update(acc10.item(), n=batch_size)
        meters['loss_a'].update(loss_a.data.item(), n=batch_size)
        meters['loss_q'].update(loss_q.data.item(), n=batch_size)
        meters['dual_loss'].update(additional_loss.item(), n=batch_size)
        # meters['bleu_score'].update(bleu_score, n=batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('[Train]\tEpoch: [{0}][{1}/{2}] '
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}, Dual Loss: {loss_d.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@5 {acc5.avg:.3f}\t'
                  'Acc@10 {acc10.avg:.3f}\t'.format(
                   epoch, i + 1, len(loader),
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   acc1=meters['acc1'], acc5=meters['acc5'], 
                   acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q'], 
                   loss_d=meters['dual_loss']))

    print('[Train]\tEpoch: [{0}]'
                  'Time {batch_time.avg:.3f}\t'
                  'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}, Dual Loss: {loss_d.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@5 {acc5.avg:.3f}\t'
                  'Acc@10 {acc10.avg:.3f}\t'.format(
                   epoch, 
                   batch_time=meters['batch_time'], 
                   acc1=meters['acc1'], acc5=meters['acc5'], 
                   acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q'], 
                   loss_d=meters['dual_loss']))

    logger.log_meters('train', n=epoch)


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# def validate(loader, model, logger, epoch=0, print_freq=100):
#     # switch to train mode
#     model.eval()
#     meters = logger.reset_meters('val')
#     end = time.time()
#     for i, sample in enumerate(loader):
#         batch_size = sample['visual'].size(0)
#         # measure data loading time
#         meters['data_time'].update(time.time() - end, n=batch_size)
#         target_question = sample['question']
#         # To arrange the length of mini-batch by the descending order of question length
#         new_ids, lengths = process_lengths_sort(target_question) 
#         target_question = Variable(target_question.cuda( ), volatile=True)
#         input_visual = Variable(sample['visual'].cuda( ), volatile=True)
#         target_answer = Variable(sample['answer'].cuda( ), volatile=True)
        
#         # compute output
#         output =  model(input_visual, target_question, target_answer)
#         generated_a = output[0]
#         generated_q = output[1]
#         additional_loss =output[2].mean()
#         torch.cuda.synchronize()
        
#         # Hack for the compatability of reinforce() and DataParallel()
#         target_question = pack_padded_sequence(target_question.index_select(0, new_ids)[:, 1:], lengths, batch_first=True)[0]
#         output = pack_padded_sequence(generated_q.index_select(0, new_ids), lengths, batch_first=True)[0] 
#         loss_q = F.cross_entropy(output, target_question)
#         loss_a = F.cross_entropy(generated_a, target_answer)
#         # measure accuracy 
#         acc1, acc5, acc10 = utils.accuracy(generated_a.data, target_answer.data, topk=(1, 5, 10))
#         # bleu_score = calculate_bleu_score(generated_q.cpu().data, sample['question'], loader.dataset.wid_to_word)
#         meters['acc1'].update(acc1.item(), n=batch_size)
#         meters['acc5'].update(acc5.item(), n=batch_size)
#         meters['acc10'].update(acc10.item(), n=batch_size)
#         meters['loss_a'].update(loss_a.item(), n=batch_size)
#         meters['loss_q'].update(loss_q.item(), n=batch_size)
#         meters['dual_loss'].update(additional_loss.item(), n=batch_size)
#         # measure elapsed time
#         meters['batch_time'].update(time.time() - end, n=batch_size)
#         # meters['bleu_score'].update(bleu_score, n=batch_size)
#         end = time.time()
            
#     print('[Val]\tEpoch: [{0}]'
#                   'Time {batch_time.avg:.3f}\t'
#                   'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}, Dual Loss: {loss_d.avg:.3f}\t'
#                   'Acc@1 {acc1.avg:.3f}\t'
#                   'Acc@5 {acc5.avg:.3f}\t'
#                   'Acc@10 {acc10.avg:.3f}\t'.format(
#                    epoch, 
#                    batch_time=meters['batch_time'], 
#                    acc1=meters['acc1'], acc5=meters['acc5'], 
#                    acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q'], 
#                    loss_d=meters['dual_loss']))

#     logger.log_meters('val', n=epoch)
#     return meters['acc1'].avg, meters['acc5'].avg, meters['acc10'].avg, meters['loss_q'].avg



def new_validate(loader, model, logger, epoch=0, print_freq=100):
    # switch to train mode
    model.eval()
    meters = logger.reset_meters('val')
    end = time.time()
    for i, (imgs, captions, qlengths,  answers, cat, qindices) in enumerate(loader):
        batch_size = imgs.size(0)
        imgs = imgs.permute(0, 2, 1, 3)
        num_classes = model.num_classes
        # print(num_classes)
        new_ans = torch.zeros((batch_size, 1)).long()
        for s in range(batch_size):
            ans = answers[s][1:]
            for a in ans:
                texta = loader.dataset.wid_to_word[str(a.item())]
                try:
                    aid = loader.dataset.ans_to_aid[texta]
                    new_ans[s, 0] = aid
                except KeyError:
                    pass
                

        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)
        target_question = captions
        # To arrange the length of mini-batch by the descending order of question length
        # new_ids, lengths = process_lengths_sort(target_question) 
        new_ids = qindices.cuda()
        lengths = qlengths
        target_question = Variable(target_question.cuda())
        input_visual = Variable(imgs.cuda())
        target_answer = Variable(new_ans.cuda())
        
        # compute output
        output =  model(input_visual, target_question, target_answer)
        generated_a = output[0]
        generated_q = output[1]
        additional_loss =output[2].mean()
        torch.cuda.synchronize()
        
        # Hack for the compatability of reinforce() and DataParallel()
        new_target_question = target_question[:, 1:]
        target_question = pack_padded_sequence(target_question.index_select(0, new_ids)[:, 1:], lengths, batch_first=True, enforce_sorted=False)
        output = pack_padded_sequence(generated_q.index_select(0, new_ids), lengths, batch_first=True, enforce_sorted=False) 
        loss_q = F.cross_entropy(output.data, target_question.data)
        loss_a = F.cross_entropy(generated_a, target_answer[:,0])
        # measure accuracy 
        acc1, acc5, acc10 = utils.accuracy(generated_a.data, target_answer.data, topk=(1, 5, 10))
        # nlg_metrics = calculate_nlg_score(generated_q.cpu().data, target_question, loader.dataset.wid_to_word)
        # bleu_score = calculate_bleu_score(generated_q.cpu().data, sample['question'], loader.dataset.wid_to_word)
        preds = []
        gts = []
        output = torch.argmax(generated_q, dim=2)
        for j in range(batch_size):
            new_question = output[j]
            # print(new_question.size())
            # break
            new_q = new_target_question[j]
            preds.append(translate_tokens(new_question, loader.dataset.wid_to_word))
            gts.append(translate_tokens(new_q, loader.dataset.wid_to_word))


        print('='*80)
        print('GROUND TRUTH')
        print(gts[:10])
        print('-'*80)
        print('PREDICTIONS')
        print(preds[:10])
        print('='*80)

        meters['acc1'].update(acc1.item(), n=batch_size)
        meters['acc5'].update(acc5.item(), n=batch_size)
        meters['acc10'].update(acc10.item(), n=batch_size)
        meters['loss_a'].update(loss_a.item(), n=batch_size)
        meters['loss_q'].update(loss_q.item(), n=batch_size)
        meters['dual_loss'].update(additional_loss.item(), n=batch_size)
        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        # meters['bleu_score'].update(bleu_score, n=batch_size)
        end = time.time()
        break    
    print('[Val]\tEpoch: [{0}]'
                  'Time {batch_time.avg:.3f}\t'
                  'A Loss: {loss_a.avg:.3f}, Q Loss: {loss_q.avg:.3f}, Dual Loss: {loss_d.avg:.3f}\t'
                  'Acc@1 {acc1.avg:.3f}\t'
                  'Acc@5 {acc5.avg:.3f}\t'
                  'Acc@10 {acc10.avg:.3f}\t'.format(
                   epoch, 
                   batch_time=meters['batch_time'], 
                   acc1=meters['acc1'], acc5=meters['acc5'], 
                   acc10=meters['acc10'], loss_a=meters['loss_a'], loss_q=meters['loss_q'], 
                   loss_d=meters['dual_loss']))

    logger.log_meters('val', n=epoch)
    return meters['acc1'].avg, meters['acc5'].avg, meters['acc10'].avg, meters['loss_q'].avg
# to generate single image result with beam search
def generate(resized_img, cnn_model, vqg_model, ):
    raise NotImplementedError


def evaluate(loader, model, logger, print_freq=10, sampling_num=5):
    model.eval()
    model.set_testing(True, sample_num=sampling_num)
    meters = logger.reset_meters('test')

    meters['Bleu_1']=logger2.AvgMeter()
    meters['Bleu_2']=logger2.AvgMeter()
    meters['Bleu_3']=logger2.AvgMeter()
    meters['Bleu_4']=logger2.AvgMeter()
    meters['METEOR']=logger2.AvgMeter()
    meters['ROUGE_L']=logger2.AvgMeter()
    meters['CIDEr']=logger2.AvgMeter()

    results = []
    end = time.time()
    blue_score_all = 0
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        # measure data loading time
        input_visual = Variable(sample['visual'].cuda( ), volatile=True )
        input_answer = Variable(sample['answer'].cuda( ), volatile=True)
        target_answer = sample['answer']
        input_question = Variable(sample['question'].cuda( ), volatile=True)
        # nlg_metrics = calculate_nlg_score(generated_q.cpu().data, sample['question'], loader.dataset.wid_to_word)

        output_answer, g_answers, g_answers_score, generated_q = model(input_visual, input_question, input_answer)
        # bleu_score = calculate_bleu_score(generated_q.cpu().data, sample['question'], loader.dataset.wid_to_word)
        acc1, acc5, acc10 = utils.accuracy(output_answer.cpu().data, target_answer, topk=(1, 5, 10))
        meters['acc1'].update(acc1.item(), n=batch_size)
        meters['acc5'].update(acc5.item(), n=batch_size)
        meters['acc10'].update(acc10.item(), n=batch_size)
        meters['bleu_score'].update(bleu_score, n=batch_size)
        meters['Bleu_1'].update(nlg_metrics['Bleu_1'], n=batch_size)
        meters['Bleu_2'].update(nlg_metrics['Bleu_2'], n=batch_size)
        meters['Bleu_3'].update(nlg_metrics['Bleu_3'], n=batch_size)
        meters['Bleu_4'].update(nlg_metrics['Bleu_4'], n=batch_size)
        meters['METEOR'].update(nlg_metrics['METEOR'], n=batch_size)
        meters['ROUGE_L'].update(nlg_metrics['ROUGE_L'], n=batch_size)
        meters['CIDEr'].update(nlg_metrics['CIDEr'], n=batch_size)

        g_answers = g_answers.cpu().data
        g_answers_score = g_answers_score.cpu().data

        for j in range(batch_size):
            new_question = generated_q.cpu().data[j].tolist()
            new_answer = g_answers[j]
            new_answer_score = g_answers_score[j]
            sampled_aqa = [[new_question, new_answer, new_answer_score],]
            num_result = {  'gt_question': sample['question'][j][1:].tolist(), #sample['question'][j].numpy(),
                            'gt_answer': sample['answer'][j],
                            'augmented_qa': sampled_aqa,}
            readable_result = {  
                            'gt_question': translate_tokens(sample['question'][j][1:], loader.dataset.wid_to_word), 
                            'gt_answer': loader.dataset.aid_to_ans[sample['answer'][j]], 
                            'augmented_qa': [ [
                                        translate_tokens(item[0], loader.dataset.wid_to_word), # translate question
                                        loader.dataset.aid_to_ans[item[1]], # translate answer
                                        ] for item in sampled_aqa],}
            results.append({'image': sample['image'][j], 
                            'numeric_result': num_result, 
                            'readable_result': readable_result}, )
        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

    print('* [Evaluation] Result: Acc@1:{acc1.avg:.3f}\t'
          'Acc@5:{acc5.avg:.3f}\tAcc@10:{acc10.avg:.3f}\t'
          'Time: {batch_time.avg:.3f}\t'
          'Bleu_1: {Bleu_1.avg:.5f}\t'
          'Bleu_2: {Bleu_2.avg:.5f}\t'
          'Bleu_3: {Bleu_3.avg:.5f}\t'
          'Bleu_4: {Bleu_4.avg:.5f}\t'
          'METEOR: {METEOR.avg:.5f}\t'
          'ROUGE_L: {ROUGE_L.avg:.5f}\t '
          'CIDEr: {CIDEr.avg:.5f}\t'
          'BLEU: {bleu_score.avg:.5f}'.format(
          acc1=meters['acc1'], acc5=meters['acc5'], acc10=meters['acc10'], 
          batch_time=meters['batch_time'], 
          Bleu_1=meters['Bleu_1'], 
          Bleu_2=meters['Bleu_2'], 
          Bleu_3=meters['Bleu_3'], 
          Bleu_4=meters['Bleu_4'], 
          METEOR=meters['METEOR'], 
          ROUGE_L=meters['ROUGE_L'], 
          CIDEr =meters['CIDEr'], 
          bleu_score=meters['bleu_score']))

    model.set_testing(False)
    return results
