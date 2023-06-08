from __future__ import print_function
import os, sys
import math
import argparse
import numpy as np
import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import model_zoo
from sklearn.metrics import roc_curve, auc
# import tensorflow as tf
#from torch.utils.tensorboard import SummaryWriter
import wandb



import ../include/data_loader
import models
sys.path.insert(0, './WSI')
import ../include/datasets

###################################

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', help = 'root to the data')
parser.add_argument('--source', type = str, default = 'usps', help = 'the source domain')
parser.add_argument('--target', type = str, default = 'mnist', help = 'the target domain')
parser.add_argument('--model_dir', type = str, default = './models/', help = 'the path to save models')
parser.add_argument('--batch_size', type = int, default = 100, help = 'the size of mini-batch')
parser.add_argument('--epochs', type = int, default = 100, help = 'the number of epochs')
parser.add_argument('--lr', type = float, default = 0.001, help = 'the initial learning rate')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'the momentum of gradient')
parser.add_argument('--l2_decay', type = float, default = 5e-4, help = 'the l2 decay used in training')
parser.add_argument('--seed', type = int, default = 100, help = 'the manual seed')
parser.add_argument('--log_interval', type = int, default = 50, help = 'the interval of print')
parser.add_argument('--gpu_id', type = str, default = '0', help = 'the gpu device id')

opt = parser.parse_args()
print (opt)

###################################

# Training settings
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(opt.seed)

# Dataloader

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# source_loader = data_loader.load_training(opt.root_path, opt.source, opt.batch_size, kwargs)
# target_train_loader = data_loader.load_training(opt.root_path, opt.target, opt.batch_size, kwargs)
# target_test_loader = data_loader.load_testing(opt.root_path, opt.target, opt.batch_size, kwargs)
train_source_dset = datasets.WSI_REGdataset(DataSet='CARMEL')
train_target_dset = datasets.WSI_REGdataset(DataSet='HAEMEK')                                     
test_dset = datasets.WSI_REGdataset(DataSet='HAEMEK', train=False)

opt.batch_size = 50
opt.lr = 0.001
opt.epochs = 400
                                   
source_loader = torch.utils.data.DataLoader(train_source_dset, batch_size=opt.batch_size, shuffle=True,
                                                 num_workers=2, drop_last=True, pin_memory=True)
target_train_loader = torch.utils.data.DataLoader(train_target_dset, batch_size=opt.batch_size, shuffle=True,
                                                 num_workers=2, drop_last=True, pin_memory=True)
target_test_loader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batch_size, shuffle=False,
                                                 num_workers=2, drop_last=False, pin_memory=True)


len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)
len_target_test_loader = len(target_test_loader)
nclasses = 2 #len(source_loader.dataset.classes)

###################################

# For every epoch training
def train(epoch, model):

    optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_train_loader)
    dlabel_src = Variable(torch.ones(opt.batch_size).long().cuda())
    dlabel_tgt = Variable(torch.zeros(opt.batch_size).long().cuda())
    
    #tensorboard
    train_total_label_loss = 0
    train_total_domain_loss = 0
    # epoch_label_loss_avg = tf.keras.metrics.Mean()
    # epoch_domain_loss_avg = tf.keras.metrics.Mean()
    # #epoch_label_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    # #epoch_domain_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    i = 1
    while i <= len_source_loader:
        model.train()

        # the parameter for reversing gradients
        p = float(i + epoch * len_source_loader) / opt.epochs / len_source_loader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # for the source domain batch
        next_source = next(data_source_iter)
        source_data = next_source['Data']
        source_label = next_source['Target']
        
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        _, clabel_src, dlabel_pred_src = model(source_data, alpha = alpha)
        label_loss = loss_class(clabel_src, source_label.ravel())
        domain_loss_src = loss_domain(dlabel_pred_src, dlabel_src)

        # for the target domain batch
        next_target = next(data_target_iter)
        target_data = next_target['Data']
        target_label = next_target['Target']
        if i % len_target_loader == 0:
            data_target_iter = iter(target_train_loader)
        if cuda:
            target_data, target_label = target_data.cuda(), target_label.cuda()
        target_data = Variable(target_data)

        _, clabel_tgt, dlabel_pred_tgt = model(target_data, alpha = alpha)
        domain_loss_tgt = loss_domain(dlabel_pred_tgt, dlabel_tgt)

        domain_loss_total = domain_loss_src + domain_loss_tgt
        loss_total = label_loss + domain_loss_total

        optimizer.zero_grad()
        # label_loss.backward()
        loss_total.backward()
        optimizer.step()

        if i % opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlabel_Loss: {:.6f}\tdomain_Loss: {:.6f}'.format(
                epoch, i * len(source_data), len_source_dataset,
                100. * i / len_source_loader, label_loss.item(), domain_loss_total.item()))
        i = i + 1
        
        #tensorboard
        train_total_domain_loss += domain_loss_total
        train_total_label_loss += label_loss
        # epoch_label_loss_avg(loss_total)
        # epoch_domain_loss_avg(domain_loss_total)
        # #epoch_label_accuracy(source_label, model(x))
        # #epoch_domain_accuracy(y, model(x))

    # with summary_writer.as_default():
        # tf.summary.scalar('epoch_label_loss_avg', epoch_label_loss_avg.result(), step=optimizer.iterations)
        # tf.summary.scalar('epoch_domain_loss_avg', epoch_domain_loss_avg.result(), step=optimizer.iterations)
        # #tf.summary.scalar('epoch_label_accuracy', epoch_accuracy.result(), step=optimizer.iterations)
        # #tf.summary.scalar('epoch_domain_accuracy', epoch_accuracy.result(), step=optimizer.iterations)
    #summary_writer.add_scalar('label_loss', train_total_label_loss / len_source_loader, epoch)
    #summary_writer.add_scalar('domain_loss', train_total_domain_loss / len_source_loader, epoch)
    

# For every epoch evaluation
def test(epoch, model):
    model.eval()
    test_loss = 0
    correct = 0
    #added for f1
    true_positive = 0
    true_negative = 0
    positive = 0
    negative = 0
    #--------
    
    target_test_iter = iter(target_test_loader)
    true_labels_test, scores_test = np.zeros(0), np.zeros(0) #metrics
    total_pos_test, total_neg_test = 0,0
    correct_domain = 0
    i = 1    
    while i <= len_target_test_loader:    
    #for data, target in target_test_loader:
        # for the source domain batch
        next_target = next(target_test_iter)
        data = next_target['Data']
        target = next_target['Target'] 
        
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        _, s_output, t_output = model(data, alpha = 0)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target.ravel(), size_average=False).item()
        pred = s_output.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        #added for f1
        target = target.squeeze()
        
        true_positive += (pred[pred.eq(target.data.view_as(pred)).cpu()]==1).sum()
        true_negative += (pred[pred.eq(target.data.view_as(pred)).cpu()]==0).sum()
        positive += (pred==1).sum()
        negative += (pred==0).sum()
        outputs = torch.nn.functional.softmax(s_output, dim=1)
        total_pos_test += target.eq(1).sum().item()
        total_neg_test += target.eq(0).sum().item()
        scores_test = np.concatenate((scores_test, outputs[:, 1].cpu().detach().numpy()))
        true_labels_test = np.concatenate((true_labels_test, target.cpu().detach().numpy()))
        #--------
        # :le:
        # test_loss += F.nll_loss(F.log_softmax(t_output, dim = 1), target, size_average=False).item()
        pred_domain = t_output.max(1)[1]
        correct_domain += len(pred_domain) - pred_domain.sum().item()
        i = i + 1

    test_loss /= len_target_dataset

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Domain accuracy: {}/{} ({:.2f}%)\n'.format(
        opt.target, test_loss, correct, len_target_dataset,
        100. * correct / len_target_dataset, correct_domain, len_target_dataset, 100. * correct_domain / len_target_dataset))
    #added for f1
    precision = true_positive/positive
    recall = true_positive/(true_positive + negative- true_negative)
    f1= 2*(precision*recall)/(precision+recall)
    balanced_acc = 100 * (true_positive / (total_pos_test + 1e-7) + true_negative / (total_neg_test + 1e-7)) / 2
    fpr, tpr, _ = roc_curve(true_labels_test, scores_test)
    roc_auc = auc(fpr, tpr)
    #-------------

    
    return balanced_acc , roc_auc


if __name__ == '__main__':

    model = models.RevGrad(num_classes = nclasses)
    print (model)

    max_balnced_acc = 0
    max_auc = 0
    if cuda:
        model.cuda()
    
    #tensorboard
    #log_dir = os.path.join("//home//lior.grauer//UDA//Mixup_RevGrad//logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #summary_writer = SummaryWriter(log_dir=log_dir) #tf.summary.create_file_writer(logdir=log_dir)
    
    # start training
    for epoch in range(1, opt.epochs + 1):
        train(epoch, model)
        # test for every epoch
        balnced_acc, roc_auc = test(epoch, model)
        if balnced_acc > max_balnced_acc:
            max_balnced_acc = balnced_acc
            if not os.path.exists(opt.model_dir):
                os.mkdir(opt.model_dir)
            torch.save(model.state_dict(), os.path.join(opt.model_dir, 'model_name_best_ballanced_acc.pkl'))
        if roc_auc > max_auc:
            max_auc = roc_auc
            if not os.path.exists(opt.model_dir):
                os.mkdir(opt.model_dir)
            torch.save(model.state_dict(), os.path.join(opt.model_dir, 'model_name_best_auc.pkl'))

        print('source: {} to target: {} max balnced acc: {} max auc{: .2f}%\n'.format(
              'Carmel', 'Heemek',max_balnced_acc, max_auc ))
              
    #summary_writer.close()