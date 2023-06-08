import os, sys
import numpy as np
import torch
import models_baseline as models
#import models
import data_loader
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

sys.path.insert(0, './WSI')
import datasets
import wandb


file_ = open("./logs/logs.txt", 'w+') 
cuda = torch.cuda.is_available()

test_dset = datasets.Infer_Dataset(DataSet="HAEMEK" , tiles_per_iter = 50)
target_test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False,
                                                 num_workers=0, drop_last=False, pin_memory=True)
len_target_dataset = len(target_test_loader.dataset)
len_target_test_loader = len(target_test_loader)


nclasses = 2 #len(source_loader.dataset.classes)

model = models.RevGrad(num_classes = nclasses)
if torch.cuda.is_available():
    model.to('cuda')
model.load_state_dict(torch.load("/home/edenko/UDA/our_rev_grads/models/best_balnced_acc_baseline_carmel_model_2_7.pkl"))
model.eval()
test_loss = 0
correct = 0

target_test_iter = iter(target_test_loader)

correct_domain = 0
total_test, correct_pos_test, correct_neg_test = 0, 0, 0
total_pos_test, total_neg_test = 0, 0
true_labels_test, scores_test = np.zeros(0), np.zeros(0)
label_output_tot = None
correct_labeling_test, loss_test = 0, 0
total_pos_pred, total_neg_pred = 0,0
i = 1
j = 1  
slide_name=''  
while i <= len_target_test_loader: #len_target_test_loader:    
#for data, target in target_test_loader:
    # for the source domain batch
    
    next_target = next(target_test_iter)
    
    data = next_target['Data']
    target = next_target['Label'] 
    
    if cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    data= data.squeeze()
    #_, _, label_output = model(data, alpha = 0)
    _, label_output = model(data, alpha = 0)
    
    if slide_name == next_target['Slide Filename'][0]:
       label_output_tot = np.concatenate((label_output_tot, label_output.cpu().detach().numpy()), axis=0)

    else:
       
        #print(str(next_target['Slide Filename'][0]))
        if  slide_name!='':
            mean = label_output_tot.mean(axis=0)
            outputs = torch.nn.functional.softmax(torch.from_numpy(mean))
            target = target.squeeze()
            predicted =  outputs.max(0)[1].item()
            scores_test = np.concatenate((scores_test, outputs[1].cpu().detach().numpy().reshape(1)))
            true_labels_test = np.concatenate((true_labels_test, target.cpu().detach().numpy().reshape(1)))
            total_test += 1
            correct_labeling_test += (predicted==target)
            total_pos_test += (target==1)
            total_neg_test += (target==0)
            correct_pos_test += (predicted==1 and target==1)
            correct_neg_test += (predicted==0 and target==0)
            total_pos_pred += (predicted==1)
            total_neg_pred += (predicted==0)
        slide_name = next_target['Slide Filename'][0]
        label_output_tot= label_output.cpu().detach().numpy()

    i = i + 1
    
acc = 100 * float(correct_labeling_test) / total_test
balanced_acc = 100 * (correct_pos_test / (total_pos_test + 1e-7) + correct_neg_test / (total_neg_test + 1e-7)) / 2

fpr, tpr, _ = roc_curve(true_labels_test, scores_test)
roc_auc = auc(fpr, tpr)


#F1
precision = correct_pos_test/total_pos_pred
recall = correct_pos_test/(correct_pos_test + total_neg_pred - correct_neg_test)
f1= 2*(precision*recall)/(precision+recall) 

file_.write("accuracy: " + str(acc))
file_.write("\nbalanced_accuracy: " + str(balanced_acc))
file_.write("\nAUC: " + str(roc_auc))
file_.write("\nprecision: " + str(roc_auc))
file_.write("\nrecall: " + str(recall))
file_.write("\nF1: " + str(f1))

file_.close()

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('./logs/ROC_curve_best_balnced_acc_model_07_02_baseline_carmel_haemek_test.png')
plt.show()

