from utils import*
from preprocessing import feat_extr
import torch
import os
import torch.nn as nn
data_path = ''
working_path = ''

validation = False

# =============================================================================
# Extraction of feature vectors
# =============================================================================

#load the pretrained model
path_model = ''
model = torch.load(path_model)
model.classifier = nn.Identity()
model.eval()
os.mkdir(working_path+'feat_vect_vgg16')
for patient in os.listdir(data_path + 'train/'):
    os.mkdir(working_path+'feat_vect_vgg16/'+patient)
    for scan in os.listdir(data_path + 'train/' + patient):
        torch.save(feat_extr(model,data_path + 'train/' + patient + '/' + scan),working_path+'/feat_vect_vgg16/'+patient + '/'+scan[:-4]+'.pt')

# =============================================================================
# Prepare the data for training
# =============================================================================
X_train, Y_train = prepare_train_data(data_path, working_path, validation)
X_test = prepare_test_data(data_path, working_path, validation)
# =============================================================================
# Train model
# =============================================================================
net = RegressionNetwork()
num_epochs = 1
batch_size = 64
learning_rate = 0.0075
train_linear_model(data_path, working_path, net, X_train, Y_train,
                       num_epochs, batch_size, learning_rate)
# =============================================================================
# Predict
# =============================================================================
coef_confi = (80,4)
predict_linear_model(X_test, net, working_path, coef_confi)
