# Copyright (C) 2021 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
#
# Author: Cristian Cioflan, ETH (cioflanc@iis.ee.ethz.ch)


import torch
STEP_DO_QAT_TRAIN = False       
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from torch.autograd import Variable
from utils import npy_to_txt


class DSCNN(torch.nn.Module):
    def __init__(self, use_bias=False, total_speakers = 3000):
        super(DSCNN, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.pad1  = nn.ConstantPad2d((1, 1, 5, 5), value=0.0)
        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (10, 4), stride = (2, 2), bias = use_bias)
        self.bn1   = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()

        self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
        self.bn2   = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn3   = torch.nn.BatchNorm2d(64)
        self.relu3 = torch.nn.ReLU()

        self.pad4  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv4 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
        self.bn4   = torch.nn.BatchNorm2d(64)
        self.relu4 = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn5   = torch.nn.BatchNorm2d(64)
        self.relu5 = torch.nn.ReLU()

        self.pad6  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv6 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
        self.bn6   = torch.nn.BatchNorm2d(64)
        self.relu6 = torch.nn.ReLU()
        self.conv7 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn7   = torch.nn.BatchNorm2d(64)
        self.relu7 = torch.nn.ReLU()

        self.pad8  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv8 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
        self.bn8   = torch.nn.BatchNorm2d(64)
        self.relu8 = torch.nn.ReLU()
        self.conv9 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn9   = torch.nn.BatchNorm2d(64)
        self.relu9 = torch.nn.ReLU()
        #self.bn10   = torch.nn.BatchNorm2d(64)

        #self.emb_norm = nn.InstanceNorm2d(num_features = 1, affine=False, track_running_stats=False)

        self.avg   = torch.nn.AvgPool2d(kernel_size=(25, 5), stride=1)
        

        self.emb_vec = []

        something = torch.ones(64, 1, 1, requires_grad=True)
        something_else = torch.ones(64, 1, 1, requires_grad=False)

        for i in range(total_speakers):
            self.emb_vec.append(something.to(torch.device('cuda')))

        self.emb_vec[2999] = something_else.to(torch.device('cuda'))

            
        self.fc1   = torch.nn.Linear(64, 12, bias=use_bias)
        self.dequant = torch.ao.quantization.DeQuantStub()
        # self.soft  = torch.nn.Softmax(dim=1)
        # self.soft = F.log_softmax(x, dim=1)


        # CONV2D replacing Block1 for evaluation purposes
        # self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        # self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 1, bias = use_bias)
        # self.bn2   = torch.nn.BatchNorm2d(64)
        # self.relu2 = torch.nn.ReLU()
    def forward(self, x, embeddings = 0):
        if getattr(self, "save", False):  # Use an attribute to control the saving behavior
            if STEP_DO_QAT_TRAIN:
                x = self.quant(x)
            x = self.pad1(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            npy_to_txt(0, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.pad2(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            npy_to_txt(1, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            # Repeat for other layers...
            x = self.pad8(x)
            x = self.conv8(x)
            x = self.bn8(x)
            x = self.relu8(x)
            npy_to_txt(7, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.conv9(x)
            x = self.bn9(x)
            x = self.relu9(x)
            npy_to_txt(8, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))
            x = self.avg(x)
            npy_to_txt(9, x.int().cpu().detach().numpy())
    
            x = self.avg(x)
            #selected_emb_vec = torch.stack([self.emb_vec[idx] for idx in embeddings])
            #x = selected_emb_vec * x
            #print(x[0,0,0,0])
            #print(embeddings[0,0,0,0])
            #x = embeddings * x
            #print(x[0,0,0,0])
            
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            
            if STEP_DO_QAT_TRAIN:
                x = self.dequant(x)

            npy_to_txt(10, x.int().cpu().detach().numpy())
        else:
            if STEP_DO_QAT_TRAIN:
                x = self.quant(x)
            x = self.pad1(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)

            x = self.pad2(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)

            x = self.pad4(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.relu5(x)

            x = self.pad6(x)
            x = self.conv6(x)
            x = self.bn6(x)
            x = self.relu6(x)
            x = self.conv7(x)
            x = self.bn7(x)
            x = self.relu7(x)

            x = self.pad8(x)
            x = self.conv8(x)
            x = self.bn8(x)
            x = self.relu8(x)
            x = self.conv9(x)
            x = self.bn9(x)
            x = self.relu9(x)
            
            x = self.avg(x)
            #selected_emb_vec = torch.stack([self.emb_vec[idx] for idx in embeddings])
            #x = selected_emb_vec * x
            #print(x[0,0,0,0])
            #print(embeddings[0,0,0,0])
            #x = embeddings * x
            #print(x[0,0,0,0])
            
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            if STEP_DO_QAT_TRAIN:
                print("awfjoiawjf")
                x = self.dequant(x)

        return x

class DSCNN_fusable(torch.nn.Module):
    def __init__(self, use_bias=False, total_speakers = 3000):
        super(DSCNN_fusable, self).__init__()
        
        self.quant = torch.ao.quantization.QuantStub()
        
        self.pad1  = nn.ConstantPad2d((1, 1, 5, 5), value=0.0)
        self.ConvBNReLU1 = torch.nn.Sequential (
            torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (10, 4), stride = (2, 2), bias = use_bias),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )   
        
        self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.ConvBNReLU2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.ConvBNReLU3 = torch.nn.Sequential (
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        
        self.pad4  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.ConvBNReLU4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.ConvBNReLU5 = torch.nn.Sequential (
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        
        self.pad6  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.ConvBNReLU6 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.ConvBNReLU7 = torch.nn.Sequential (
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        
        self.pad8  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.ConvBNReLU8 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.ConvBNReLU9 = torch.nn.Sequential (
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        #self.bn10   = torch.nn.BatchNorm2d(64)

        #self.emb_norm = nn.InstanceNorm2d(num_features = 1, affine=False, track_running_stats=False)

        self.avg   = torch.nn.AvgPool2d(kernel_size=(25, 5), stride=1)
        

        self.emb_vec = []

        something = torch.ones(64, 1, 1, requires_grad=True)
        something_else = torch.ones(64, 1, 1, requires_grad=False)

        for i in range(total_speakers):
            self.emb_vec.append(something.to(torch.device('cuda')))

        self.emb_vec[2999] = something_else.to(torch.device('cuda'))

            
        self.fc1   = torch.nn.Linear(64, 12, bias=use_bias)
        self.dequant = torch.ao.quantization.DeQuantStub()
        # self.soft  = torch.nn.Softmax(dim=1)
        # self.soft = F.log_softmax(x, dim=1)


        # CONV2D replacing Block1 for evaluation purposes
        # self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        # self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 1, bias = use_bias)
        # self.bn2   = torch.nn.BatchNorm2d(64)
        # self.relu2 = torch.nn.ReLU()
    def forward(self, x, embeddings = 0):
        if getattr(self, "save", False):  # Use an attribute to control the saving behavior
            x = self.quant(x)
            x = self.pad1(x)
            x = self.ConvBNReLU1(x)
            npy_to_txt(0, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.pad2(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            npy_to_txt(1, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            # Repeat for other layers...
            x = self.pad8(x)
            x = self.ConvBNReLU8(x)
            npy_to_txt(7, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.ConvBNReLU9(x)
            npy_to_txt(8, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))
            x = self.avg(x)
            npy_to_txt(9, x.int().cpu().detach().numpy())
    
            x = self.avg(x)
            #selected_emb_vec = torch.stack([self.emb_vec[idx] for idx in embeddings])
            #x = selected_emb_vec * x
            #print(x[0,0,0,0])
            #print(embeddings[0,0,0,0])
            #x = embeddings * x
            #print(x[0,0,0,0])
            
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            
            x = self.dequant(x)

            npy_to_txt(10, x.int().cpu().detach().numpy())
        else:
            x = self.quant(x)
            
            x = self.pad1(x)
            x = self.ConvBNReLU1(x)
            
            x = self.pad2(x)
            x = self.ConvBNReLU2(x)
            x = self.ConvBNReLU3(x)
            
            x = self.pad4(x)
            x = self.ConvBNReLU4(x)
            x = self.ConvBNReLU5(x)
            
            x = self.pad6(x)
            x = self.ConvBNReLU6(x)
            x = self.ConvBNReLU7(x)
            
            x = self.pad8(x)
            x = self.ConvBNReLU8(x)
            x = self.ConvBNReLU9(x)
            
            x = self.avg(x)
            #selected_emb_vec = torch.stack([self.emb_vec[idx] for idx in embeddings])
            #x = selected_emb_vec * x
            #print(x[0,0,0,0])
            #print(embeddings[0,0,0,0])
            #x = embeddings * x
            #print(x[0,0,0,0])
            
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.dequant(x)

        return x
