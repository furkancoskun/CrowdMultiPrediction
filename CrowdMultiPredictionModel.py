import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ConvLSTM import ConvLSTMEncoder

class AnomalyClassificationHead(nn.Module):
    def __init__(self, channels=2048, layerCount=3):
        super(AnomalyClassificationHead, self).__init__()
        anomalyClsHeadLayerList = []
        for i in range(layerCount):
            anomalyClsHeadLayerList.append(nn.Conv2d(channels, channels, 
                                                     kernel_size=3, stride=1, padding=1))
            anomalyClsHeadLayerList.append(nn.BatchNorm2d(channels))
            anomalyClsHeadLayerList.append(nn.ReLU())
        anomalyClsHeadLayerList.append(nn.AdaptiveAvgPool2d((1,1)))
        self.add_module('anomalyClsModule', nn.Sequential(*anomalyClsHeadLayerList))
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.anomalyClsModule(x)
        x = torch.flatten(x)
        x = self.fc(x)
        output = F.softmax(x, dim=0)
        return output


class CountingHead(nn.Module):
    def __init__(self, channels=2048, layerCount=3):
        super(CountingHead, self).__init__()
        countingHeadLayerList = []
        for i in range(layerCount):
            countingHeadLayerList.append(nn.Conv2d(channels, channels, 
                                                   kernel_size=3, stride=1, padding=1))
            countingHeadLayerList.append(nn.BatchNorm2d(channels))
            countingHeadLayerList.append(nn.ReLU())
        countingHeadLayerList.append(nn.AdaptiveAvgPool2d((1,1)))
        self.add_module('countingModule', nn.Sequential(*countingHeadLayerList))
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.countingModule(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.relu(x)
        return output


class CrowdCounting(nn.Module):
    def __init__(self, pretrainedBackbone=False):
        super(CrowdCounting, self).__init__()
        #resnet = models.resnet50(pretrained=pretrainedBackbone)  
        resnet = models.resnet34(pretrained=pretrainedBackbone)  
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.backbone_out_channels = 512 #2048 when resnet50 used
        self.countingHead = CountingHead(channels=self.backbone_out_channels, layerCount=4)
        self.countingHeadLoss = nn.MSELoss()
        
    def forward(self, frames):
        xf = self.backbone(frames)
        count_out = self.countingHead(xf)
        return count_out

    def loss(self, count_out, count_gt):
        count_loss = self.countingHeadLoss(count_out, count_gt)
        return count_loss

class CrowdMultiPrediction(nn.Module):
    def __init__(self, pretrainedBackbone=False):
        super(CrowdMultiPrediction, self).__init__()
        #resnet = models.resnet50(pretrained=pretrainedBackbone)  
        resnet = models.resnet34(pretrained=pretrainedBackbone)  
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.backbone_out_channels = 512 #2048 when resnet50 used
        self.backbone_out_tensor_height = 34 #45 when input height 1440
        self.backbone_out_tensor_width = 60 #80 when input height 2560
        self.lstm_encoder = ConvLSTMEncoder(input_dim=self.backbone_out_channels, 
                                            hidden_dim=self.backbone_out_channels, 
                                            kernel_size=(3, 3), bias=True)
        self.countingHead = CountingHead(channels=self.backbone_out_channels, layerCount=4)
        self.anomalyClsHead = AnomalyClassificationHead(channels=self.backbone_out_channels, layerCount=4)
        self.countingHeadLoss = nn.MSELoss()
        self.anomalyClsHeadLoss = nn.BCELoss()
        
    def forward(self, frames):
        xf = self.backbone(frames)
        count_out = self.countingHead(xf)
        (h1_t, c1_t), (h2_t, c2_t) = self.lstm_encoder.init_hidden(batch_size=1,
                                                                   image_size=(self.backbone_out_tensor_height, 
                                                                   self.backbone_out_tensor_width))
        for i in range(frames.shape[0]):
            h1_t, c1_t, h2_t, c2_t = self.lstm_encoder(input_tensor=xf[i].unsqueeze(dim=0), cur_state=[h1_t, c1_t, h2_t, c2_t])
        anomaly_out = self.anomalyClsHead(h2_t)
        return count_out, anomaly_out

    def loss(self, count_out, anomaly_out, count_gt, anomaly_gt):
        count_loss = self.countingHeadLoss(count_out, count_gt)
        anomaly_loss = self.anomalyClsHeadLoss(anomaly_out, anomaly_gt)
        return count_loss, anomaly_loss
