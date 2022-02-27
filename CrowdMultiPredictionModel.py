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
        x = torch.flatten(x)
        x = self.fc(x)
        output = F.relu(x)
        return output


class CrowdMultiPrediction(nn.Module):
    def __init__(self, pretrainedBackbone=False):
        super(CrowdMultiPrediction, self).__init__()
        #resnet50 = models.resnet50(pretrained=pretrainedBackbone)  
        resnet50 = models.resnet34(pretrained=pretrainedBackbone)  
        self.backbone = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4
        )
        self.backbone_out_channels = 512
        self.backbone_out_tensor_height = 45
        self.backbone_out_tensor_width = 80
        self.lstm_encoder = ConvLSTMEncoder(input_dim=self.backbone_out_channels, 
                                            hidden_dim=self.backbone_out_channels, 
                                            kernel_size=(3, 3), bias=True)
        self.countingHead = CountingHead(channels=self.backbone_out_channels, layerCount=4)
        self.anomalyClsHead = AnomalyClassificationHead(channels=self.backbone_out_channels, layerCount=4)
        self.countingHeadLoss = nn.MSELoss()
        self.anomalyClsHeadLoss = nn.BCELoss()
        
    def forward(self, frames, device):
        (h1_t, c1_t), (h2_t, c2_t) = self.lstm_encoder.init_hidden(batch_size=1,
                                                                   image_size=(self.backbone_out_tensor_height, 
                                                                   self.backbone_out_tensor_width))
        count_outs = []
        for frame in frames:
            frame = frame.to(device)
            xf = self.backbone(frame)
            count_outs.append(self.countingHead(xf))
            h1_t, c1_t, h2_t, c2_t = self.lstm_encoder(input_tensor=xf, cur_state=[h1_t, c1_t, h2_t, c2_t])
        anomaly_out = self.anomalyClsHead(h2_t)
        return count_outs, anomaly_out

    def loss(self, count_outs, anomaly_out, count_gts, anomaly_gt, device):
        count_losses = []
        for i in range(len(count_outs)):
            count_gt = count_gts[i].float().to(device)
            count_losses.append(self.countingHeadLoss(count_outs[i], count_gt))
        count_loss = torch.mean(torch.stack(count_losses))
        anomaly_gt = anomaly_gt.float().to(device)
        anomaly_loss = self.anomalyClsHeadLoss(anomaly_out, anomaly_gt)
        return count_loss, anomaly_loss
