import torch
import torch.nn as nn
import torch.functional as F
import torchvision.models as models
import ConvLSTM

class AnomalyDetectionHead(nn.Module):
    def __init__(self):
        super(AnomalyDetectionHead, self).__init__()
        #self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc(x)
        output = F.softmax(x, dim=1)
        return output

    def initWeights(self):
        torch.nn.init.xavier_uniform(self.fc.weight)


class CountingHead(nn.Module):
    def __init__(self, channels=256, layerCount=4):
        super(CountingHead, self).__init__()
        countingHeadLayerList = []
        for i in range(layerCount):
            countingHeadLayerList.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            countingHeadLayerList.append(nn.BatchNorm2d(channels))
            countingHeadLayerList.append(nn.ReLU())
        countingHeadLayerList.append(nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1))

        self.add_module('countingModule', nn.Sequential(*countingHeadLayerList))

    def forward(self, x):
        output = self.countingModule(x)
        return output

    def initWeights(self):
        torch.nn.init.xavier_uniform(self.countingModule.weight)


class CrowdMultiPrediction(nn.Module):
    def __init__(self, pretrainedBackbone=False):
        self.backbone = models.resnet50(pretrained=pretrainedBackbone)  
        self.countingHead = CountingHead()
        self.anomalyDetectionHead = AnomalyDetectionHead()
        self.countingHeadLoss = nn.MSELoss()
        
    def forward(self, x):
        xf = self.backbone(x)
        count = self.countingHead(xf)
        anomaly = self.AnomalyDetectionHead(xf)
        return count, anomaly

    def initWeights(self):
        self.countingHead.initWeights()
        self.anomalyDetectionHead.initWeights()

    def loss(self, result, groundTruth):
        # for training
        behavClsResult, densLevelClsResult, countResult, heatmapResult = result
        behavClsGt, densLevelClsGt, countGt, heatmapGt = groundTruth
        behavClsLoss = self.behavClsHeadLoss(behavClsResult, behavClsGt)
        densLevelClsLoss = self.densLevClsHeadLoss(densLevelClsResult, densLevelClsGt)
        countLoss = self.countingHeadLoss(countResult, countGt)
        heatmapLoss = self.heatMapHeadLoss(heatmapResult, heatmapGt)
        loss = behavClsLoss + densLevelClsLoss + countLoss + heatmapLoss
        return loss


