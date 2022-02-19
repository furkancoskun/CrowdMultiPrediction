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
    def __init__(self):
        super(CountingHead, self).__init__()
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc(x)
        output = F.relu(x)
        return output

    def initWeights(self):
        torch.nn.init.xavier_uniform(self.fc.weight)

class CrowdMultiPrediction(nn.Module):
    def __init__(self, pretrainedBackbone=False):
        resnet50_backbone = models.resnet50(pretrained=pretrainedBackbone)  
        self.countingHead = CountingHead()
        self.anomalyDetectionHead = AnomalyDetectionHead()

    def forward(self, x):
        x = self.backbone(x)
        heatmap = self.countingHead(x)
        x = self.averagePool(x)
        x = x.view(64*154*84)
        x = self.fc32(x)
        behavCls = self.behavClsHead(x)
        densLevelCls = self.densLevClsHead(x)
        count = self.countingHead(x)
        return behavCls, densLevelCls, count, heatmap

    def initWeights(self):
        torch.nn.init.xavier_uniform(self.fc32.weight)
        self.behavClsHead.initWeights()
        self.densLevClsHead.initWeights()
        self.countingHead.initWeights()
        self.heatMapHead.initWeights()

    def calculateLoss(self, result, groundTruth):
        behavClsResult, densLevelClsResult, countResult, heatmapResult = result
        behavClsGt, densLevelClsGt, countGt, heatmapGt = groundTruth
        behavClsLoss = self.behavClsHeadLoss(behavClsResult, behavClsGt)
        densLevelClsLoss = self.densLevClsHeadLoss(densLevelClsResult, densLevelClsGt)
        countLoss = self.countingHeadLoss(countResult, countGt)
        heatmapLoss = self.heatMapHeadLoss(heatmapResult, heatmapGt)
        loss = behavClsLoss + densLevelClsLoss + countLoss + heatmapLoss
        return loss


