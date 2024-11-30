import torch
import torchvision.models as models

def load_alexnet(num_classes, pretrained=False):
    model = models.alexnet(pretrained=pretrained)
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_features, num_classes)
    return model

def load_vgg16(num_classes, pretrained=False):
    model = models.vgg16(pretrained=pretrained)
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_features, num_classes)
    return model

def load_resnet18(num_classes, pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    return model

def load_inception_v3(num_classes, pretrained=False):
    model = models.inception_v3(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    return model

def load_densenet121(num_classes, pretrained=False):
    model = models.densenet121(pretrained=pretrained)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, num_classes)
    return model

def load_xception(num_classes, pretrained=False):
    model = models.xception(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    return model
