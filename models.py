import torchvision.models as models
from torch import nn

class model_feature(nn.Module):
   def __init__(self, hidden_dim):
      super(model_feature, self).__init__()
      self.resnet50 = models.resnet50(pretrained=True)
      # for param in self.resnet50.parameters():
      #    param.requires_grad = False
      self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, hidden_dim)

   def logits(self, input):
      x = self.resnet50(input)
      return x

   def forward(self, input):
      input = input.permute(0,3,1,2).cuda()
      x = self.logits(input)
      return x

class model_task(nn.Module):
   def __init__(self, model_feature, hidden_dim, num_classes):
      super(model_task, self).__init__()
      self.num_classes = num_classes
      self.linear1 = nn.Linear(hidden_dim, num_classes)
      self.model_feature = model_feature
      self.dropout = nn.Dropout(0.5)
      self.relu = nn.ReLU(True)

   def logits(self, input):
      x = self.model_feature(input)
      x = self.relu(x)
      #x = self.dropout(x)
      x = self.linear1(x)
      return x

   def forward(self, input):
      x = self.logits(input)
      return x

class model_regularizer(nn.Module):
   def __init__(self, hidden_dim, num_classes):
      super(model_regularizer, self).__init__()
      self.num_classes = num_classes
      self.linear1 = nn.Linear(hidden_dim * num_classes, 1, bias=False)

   def logits(self, input):
      x = self.linear1(input)
      return x

   def forward(self, input):
      input = input.cuda()
      x = self.logits(input)
      return x