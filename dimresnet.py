import argparse
import torch
import torchvision
import time
import os
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule

class DimResnetTraining:
  def __init__(self, batchSize,lr,mom,wd,num_workers,drop_last,image_size):
    """Init training configuration"""
    #torch.manual_seed(123)
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    self.cuda = torch.cuda.is_available()
    # BatchSize
    self.batchSize = batchSize
    self.initData(num_workers,drop_last,image_size)
    self.initModel(lr,mom,wd)
  
  def initData(self,num_workers,drop_last,image_size) :
    # Data for train
    transform = torchvision.transforms.Compose([
     # Random resize - Data Augmentation
     torchvision.transforms.RandomResizedCrop(image_size),
     # Horizontal Flip - Data Augmentation
     torchvision.transforms.RandomHorizontalFlip(),
     # convert the PIL Image to a tensor
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
      std=(0.229, 0.224, 0.225))
     ])
    DSDIR = os.environ['DSDIR']
    trainData = torchvision.datasets.ImageNet(
     root=DSDIR+'/imagenet', transform=transform)
    # Put trainData in batch with shuffle
    self.trainLoader = torch.utils.data.DataLoader(dataset=trainData,
     batch_size=self.batchSize,shuffle=True,num_workers=num_workers,
     drop_last=drop_last)
    # Get data for test
    test_transform = torchvision.transforms.Compose([
     torchvision.transforms.Resize((256, 256)),
     torchvision.transforms.CenterCrop(224),
     torchvision.transforms.ToTensor(),   # convert the PIL Image to a tensor
     torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), 
      std=(0.229, 0.224, 0.225))])
    testData = torchvision.datasets.ImageNet(
     root=DSDIR+'/imagenet', split='val',transform=test_transform)
     #root=os.environ['SCRATCH']+'/imagenet', split='val',transform=test_transform)
    # Put testData in batch without shuffle
    self.testLoader = torch.utils.data.DataLoader(dataset=testData,
     batch_size=self.batchSize, shuffle=False, num_workers=num_workers)
  
  def initModel(self,lr,mom,wd) :
    # Model
    self.model = torchvision.models.resnet50()
    print('model: resnet50')
    print('number of parameters: {}'
     .format(sum([p.numel() for p in self.model.parameters()])))
    if self.cuda :
      gpu = torch.device("cuda")
      self.model.to(gpu,memory_format=torch.channels_last)
    #self.model = torch.compile(self.model)
    self.lossFn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=mom,
     weight_decay=wd)
    self.model = torch.compile(self.model)
    self.startEpoch = 0
    self.bestAccuracy = 0
    # Checkpoint file
    self.resumeFile = 'checkpoint.pth.tar'
    # Read checkpoint file if exist
    if os.path.isfile(self.resumeFile):
      loadt1 = time.perf_counter()
      print("=> loading checkpoint '{}' ...".format(self.resumeFile))
      if self.cuda:
        checkpoint = torch.load(self.resumeFile)
      else:
        # Load GPU model on CPU
        checkpoint = torch.load(self.resumeFile, weights_only=False,
         map_location=lambda storage, loc: storage)
      loadt2 = time.perf_counter()
      self.startEpoch = checkpoint['epoch']
      self.bestAccuracy = checkpoint['best_accuracy']
      self.model.load_state_dict(checkpoint['state_dict'])
      print("=> loaded checkpoint '{}' (trained for {} epochs & {} accuracy)"
       .format(self.resumeFile, checkpoint['epoch'],
       checkpoint['best_accuracy']))
      print(f"Load time :{loadt2 - loadt1:0.4f} seconds")
  
  def train(self,numEpochs,chkpt,prof):
    """Train for numEpochs"""
    profil =  profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
     schedule=schedule(wait=1, warmup=1, active=6, repeat=1),
     on_trace_ready=tensorboard_trace_handler('./profiler/'
     + os.environ['SLURM_JOB_NAME'] + '_' + os.environ['SLURM_JOBID']),
     profile_memory=True,
     record_shapes=False,
     with_stack=False,
     with_flops=False)
    if prof:
      profil.start()
    tic = time.perf_counter()
    for epoch in range(numEpochs) :
      self.trainOne(epoch,numEpochs,profil,prof)
      acc = self.evalOne()
      isBest = acc > self.bestAccuracy
      if isBest:
        self.bestAccuracy = acc
      if chkpt :
        self.saveCheckpoint({'epoch': self.startEpoch + epoch + 1,
         'state_dict': self.model.state_dict(),
         'best_accuracy': self.bestAccuracy},isBest)
    toc = time.perf_counter()
    print(f"Train time :{toc - tic:0.4f} seconds")
  
  def trainOne(self,epoch,numEpochs,profil,prof) :
    """Train one epoch"""
    traint1 = time.perf_counter()
    self.model.train()
    for i, (images, labels) in enumerate(self.trainLoader):
      if self.cuda:
        images = images.cuda(memory_format=torch.channels_last)
        labels = labels.cuda()
      # Gradient = 0
      self.optimizer.zero_grad()
      # Forward
      outputs = self.model(images)
      # Loss compute
      loss = self.lossFn(outputs,labels)
      #if self.cuda:
      #  loss.cpu()
      # Backward
      loss.backward()
      self.optimizer.step()
      if prof:
        profil.step()
    traint2 = time.perf_counter()
    print('Epoch: [%d/%d], Loss: %.4f, Epoch time: %f' 
     % (self.startEpoch+epoch+1, self.startEpoch+numEpochs,loss.item(),
     traint2-traint1))
  
  def evalOne(self) :
    """Evaluation"""
    evalt1 = time.perf_counter()
    self.model.eval()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(self.testLoader):
      if self.cuda:
        images = images.cuda(memory_format=torch.channels_last)
        labels = labels.cuda()
      outputs = self.model(images)
      loss = self.lossFn(outputs,labels)
      _, predicted = torch.max(outputs.data, 1)
      correct += (predicted == labels).sum().item()
      total += labels.size(0)
    acc = 100*(correct / total)
    evalt2 = time.perf_counter()
    print('=> Test set: Accuracy: {:.2f}%'.format(acc))
    print('Eval time: ', evalt2-evalt1)
    return acc
  
  def saveCheckpoint(self,state, isBest):
    """Save Chekpoint if isBest"""
    if isBest:
      print ("=> Saving a new best")
      savet1 = time.perf_counter()
      torch.save(state, "{}-{}-{}".format(
       state['epoch'],state['best_accuracy'],self.resumeFile))
      savet2 = time.perf_counter()
      print(f"Save time :{savet2 - savet1:0.4f} seconds")
    else:
      print ("=> Validation Accuracy did not improve")
  

if __name__ == '__main__' :
  parser = argparse.ArgumentParser()
  parser.add_argument('-b','--batch-size', default=128, type=int,
   help='batch size per GPU')
  parser.add_argument('-e','--epochs', default=1, type=int,
   help='number of total epochs to run')
  parser.add_argument('--lr',default=0.1,type=float,help='learning rate')
  parser.add_argument('-n','--num-workers',default=1, type=int,
   help='num workers in dataloader')
  parser.add_argument('--drop-last', default=False,action='store_true',
   help='activate drop_last option in dataloader')
  parser.add_argument('--chkpt',default=False,action='store_true',
   help='save checkpoint')
  parser.add_argument('--mom', default=0.9, type=float, help='momentum')
  parser.add_argument('--wd', default=0., type=float, help='weight decay')
  parser.add_argument('--image-size', default=224, type=int, help='Image size')
  parser.add_argument('--prof',default=False,action='store_true',
   help='profiling')
  args = parser.parse_args()
  print("DimResnet")
  print("==============")
  print("Batch size : ", args.batch_size)
  print("Learning rate : ", args.lr)
  print("Num workers : ", args.num_workers)
  print("Epoch : ", args.epochs)
  print("Checkpoint : ", args.chkpt)
  print("Drop Last : ", args.drop_last)
  print("Momentum :", args.mom)
  print("Weight Decay ", args.wd)
  print("Profile : ", args.prof)
  print("==============")
  training = DimResnetTraining(batchSize=args.batch_size,lr=args.lr,
   mom=args.mom, wd=args.wd, num_workers=args.num_workers,
   drop_last=args.drop_last,image_size=args.image_size)
  training.train(numEpochs=args.epochs,chkpt=args.chkpt,prof=args.prof)
  #training.train()
