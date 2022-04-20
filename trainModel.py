import torch
import torch.nn as nn
from tqdm import tqdm

def preprocessData(dataDir):
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    transRules = {
        'train':
        transforms.Compose([
            transforms.ToTensor(),transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),transforms.RandomResizedCrop(256, (0.8, 1.0)),
            transforms.CenterCrop(224),transforms.ColorJitter(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test':
        transforms.Compose([
            transforms.ToTensor(),transforms.Resize(256),transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data = {
        'train': ImageFolder(root='./' + dataDir + '/train', transform=transRules['train']),
        'test': ImageFolder(root='./' + dataDir + '/test', transform=transRules['test'])
    }

    dataLoaders = {
        'train': DataLoader(data['train'], batch_size = 60, shuffle=True),
        'test': DataLoader(data['test'], shuffle=True)
    }

    return data, dataLoaders

def train(epochs, model, trainingLoader, trainingSize):
    cel = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loaderSize = len(trainingLoader)

    print("START TRAINING MODEL")
    for epoch in range(epochs):

        print("TRAINING PROGRESS: {}/{} Epoch".format((epoch + 1), epochs))
        model.train()

        with tqdm(total = trainingSize, unit = 'imgs') as pbar:
            for ind, (imgs, label) in enumerate(trainingLoader):
                opt.zero_grad()
                output = model(imgs)
                cel(output, label).backward()
                opt.step()

                if ind == loaderSize - 1:
                    pbar.update(trainingSize % 60)
                else:
                    pbar.update(60)

def test(model, testingLoarder, testingSize):
    print("\nSTART TESTING MODEL")
    model.eval()
    correct = 0
    for imgs, labels in tqdm(testingLoarder, unit = 'imgs'):
        outputs = model(imgs)
        if torch.argmax(outputs.data).item() == labels:
            correct += 1
    print('Testing Report:')
    print('The accuracy of the trained model is {:.2f}% ({:d}/{:d})'.format(correct / testingSize * 100, correct, testingSize))

def run(modelName = ''):
    import os
    import math
    try:
        dataDirPrompt = "Choose the data directory for training and testing (skip to use ./data): "
        while True:
            dataDir = input(dataDirPrompt)
            if dataDir == "":
                dataDir = 'data'
                break
            else:
                if not os.path.isdir(os.getcwd()+'/' + dataDir + '/train/') or not os.path.isdir(os.getcwd()+'/' + dataDir + '/test/'):
                    dataDirPrompt = 'Fail to find specified directory or it does not have desired structure!\nPlease try again (or skip to use ./data): '
                else:
                    break

        data, dataLoaders = preprocessData(dataDir)

        model = torch.hub.load('pytorch/vision:v0.12.0', 'mobilenet_v2', pretrained=True)

        for p in model.parameters():
            p.requires_grad = False

        model.classifier[1] = nn.Sequential(
            nn.Linear(1280, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), 
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4), 
            nn.Linear(32, len(data['train'].classes)), nn.LogSoftmax(dim=1))

        epochsPrompt = "Enter the Number of Epochs (skip to use the recommended Epochs {}): "
        while True:
            try:
                recommended = math.ceil(4000 / len(data['train']))
                epochsIn = input(epochsPrompt.format(recommended))
                epochs = recommended if  epochsIn == "" else int(epochsIn)
                break
            except ValueError:
                epochsPrompt = "Integer required, please try again (skip to use the recommended Epochs {}): "
        
        model.lableValMap = data['train'].classes
        train(epochs, model, dataLoaders['train'], len(data['train']))
        test(model, dataLoaders['test'], len(data['test']))

        ab = 'abcdefghijklmnopqrstuvwxyz'
        strFilter= ab + ab.upper() +'1234567890'
        nameModelPrompt = "Enter the name for the model: "
        while True:
            modelName = input(nameModelPrompt) if not modelName else modelName
            modelName = modelName[:-4] if modelName.endswith('.pth') else modelName
            if all(c in strFilter for c in modelName):
                if modelName:
                    break
                else:
                    nameModelPrompt = "Model Name cannot be empty: "
            else:
                modelName = ''
                nameModelPrompt = "Invalid name for the model, use letters and numbers only: "

        torch.save(model, "./models/"+modelName+".pth")
        print('==== Model successively saved as ./models/{}.pth ===='.format(modelName))
        runModelNow = (input('Run the model now (y/n)? ') == 'y')
        return runModelNow, modelName
    except KeyboardInterrupt:
        print("\nExiting...", sep="")
        return False, None

if __name__ == '__main__':
    import runModel
    runModelNow, modelName = run()
    if runModelNow:
        runModel.run(modelName)
    