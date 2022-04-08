import os
import torch
import torch.nn as nn

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
        'train': ImageFolder(root=os.getcwd()+'/' + dataDir + '/train', transform=transRules['train']),
        'test': ImageFolder(root=os.getcwd()+'/' + dataDir + '/test', transform=transRules['test'])
    }

    dataLoaders = {
        'train': DataLoader(data['train'], batch_size = 60, shuffle=True),
        'test': DataLoader(data['test'], batch_size = 60, shuffle=True)
    }

    return data, dataLoaders

def train(epochs, model, trainingData, trainingLoader):
    
    from tqdm import tqdm
    cel = torch.nn.CrossEntropyLoss()
    trainingSize = len(trainingData)
    loaderSize = len(trainingLoader)

    for epoch in range(epochs):

        print("EPOCH {} - TRAINING MODEL".format((epoch + 1)))
        model.train()

        with tqdm(total = trainingSize, unit = 'imgs') as pbar:
            for ind, (img, label) in enumerate(trainingLoader):
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                optimizer.zero_grad()
                output = model(img)
                cel(output, label).backward()
                optimizer.step()

                if ind == loaderSize - 1:
                    pbar.update(trainingSize % 60)
                else:
                    pbar.update(60)
        
        print("EPOCH {} - EVALUATING MODEL".format((epoch + 1)))
        model.eval()

        truePos = 0 
        totalTrail = 0

        with tqdm(total = trainingSize, unit = 'imgs') as pbar:
            for ind, (img, label) in enumerate(trainingLoader):
                output = model(img)
                cel(output,label) 
                correct = torch.eq(torch.max(nn.functional.softmax(output, dim=1), dim=1)[1], label).view(-1)
                truePos += torch.sum(correct).item()
                totalTrail += correct.shape[0]
                
                if ind == loaderSize - 1:
                    pbar.update(trainingSize % 60)
                else:
                    pbar.update(60)

        print('ACCURACY FOR EPOCH {} : {:.4f}'.format((epoch + 1), truePos / totalTrail))
        print()

def test(model, testingLoarder):
    correct = 0
    total = 0
    with torch.no_grad():
        for img, labels in testingLoarder:
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print('Testing Report:')
    print('correct: {:d}  total: {:d}'.format(correct, total))
    print('accuracy = {:f}'.format(correct / total))
    print()

def run(dataDir = 'data', modelName = None):
    
    data, dataLoaders = preprocessData(dataDir)

    model = torch.hub.load('pytorch/vision:v0.12.0', 'mobilenet_v2', pretrained=True)

    model.classifier[1] = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(), 
        nn.Linear(256, 128),
        nn.ReLU(), 
        nn.Dropout(0.4), 
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.4), 
        nn.Linear(32, 2), 
        nn.LogSoftmax(dim=1))

    try:
        epochsPrompt = "Enter the Number of Epochs (skip to use the recommended Epochs {}): "
        lrPrompt = "Select in [0 - 100] as the Learning Rate (skip to use the default learning rate 75): "
        while True:
            try:
                import math
                recommended = math.ceil(2000 / len(data['train']))
                epochsIn = input(epochsPrompt.format(recommended))
                epochs = recommended if  epochsIn == "" else int(epochsIn)
                break
            except ValueError:
                epochsPrompt = "Integer required, please try again (skip to use the recommended Epochs {}): "
        while True:
            try:
                lrIn = input(lrPrompt)
                lr = 75 if  lrIn == "" else int(lrIn)
                break
            except ValueError:
                lrPrompt = "Integer required, please try again (skip to use the default learning rate 75):"
        
        train(epochs, model, data['train'], dataLoaders['train'])
        test(model, dataLoaders['test'])

        modelName = input("Enter the name for the model: ") if modelName == None else modelName
        torch.save(model, "./model_data/"+modelName+".pth")
        return data['train'].classes
    except KeyboardInterrupt:
        print()
        print("Exiting...")

if __name__ == '__main__':
    run()
    
