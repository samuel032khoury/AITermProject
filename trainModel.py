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
        'test': DataLoader(data['test'], shuffle=True)
    }

    return data, dataLoaders

def train(epochs, model, trainingLoader, trainingSize):
    
    from tqdm import tqdm
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
    from tqdm import tqdm
    print("\nSTART TESTING MODEL")
    model.eval()
    correct = 0
    for imgs, labels in tqdm(testingLoarder, unit = 'imgs'):
        outputs = model(imgs)
        correct = correct + 1 if torch.argmax(outputs.data).item() == labels else correct
    print('Testing Report:')
    print('The accuracy of the trained model is {:.2f}% ({:d}/{:d})'.format(correct / testingSize * 100, correct, testingSize))

def run(modelName = None):
    import math
    try:
        dataDirPrompt = "Choose the data directory for training and testing (skip to use ./data): "
        while True:
            try:
                dataDirIn = input(dataDirPrompt)
                if dataDirIn == "":
                    dataDir = 'data'
                    break
                else:
                    dataDir = dataDirIn
                    if not os.path.isdir(os.getcwd()+'/' + dataDir + '/train/') or not os.path.isdir(os.getcwd()+'/' + dataDir + '/test/'):
                        raise FileNotFoundError
                    break
            except FileNotFoundError:
                dataDirPrompt = 'Fail to find specified directory or it does not have desired structure! \nPlease try again (or skip to use ./data): '
                dataDir = 'data'


        data, dataLoaders = preprocessData(dataDir)

        model = torch.hub.load('pytorch/vision:v0.12.0', 'mobilenet_v2', pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier[1] = nn.Sequential(
            nn.Linear(1280, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), 
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4), 
            nn.Linear(32, 2), nn.LogSoftmax(dim=1))

        epochsPrompt = "Enter the Number of Epochs (skip to use the recommended Epochs {}): "
        while True:
            try:
                recommended = math.ceil(2000 / len(data['train']))
                epochsIn = input(epochsPrompt.format(recommended))
                epochs = recommended if  epochsIn == "" else int(epochsIn)
                break
            except ValueError:
                epochsPrompt = "Integer required, please try again (skip to use the recommended Epochs {}): "
        
        train(epochs, model, dataLoaders['train'], len(data['train']))
        test(model, dataLoaders['test'], len(data['test']))

        modelName = input("Enter the name for the model: ") if modelName == None else modelName
        torch.save(model, "./model_data/"+modelName+".pth")
        return data['train'].classes
    except KeyboardInterrupt:
        print("\nExiting...", sep="")

if __name__ == '__main__':
    run()
    