import cv2
import torch

def predict(maskDetectModel, frame):
    import PIL
    import numpy
    from torchvision import transforms

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    prototextPath = './models/deploy.prototxt'
    caffeModel = './models/faceDetect.caffemodel'
    faceDetectModel = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)
    faceDetectModel.setInput(blob)
    
    detections = faceDetectModel.forward()
    print(detections.shape)
    
    taskList = []
    h, w = frame.shape[:2]
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (confidence > 0.5):
            box = detections[0, 0, i, 3:] * numpy.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            face = frame[startY:endY, startX:endX]
            poistion = (startX,startY,endX,endY)
            taskList.append((face, poistion))

    predictions = []
    transRules = transforms.Compose([
        transforms.ToTensor(), transforms.Resize(size=(114,114)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for face, position in taskList:
        face = PIL.Image.fromarray(face)
        face = transRules(face).unsqueeze(0)
        prediction = maskDetectModel(face).argmax().item()
        predictions.append((prediction,position))
    return predictions

def run(modelName=None):
    cap = None
    modelNamePrompt = 'Enter the model name: '

    try:
        while True:
            if modelName == None:
                modelName = input(modelNamePrompt)
            try:
                maskDetectFile = './models/' + modelName + '.pth'
                maskDetectModel = torch.load(maskDetectFile)
                print('Loading Model...')
                break
            except (FileNotFoundError, IsADirectoryError):
                modelName = None
                modelNamePrompt = 'Fail to find the model! Please check ./models and enter agian: '

        maskDetectModel.eval()
        cap = cv2.VideoCapture(0)
        while True:
            on, frame = cap.read()
            if not on:
                print('No camera is available!')
                raise ValueError
            
            for prediction, poistion in predict(maskDetectModel, frame):
                (startX, startY, endX, endY) = poistion
                label = ' '.join([w.capitalize() for w in maskDetectModel.lableValMap[prediction].split('_')])
                color = (0,0,255) if "No " in label else (0, 255, 0)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                cv2.rectangle(frame,(startX, startY),(endX, endY),color, 2)
            cv2.imshow('Detecting...',frame)
            cv2.waitKey(1)
    except (KeyboardInterrupt, ValueError):
        if not cap is None:
            cap.release()
        cv2.destroyAllWindows()
        print('\nProgram terminated!')

if __name__ == '__main__':
    run('big')