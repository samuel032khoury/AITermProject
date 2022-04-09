import os
import cv2
import torch

def detectFace(model, frame):
    import numpy
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    array = numpy.reshape(blob, (300, 300, 3))
    # print(Image.fromarray(array))
    model.setInput(blob)
    detections = model.forward()
    faces=[]
    positions=[]
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX,startY)=(max(0,startX-15),max(0,startY-15))
        (endX,endY)=(min(w-1,endX+15),min(h-1,endY+15))
        confidence = detections[0, 0, i, 2]
        # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            face = frame[startY:endY, startX:endX]
            faces.append(face)
            positions.append((startX,startY,endX,endY))
    return faces,positions

def detectMask(model, faces):
    import PIL
    from torchvision import transforms
    predictions = []
    transRules = transforms.Compose([
        transforms.ToTensor(), transforms.Resize(size=(114,114)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if (len(faces)>0):
        for face in faces:
            face = PIL.Image.fromarray(face)
            face = transRules(face)
            face = face.unsqueeze(0)
            prediction = model(face)
            prediction = prediction.argmax()
            predictions.append(prediction.data)
    return predictions

def run(modelName=None):
    prototxtFile = os.getcwd() + '/model_data/deploy.prototxt'
    faceDetectFile = os.getcwd() + '/model_data/faceDetect.caffemodel'
    faceDetectModel = cv2.dnn.readNetFromCaffe(prototxtFile, faceDetectFile)

    modelNamePrompt = 'Enter the file name of the model want to use: '

    while True:
        if modelName == None:
            modelName = input(modelNamePrompt)
            maskDetectFile = os.getcwd() + '/model_data/' + modelName
        try:
            maskDetectModel = torch.load(maskDetectFile)
            break
        except (FileNotFoundError, IsADirectoryError):
            modelName = None
            modelNamePrompt = 'Fail to find the model! Please check the model name and enter agian: '

    print('Loading Model...')
    maskDetectModel.eval()

    cap = cv2.VideoCapture(0)
    try:
        while True:
            _, frame = cap.read()
            (faces,postions) = detectFace(faceDetectModel, frame)
            predictions=detectMask(maskDetectModel, faces)
            
            for(box,prediction) in zip(postions,predictions):
                (startX, startY, endX, endY) = box
                label = "Mask" if prediction == 0 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (255,0,0)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame,(startX, startY),(endX, endY),color,2)
            cv2.imshow('Detecting...',frame)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
        print()
        print('Program terminated!')

if __name__ == '__main__':
    run()
