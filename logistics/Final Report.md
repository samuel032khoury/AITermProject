<h1 align='center'>CS4100 Artificial Intelligence</h1>
<h2 align='center'>Term Project Final Report</h2>
<h4 align='center'>
    Face Mask Detection - A Deep Learning Project Using PyTorch
</h4>

## Inspiration

This project is inspired by the health concern and the indoor mask mandate during covid-19. We noticed the widely existing problem regarding the balance between the ineffective use of human labor and the potential failure of mandate. We then wanted to develop a program that can automatically identify if one individual is mask covered, and so secure a health-wise public safety without (or with least) human intervention.

After our research, we found such a program is plausible and has been implemented by the team of[ chandrikadeb7](https://github.com/chandrikadeb7/Face-Mask-Detection/commits?author=chandrikadeb7). We are highly inspired by this project, but decided to build our project from scratch instead of extending the existing one, and hope this may lead us to a better understanding of the computer vision and deep learning concepts involved in this application. We also decided to use PyTorch instead of Tensorflow, as it is an easier deep learning framework for us to learn.

## Problem Statement

We set the problem up in such a circumstance: Potentially infected people may need to work in a busy office building, such as a stock exchange or a media office. The health department/the company itself wants to secure both infected and uninfected people wearing their masks before they are entering the building/core area, so to prevent the virus spread. They can install new gate machines with cameras and use the existing CCTV system to get the images at the entrance/within the building. They want the system also to figure out if the individuals in front of the camera have their masks put on so to remind them (by turning off the gate or using acoustic/optic alarms) if they forget.

## Problem Formulation

We are going to analyze frames of video inputs. For each frame, we need to detect (and collect) human faces first, and then make further predictions. The detection model should not merely base on a **full** facial image, as noses and mouths can be covered by masks and thus make the object looks less than a (full) face - but it's still an input for our program. An alternative solution is to determine a threshold for the confidence of detection, so even an object that doesn't perfectly look like a human face (here, the mask-covered face) can still be identified and passed as inputs. By that, we know the two inputs for the model are '(snippet of) picture of bare human faces' vs '(snippet of) picture of masked human faces', and the output should be matched with our observation. To do this, we need a model trained by numerous data so it can figure out the difference between these two inputs and report classifications for different inputs consistently.

## Dataset

- We mainly used the [data](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset) (as `./data_big`) collected by the [pre-existing project](https://github.com/chandrikadeb7/Face-Mask-Detection) done by [chandrikadeb7](https://github.com/chandrikadeb7/Face-Mask-Detection/commits?author=chandrikadeb7), which contains 1,165 entries for images of faces with masks and 930 entries for images of faces without masks.
  - We split them randomly in a ratio of 7:3, and use two parts as training data and testing data, separately.
- We also collected a relative small [dataset from Kaggles](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection?resource=download&select=images) for development and debugging.

## Solution

### Data Preprocessing

Every training image will be applied to the following transforms before they are loaded into the training batches, so to make them standardized and representative enough.

```python
transforms.Compose([
            transforms.ToTensor(), # convert images to 3D tensors [h(eight),w(eight),c(hannel)]
    		transforms.RandomHorizontalFlip(), # randomly horizontally flip the image
            transforms.RandomRotation(30), # randomly rotate the image with maximum rotation of 30 degrees
    		transforms.RandomResizedCrop(256, (0.8, 1.0)), # randomly scaled the image
            transforms.CenterCrop(224), # crop the image from center to a 224 * 224 image
    		transforms.ColorJitter(), # randomly adjust color values
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images
        ])
```

We shuffle the order of the training data and load them into training batches of size 60.

Here we are applying a mini-batch strategy so to save computational power/time while secure the performance of the training.

### Training

We used a transform learning strategy for this project, as there have been already several well-built CNN architectures that are used for computer vision. Specifically, we chose to use MobileNetv2 as the CNN and from it, we can transform a 224×224×3 image tensor into a 1×1×1280 feature tensor. The final goal, then, is to classify those different 1×1×1280 feature tensors.

So we redefined the classifier of the model as the image below shows.

![Classifier Structure](../res/classifier.png)

The task left then is to train the weights and the bias factor for each of these fully connected layers.

We used ReLU as non-linear activation function between every two layers and selectively used drop off strategies. We used CrossEntropyLoss as loss function, and Adam algorithm as optimizer to conduct backward/forward propagations so to update these parameters until the loss converges.

### Face Detection & Classification

We applied an external face detection model `faceDetect.caffemodel` and lower the threshold of confidence to 50% so now even a face with masks can be identified. We then apply our own model to every face from the output of the face detection model and classify these inputs.

## Results

Overall, the model proved to work quite well under a well-lit environment. When the model was trained with `./data`, it had a classification accuracy of 88.16% (67/76) after 14 Epochs. When the model was trained with `./data_big`, it had a classification accuracy of 94.07% (1349/1434) after 2 Epochs.

### Analysis

It is evident that having a larger sample size for the training data set yielded a higher accuracy of classification for the model, which is to be expected. More Epochs were utilized for `./data` since there were far fewer samples to work with, so the extra computational time was minimal and convergence to a maximum accuracy required more Epochs. This PyTorch-based model performed almost as well as the existing Tensorflow-based project, which achieved an accuracy of 98% using the same dataset as in `./data_big`. Thus, our model proved to be quite accurate, but not without some potential room for improvement to minimize this ~4% discrepancy. With more time at our disposal, we could have made more alterations to the model architecture and performed more tests in order to improve the accuracy.

## Applications

The final product is running on a mac machine and is reading the video from the machine’s built-in front camera. The model classifies **every** face within the frame and marks label over it in real time. 

![running demo](../res/runDemo.gif)

## Future Directions

Given more time, there are a few things we would have done to improve the model. For instance, the current performance of the model under low-light conditions is sub-optimal. To remedy this, we could have added more low-light data to the training data set, which would allow the model to better recognize these cases in testing. In addition, the model cannot really distinguish between masks (of any type) and other objects that can be used to cover the face, such as a hand or a piece of paper. Thus, including these false-positive cases in the no-mask dataset would help the model better differentiate between a mask and non-mask object. Furthermore, we only had access to a large enough data set to train a binary classification model (mask or no-mask), so the performance for a multi-label classification is unknown. However, it would be doable with our model given enough data. We would need to acquire large data sets of different mask types (or separate out the current mask data into different types, such as cloth, surgical, KN95, N95, etc.) in order to assign labels to them. Thus, most of the limitations of the model come from a lack of data to cover the corner cases and different mask types, which would take a while to acquire but would greatly improve the model for more practical usage.

## References

1. **[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) Offered By [DeepLearning.AI](https://www.deeplearning.ai) on [Coursera](https://www.coursera.org/)**: Structural knowledge for DL.
2. **[Extracting faces using OpenCV Face Detection Neural Network](https://towardsdatascience.com/extracting-faces-using-opencv-face-detection-neural-network-475c5cd0c260)**: Inspiration for extracting faces from image
3. **[Face detection with OpenCV and deep learning](https://medium.com/@vinuvish/face-detection-with-opencv-and-deep-learning-90bff9028fa8)**: Inspiration for `runModel.py`, mostly the OpenCV part.
4. **[PyTorch Official Github Profile](https://github.com/pytorch)**: Reference for PyTorch relevant issue.
5. **[PyTorch Tutorials](https://github.com/pytorch/tutorials)**: Practical material of syntax and semantics of PyTorch; Help get familiar with functions.
6. **[PyTorch Vision](https://github.com/pytorch/vision)**: Official documentation as a reference for transfer learning
   - **[mobilenetv2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)**: The pre-trained model the projected used
7. **[Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)**: A similar project as a reference, also as our data set source