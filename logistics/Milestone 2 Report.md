# Milestone 2 Report

## Dataset

- Since the data downloaded from kaggle are not standardized enough (there are multiple instances that one image contains more than one face and they don't necessarily represent the same label [mask]/[no_mask]). We extracted individual faces[^1] from the original data and saved as copies. We manually removed all the false positive results (non-human faces) and separate it in a ration of 7:3 as training set and testing set.
  - Still, the data set is not perfectly ideal as it can be too small for a deep learning network. So we downloaded data from an existing project having the same goal[^2] and make it an alternative data set.

## Conceptual Work

- Team members have had enough PyTorch and OpenCV knowledge for implementing a program by the due of this report.
- Team members have started learning DL knowledge including logistic regression, neural networks, forward and backward propagation, loss function, activation function, optimization, CNN, etc[^3]. Learning notes were taken for future reference.

## Practical Work

By the due of this report, we already have a functioning model-training program which takes a dataset and produces a prediction model stored into `./model_data`.

The program uses a pre-trained convolution neural network model, *mobilenet_v2*[^4], from pytorch.vision, with its classifier modified by us, and is able to conduct forward/backward propagation to optimize model parameters.

The testing reports it generates a model has an ~97% prediction accuracy after training 10 Epochs using 300 training instances or 1 Epoch using 2200 training instances.

The program also has a descent user interface and relatively easy/intuitive to use.;



## Weekly Plan Completeness

- Week 1 - 100%
  - Data were collected
  - PyTorch basics were learned
- Week 2 - 100%
  - Learning algorithm/model was determined: DNN & CNN[^4]
  - Members are able to understand and use PyTorch/OpenCV functions
- Week 3 - 100%
  - The prediction model can be built, the application program (which applies the trained model for synchronous detecting) is still under development (due pushed off to week 6)
  - Research for NN has been at the end, more detailed information may be researched for a better understanding
  - Code review (for model-building) has been conducted
- Week 4 - 100%
  - Testing reported the trained model (307 training images, 10 Epochs) has an accuracy of ~97%, considered as a satisfactory result by the team.
  - This is going to be reviewed after the application program is fully developed, to see if the model can make low bias and low variance prediction in practice
- Week 5 - 100%
  - As the testing gives good results, we are not going to further modify the algorithm and the model, unless practical usage cause unexpected issues.
- Week 6 (Ongoing)
  - Code review (for model-using) will be conducted after the program is fully developed
  - Presentation will be created after the program is finalized

## Next Steps

- Finish the application program
- Start to prepare the presentation and assign team members responsibilities
- Start the final report




[^1]: https://towardsdatascience.com/extracting-faces-using-opencv-face-detection-neural-network-475c5cd0c260
[^2]:https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset
[^3]: Resourced from https://www.coursera.org/specializations/deep-learning
[^4]: Pretrained model *mobilenetv2*, https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
