# MNIST classifier

In this task I was supposed to build a DigitClassifier model that takes an algorithm as an input parameter. Possible values for the algorithm are: cnn, rf, rand for the three models
described above.

The solution contains:
- Interface for models like Convolutional Neural Network, Random Forest classifier,
Random model. Potentially other developers will develop new models, so we
need to have an interface for them. Let’s call it DigitClassificationInterface.
- 3 classes (1 for each model) that implement DigitClassificationInterface.
- DigitClassifier, which takes as an input parameter the name of the algorithm
and provides predictions with exactly the same structure (inputs and outputs) not
depending on the selected algorithm.