# ExML
Repository for advGAN and other ExML experiments.
Models are too large to be pushed to Github, so they must be shared via another method. 

**requirments.txt:** Has all required Python libraries. Note if the large libraries (such as PyTorch) do not install, use ```pip --no-cache-dir install LIBRARY```. To set up a virtual enviroment:
- ```python3 -m venv NAME```
- ```source NAME/bin/activate```
- ```pip install -r requirments.txt```
The enviroment was not included on the Github because of its size. There are also the required files to set up a Conda enviroment, see **requirments_conda.yml** and **requirments_conda.txt**.

## ./advgan_notebooks
- **pytorch_mnist.ipynb:** Basic advGAN implementation in PyTorch on the MNIST data set. Can use feedforward or convolutional nets
- **rf_mnist.ipynb:** advGAN implementation where target model is a random forest classifier
- **distilled_rf_mnist.ipynb:** advGAN implementation where the target model is a NN distilled from query access to a random foret model
- **pytorch_cifar10.ipynb:** advGAN implementation on the CIFAR-10 dataset, uses convolutional NNs
- **testing.ipynb:** Ignore, used by me for checking tensor shapes etc
- **GAN_.py:** Classes and functions to create/save a the generator from the advGAN implementation
- **gans_archs_.py:** Classes and functions to for various generator/discriminator architectures
- **net_.py:** Supporting code for creating, training, and testing a target model (feedforward for MNIST)
- **net_conv.py:** Supporting code for creating, training, and testing a target model (convolutional for MNIST)
- **net_conv_cifar.py:** Supporting code for creating, training, and testing a target model (convolutional for CIFAR-10)
- **./advgan_models:** Directory to store generators from advGAN implementations
- **./target_models:** Directory to store trained target models

To download pretrained models, please visit [this google drive folder](https://drive.google.com/file/d/1xoK0PJIr8G2vih830ANEaEoodwfGRwSd/view?usp=sharing) (models updated 2/10/21). Move both folders from ```./model``` to ```./advgan_notebooks```.

### Other Things to Do:
- advGAN attack on backdoor attacked model, what happens?
- Do other attacks on distilled feedforward transfer to RF model? 
- Experiment with different discriminator arcitectures 
- Dynamic distillation (as they did in the paper)
- GAN stabalization techniques applied to advGAN
- Concatenate a small noise vector to the input image 
- Do advGAN example on CIFAR, and use convolutional NNs

### Naming scheme for generators:
```./advgan_models/[custom text]_device[device name][number of classes]classes_[discriminator coefficient]disc_[hinge loss coefficient]hinge_[adversarial coefficient]adv.pt```

### Naming scheme for target models: 
```./target_models/[type of model]_[number of classes]classes.[extension]```

## ./lime_tests
Contains notebooks of various LIME testing. Can we use LIME to detect adversarial examples? What about using it to measure model similarity?
