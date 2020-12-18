# ExML
Repository for advGAN and other ExML experiments.
Models are too large to be pushed to Github, so they must be shared via another method. 

**requirments.txt:** Has all required Python libraries. Note if the large libraries (such as PyTorch) do not install, use ```pip --no-cache-dir install LIBRARY```. To set up a virtual enviroment:
- ```python3 -m venv NAME```
- ```source NAME/bin/activate```
- ```pip install -r requirments.txt```
The enviroment was not included on the Github because of its size. 

## ./advgan_notebooks
- **advgan_.ipynb:** Basic advGAN implementation in PyTorch
- **advgan_rftarget.ipynb:** advGAN implementation where target model is a random forest classifier
- **advggan_rfDistilled.ipynb:** advGAN implementation where the target model is a NN distilled from query access to a random foret model
- **advGAN.py:** Classes and functions to create/save a the generator from the advGAN implementation
- **net_.py:** Supporting code for creating, training, and testing a target model
- **./advgan_models:** Directory to store generators from advGAN implementations
- **./target_models:** Directory to store trained target models

To download pretrained model, please visit [this google drive folder](https://drive.google.com/file/d/1IzMjgQrjz51piyR2AcP953qZeGBczWBs/view?usp=sharing). Move both folders from ```\.model``` to ```./advgan_notebooks```.

### Naming scheme for generators:
```./advgan_models/[custom text]_[number of classes]classes_[discriminator coefficient]disc_[hinge loss coefficient]hinge_[adversarial coefficient]adv.pt```

### Naming scheme for target models: 
```./target_models/[type of model]_[number of classes]classes.[extension]```

## ./lime_tests
Contains notebooks of various LIME testing. Can we use LIME to detect adversarial examples? What about using it to measure model similarity?
