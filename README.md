# ExML
Repository for advGAN and other ExML experiments.
Models are too large to be pushed to Github, so they must be shared via another method. 

### ./lime_tests
Contains notebooks of various LIME testing. Can we use LIME to detect adversarial examples? What about using it to measure model similarity?

### ./advgan_notebooks
**advgan_.ipynb:** Basic advGAN implementation in PyTorch
**advgan_rftarget.ipynb:** advGAN implementation where target model is a random forest classifier
**advggan_rfDistilled.ipynb:** advGAN implementation where the target model is a NN distilled from query access to a random foret model
**advGAN.py:** Classes and functions to create/save a the generator from the advGAN implementation
**net_.py:** Supporting code for creating, training, and testing a target model
**./advgan_models:** Directory to store generators from advGAN implementations
**./target_models:** Directory to store trained target models
Naming scheme for generators ```test```
