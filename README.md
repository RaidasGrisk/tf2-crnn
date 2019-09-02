# TF-2 CRNN

CNN + RNN for Scene Text Recognition implemented using tf2 keras module. Mostly based on [this](https://github.com/Belval/CRNN) and [this](https://github.com/MaybeShewill-CV/CRNN_Tensorflow) repo. The project code is written as simply as I could. This is to make it easy to understand and debug. Also to get to know the tf2 keras module better.

Project structure:
1. [**Model.**](/crnn_model.py) The model definition and architecture.
2. [**Training.**](/train.py) Code for training.
3. [**Test.**](/test.py) Code for testing the pre-trained model.
4. [**Utils.**](/utils.py) All the utility functions used.

# Results

Some examples of pre-trained model at work.

| image | recognition |
|-------|-------------|
|<img src="/test_images/test.jpg" width="200" height="30"/>| Rick |
|<img src="/test_images/test2.jpg" width="200" height="30"/>| MORTY |
|<img src="/test_images/1_pontifically_58805.jpg" width="200" height="30"/>| pontifically |
|<img src="/test_images/46_Sparkling_73104.jpg" width="200" height="30"/>| Sparkling |
|<img src="/test_images/7_Bombastic_8610.jpg" width="200" height="30"/>| Bombastic |
|<img src="/test_images/43_croons_18234.jpg" width="200" height="30"/>| Croons |
|<img src="/test_images/49_Testicles_78366.jpg" width="200" height="30"/>| Testicles |


# TODO
- [x] pre-train the model on more data
- [ ] visualize training progress
