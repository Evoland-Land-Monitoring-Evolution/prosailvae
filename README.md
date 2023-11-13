# prosailvae

## Important files
### Training a model
The module `prosailvae/train.py` enable to both train a model (PROSAIL-VAE or supervised regression) and compute performance results (validation with in situ data and s2 images). 
This module can be called in the terminal with options described in the argparse parser inside the file.
The path to a json config file must be passed as an argument to this module. 
This file contains all relevant options for computations, in particular the hyperparameters and the paths to the training data-set.

### Simulate a data-set
Use `dataset/generate_dataset.py` to simulate a training dataset with PROSAIL.

### In situ data
The in-situ data was preprocessed for faster validation, using `validation/belsar_validaton.py` and `validation/frm4veg_validation.py`. 
Please be aware that preprocessing in situ data is RAM-intensive, because it requires extracting angles on the whole tile of a Theia product using `sensorsio` library.

### Computing results
Computing results for a trained model is done automatically after training with `prosailvae/train.py`. To recompute results, for a trained model, simply load the model with appropriate options in the config file, and set 0 epochs for training.

## Config note
Due to the tensor dimensions being different with a pixellic and a spatial model, the loss must be slected adequately.
For a pixellic model `rnn`, please use `diag_nll` as loss. For a spatial model `rcnn`, use `spatial_nll` as loss.

A sample config which loads a trained model is provided (for PC use).
