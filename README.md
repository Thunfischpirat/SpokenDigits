# Project: Spoken Digit Recognition

## Description

This is our submission for the final graded project for the WS22/23 course *"Neural Networks: Theory and Implementation"* at Saarland University.

We focus on developing a SDR system in a speaker-independent setting. That is, the speakers in the evaluation set are disjoint from the training set speakers. We do so because we expect real-world ASR systems to generalize to different speakers than those we have data for. Moreover, for many languages that are under-resourced, we have have limited annotated speech data from a single speaker, but we would still want the system to be deployed to work on any speaker of that language. We tackle the problem of spoken digit recognition as a sequence classification task. Concretely, the inputs are short audio clips of a specific digit (in the range 0-9), then the goal is to build deep neural network models to classify a short audio clip and predict the digit that was spoken.

## Dataset
The dataset contains 3000 audio clips of spoken digits (0-9) in English in `.wav` format it can be found in the folder `speech_data`.
The total size of the dataset is 26Mb. The file `SDR_metadata.tsv` contains information such as the labels of the audio clips and to whether they are used for training, evaluation or testing.

## Installation
The Python version used in our project is 3.9.13. You can use poetry to install the dependencies. To set up the poetry environment, run the following command:
```bash
poetry install
poetry run pip3 install torch torchvision torchaudio torchmetrics --extra-index-url https://download.pytorch.org/whl/cu117
```
The last step is necessary to ensure that pytorch is installed with GPU support. Alternatively you can use the `requirements.txt` file to install the dependencies.

## Models

Models are trained based on Mel-spectrograms of the original raw audio files. In total we train four models. The simplest one is a linear model applied to truncated Mel-spectrograms. Along with the utilities necessary to generate its train- validation and test- data it can be found in the folder `model_baseline`. The folder `model_neural` contains implementations of two distinct types of neural network architectures. The first is a 1d convolutional neural network and the second is a vision transformer based neural network. The utilities for training the models are contained in the `utils` subfolder.

Apart from supervised training, this project also has an implementation of unsupervised training based on contrastive loss. The contrastive loss based training is implemented in the file `contrastive_training.py` contained in the `model_neural` folder.

## Further Details
For further details about the project feel free to take a look at the project report pdf contained in this repository. Our most important sources for inspiration and code were:

- https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html [1D Convolutional Model]
- https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html [Vision Transformer]
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html [Contrastive Learning]
