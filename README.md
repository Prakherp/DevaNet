## DevaNet: Handwritten Hindi Text Recognition System

### Overview

DevaNet is a Handwritten Hindi Text Recognition System designed to accurately transcribe handwritten text images in the Devanagari script into machine-readable text. Leveraging the IIIT-HW-Dev dataset, DevaNet combines state-of-the-art deep learning techniques to address the complexities of handwritten Hindi text recognition.

### Features

- **Large-Scale Devanagari Dataset**: Utilizes the IIIT-HW-Dev dataset with over 95,000 handwritten words in Devanagari script, providing a rich and varied collection of samples for model training and evaluation.
- **Benchmarking with State-of-the-Art Methods**: Thorough evaluation and comparison against established techniques for handwritten word recognition.
- **Advanced Preprocessing Pipeline**: Standardizes and enhances the quality of input images through resizing, transposition, normalization, and batch processing.
- **Robust Model Architecture**: Combines Convolutional Neural Networks (CNN) for feature extraction and Recurrent Neural Networks (RNN) for sequence modeling.
- **CTC Loss Function**: Employs the Connectionist Temporal Classification (CTC) loss function to handle variable-length sequences and improve training efficiency.

### Dataset

The IIIT-HW-Dev dataset serves as the backbone of DevaNet, providing a comprehensive collection of handwritten Devanagari words for training, validation, and testing. The dataset is structured into train, validation, and test subsets, with a detailed vocabulary file mapping unique words to their corresponding labels.

### Preprocessing and Data Loading

DevaNet includes a robust preprocessing pipeline to ensure data quality and consistency:
- **Handling Damaged Files**: Replaces missing or damaged images with black images to maintain dataset consistency.
- **Resizing**: Ensures uniform dimensions of input images while preserving aspect ratio.
- **Transposition and Normalization**: Aligns images with TensorFlow requirements and normalizes intensity values for stable training.
- **Batch Processing**: Efficiently preprocesses and stacks images and labels into batches for parallel processing during training and evaluation.

### Model Architecture

The core of DevaNet's recognition capability lies in its sophisticated model architecture:
- **Convolutional Neural Network (CNN)**: Extracts meaningful features from handwritten text images through multiple convolutional and pooling layers.
- **Recurrent Neural Network (RNN)**: Captures temporal dependencies and sequential patterns in handwritten text sequences.
- **CTC Loss Function**: Facilitates learning from unsegmented and variable-length sequences, optimizing the model through the RMSProp optimizer and adaptive learning rate strategies.

  <img width="941" alt="image" src="https://github.com/Prakherp/DevaNet/assets/79146254/9c85ae74-b092-4f43-8340-630cacdf6394">


### Training and Evaluation

DevaNet's training process includes:
- **Optimizer Initialization**: Utilizes the RMSProp optimizer to minimize CTC loss and update model parameters.
- **Training Loop**: Sequentially processes batches of input images and ground truth texts, adjusting parameters through backpropagation.
- **Validation and Model Saving**: Periodically evaluates performance on a validation set, computes accuracy and error metrics, and saves the trained model for future use.

### Results

After 61 epochs of training, DevaNet achieved a character error rate (CER) of 25.22%. Although initially high, this performance lays the foundation for further optimization through additional epochs, model tuning, data augmentation, and resource scaling.

### Future Work

To improve DevaNet's accuracy and reduce the CER, future work will focus on:
- Increasing the number of training epochs.
- Experimenting with different model architectures and hyperparameters.
- Applying data augmentation techniques.
- Utilizing more powerful hardware resources for faster and more efficient training.

### Conclusion

DevaNet demonstrates significant potential in transcribing handwritten Devanagari script with high accuracy. Through continuous iterations and optimizations, DevaNet aims to achieve state-of-the-art performance in handwritten Hindi text recognition.
