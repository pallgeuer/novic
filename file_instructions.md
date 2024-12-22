# NOVIC Repository: File Explanations by Pipeline Order

This document explains the purpose and functionality of the files in the NOVIC 
repository, organized in the order of the pipeline for training and running the model.

---

## **1. Caption Generation**
These files are responsible for generating and validating text captions for images, a 
key step in creating the dataset.

### `caption_generation.py`
- **Purpose**: Automatically generates captions for images. These captions describe 
the content or objects in an image and are used as training data.
- **Functionality**:
  - Accepts raw images or image embeddings as input.
  - Uses predefined templates or a language model to create descriptive captions.
  - Enables the creation of diverse and scalable datasets.

### `caption_generation_check.py`
- **Purpose**: Validates the quality of generated captions.
- **Functionality**:
  - Ensures captions are grammatically correct, coherent, and aligned with image 
content.
  - Flags captions that fail quality checks for further review.

---

## **2. Dataset Preparation**
Handles organizing and preprocessing datasets for training and validation.

### `classification_dataset.py`
- **Purpose**: Prepares datasets for classification tasks.
- **Functionality**:
  - Reads raw image files and labels.
  - Performs data preprocessing (e.g., resizing, normalization).
  - Applies data augmentation techniques (e.g., flipping, rotation) to enhance 
generalization.
  - Splits datasets into training, validation, and testing sets.

### `dataset_annotation.py`
- **Purpose**: Annotates datasets with additional information like object labels or 
metadata.
- **Functionality**:
  - Assigns class labels or tags to images.
  - Creates structured datasets for supervised learning tasks.

---

## **3. Embedding Generation**
Embeddings are vector representations of text or images that the model processes.

### `embedders.py`
- **Purpose**: Generates embeddings for images and text.
- **Functionality**:
  - Converts raw images into embeddings using a pre-trained CLIP model.
  - Maps text into the same embedding space for comparison and classification.
  - Bridges the gap between modalities for zero-shot tasks.

---

## **4. Embedding Management**
Caching and loading embeddings to optimize training and inference.

### `embedding_cache.py`
- **Purpose**: Manages caching of embeddings to improve performance.
- **Functionality**:
  - Stores precomputed embeddings to reduce redundant calculations.
  - Provides fast access to cached embeddings during training or inference.

### `embedding_cache_writers.py`
- **Purpose**: Writes embeddings to the cache.
- **Functionality**:
  - Handles the storage of embeddings in a structured format.
  - Ensures data integrity for cached embeddings.

### `embedding_dataset.py`
- **Purpose**: Organizes datasets consisting of embeddings rather than raw data.
- **Functionality**:
  - Loads embeddings for training or evaluation.
  - Structures embeddings for batching and feeding into models.

---

## **5. Embedding Enhancement**
Improving embedding robustness and interpretability.

### `embedding_noise.py`
- **Purpose**: Introduces controlled noise to embeddings during training.
- **Functionality**:
  - Adds Gaussian or uniform angle noise to embeddings to simulate variability.
  - Helps the model generalize better and handle modality gaps (e.g., between text and 
images).

---

## **6. Decoding**
Translating embeddings into meaningful outputs.

### `embedding_decoder.py`
- **Purpose**: Decodes embeddings into human-readable object labels.
- **Functionality**:
  - Processes embeddings using an autoregressive transformer.
  - Generates free-form text outputs, such as object nouns, directly from embeddings.
  - Central to the zero-shot classification functionality of NOVIC.

---

## **7. Automation and Logging**
Streamlining processes and monitoring execution.

### `gpt_annotation.py`
- **Purpose**: Uses GPT to annotate datasets.
- **Functionality**:
  - Generates descriptive labels or captions for images using a GPT model.
  - Adds contextual information to datasets, enhancing their richness.

### `logger.py`
- **Purpose**: Implements logging for tracking execution and debugging.
- **Functionality**:
  - Records events, errors, and other significant activities during script execution.
  - Outputs logs to files or the console for easy monitoring.

---

## **8. Noun-Specific Dataset**
Focused on datasets involving object nouns.

### `noun_dataset.py`
- **Purpose**: Manages datasets centered around object nouns.
- **Functionality**:
  - Creates datasets of nouns and their associated images or embeddings.
  - Ensures balanced representation of different object categories for training.

---

## **Pipeline Overview**
1. **Caption Generation**: Use `caption_generation.py` to create captions and validate 
them with `caption_generation_check.py`.
2. **Dataset Preparation**: Prepare datasets using `classification_dataset.py` and 
annotate them with `dataset_annotation.py`.
3. **Embedding Generation**: Generate embeddings for text and images using 
`embedders.py`.
4. **Embedding Management**: Cache and load embeddings using `embedding_cache.py`, 
`embedding_cache_writers.py`, and `embedding_dataset.py`.
5. **Embedding Enhancement**: Add noise to embeddings with `embedding_noise.py` to 
improve robustness.
6. **Decoding**: Decode embeddings into meaningful outputs using 
`embedding_decoder.py`.
7. **Automation and Logging**: Automate dataset annotation with `gpt_annotation.py` 
and track execution with `logger.py`.
8. **Noun Dataset**: Use `noun_dataset.py` to handle datasets focused on object nouns.

This structure ensures a smooth workflow for training and running the NOVIC model, 
from data preparation to real-time inference.

