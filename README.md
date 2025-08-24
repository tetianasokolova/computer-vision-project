# Computer Vision Project: Animals-10 Image Classification 

This project focuses on classifying images from the **Animals-10 dataset**, which contains 10 animal classes. The primary goal is to build, train, and optimize deep learning models to accurately classify these images. The project is implemented using **PyTorch** as the primary deep learning framework.

---

## Dataset
The dataset consists of images of 10 animals: dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, and squirrel.  

---

## Setup Instructions
### Environment Setup
- The project is developed and run primarily on **Google Colab**.  
- Make sure to **enable GPU acceleration** in the Colab runtime to speed up training.

### Dataset Download
- Download the Animals-10 dataset from the official source or repository.  
- Upload the dataset to Google Colab or connect your **Google Drive** for easier access.

---

## Data Preparation

- Explore the dataset folder structure.  
- Split the dataset into **training**, **validation**, and **test** sets to evaluate model performance properly.  
- Apply **data augmentation** techniques (random flips, rotations) and **normalization** to improve model generalization.  
- Create **DataLoader** objects to efficiently load and batch the data during training and evaluation.

---

## Data Visualization

- Display sample images from each animal class to understand the dataset visually.  
- Plot class distribution histograms to check for class imbalance or bias in the dataset.

---

## Modeling Approach

### Baseline Model

- Build a **basic CNN model from scratch** as a starting point to establish baseline accuracy.

### Model Optimization

- Perform **manual hyperparameter tuning** such as adjusting learning rates and model architectures.  
- Apply **automated hyperparameter optimization** using tools like **Optuna** to systematically find better model configurations.  
- Implement **learning rate reduction** strategies on plateaus to stabilize training and improve convergence.

### Transfer Learning

- Utilize **transfer learning** with pretrained architectures like **ResNet50** on the Animals-10 dataset to improve accuracy.

---

## Evaluation

- Measure model performance using metrics like **accuracy**, **precision**, **recall**, and **F1-score**
- Generate and analyze a **confusion matrix** to understand class-wise prediction performance and identify common misclassifications.  
- Analyze failure cases and errors to identify areas of improvement.
  
---
## Conclusions

- **Baseline CNN:**  
  - Accuracy: **74.5%**  
  - This model established a starting point but struggled with classes like **cat** and **cow**.  

- **Optimized CNN (with learning rate scheduler):**  
  - Accuracy: **83.7%**  
  - Demonstrated strong improvement, especially for **spider**, **chicken**, and **butterfly** classes.
  - Still lower recall in **cow** and **sheep**.  

- **Pretrained ResNet50:**  
  - Accuracy: **98%**  
  - Excellent, near-perfect performance across all classes.  
  - Even the weakest classes (**cow**, **sheep**) achieved **96% F1-score**.
 
Overall, the project progressed from a baseline CNN (74.5% accuracy) to an optimized CNN (83.7% accuracy), and finally to a pretrained ResNet50 model achieving 98% accuracy, showing consistent improvement across approaches.

---

## Dataset Reference

- [Animals-10 Dataset on Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

