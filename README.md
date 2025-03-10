model and video google drive link: https://drive.google.com/drive/folders/1uwqU8KUQjCtb1QQ7J0Sv5VgSlscTKqE4?usp=drive_link

# Diabetic Retinopathy Detection using Deep Learning

## Authors
**Hamza Asad**, **Umema Ashar (DroidMinds)**  
📧 **Emails:**  
1. Hamzaasad26@gmail.com  
2. I222036@nu.edu.pk  

---

## 1. Introduction
Diabetic Retinopathy (DR) is a severe complication of diabetes that affects the retina, potentially leading to vision loss. Early detection is crucial for timely medical intervention. This project was developed for the **Infyma AI Hackathon 25'**, where we built an AI-based model for DR detection using retinal fundus images.

---

## 2. Problem Statement
The task involves classifying retinal images into five DR severity levels:
- **No_DR (0):** No signs of DR
- **Mild (1):** Early-stage DR
- **Moderate (2):** Noticeable blood vessel changes
- **Severe (3):** Increased risk of vision impairment
- **Proliferative_DR (4):** Advanced DR stage with significant risks

**Objective:** Develop a high-accuracy, computationally efficient model to assist in early diagnosis.

---

## 3. Dataset
**Source:** Kaggle - Diabetic Retinopathy Balanced Dataset

### Dataset Structure:
- **Train Set:** Used for model training
- **Validation Set:** Used for hyperparameter tuning
- **Test Set:** Used for final model evaluation

### Data Preprocessing:
- **Resizing:** Images resized to **224x224 pixels** (EfficientNet-B3 requirement)
- **Normalization:** Pixel values scaled to **[0,1]** using ImageNet mean & standard deviation
- **Augmentation:** Random rotation, horizontal flipping to enhance generalization

---

## 4. Model Selection
We selected **EfficientNet-B3** due to:
1. **Balance of Accuracy and Efficiency:** High classification accuracy while maintaining computational efficiency.
2. **Optimized Architecture:** Uses compound scaling, outperforming ResNet and VGG models.
3. **Proven Performance on Medical Imaging:** EfficientNet has shown strong results in medical classification tasks.

### Model Modifications:
- **Pretrained Weights:** Initialized using **ImageNet1K_V1 weights**
- **Classifier Modification:** Final fully connected layer changed to match the 5 DR severity classes

---

## 5. Training Methodology

### Phase 1: Initial Training
- **Optimizer:** Adam (`learning rate = 0.001`)
- **Loss Function:** Cross-Entropy Loss
- **Epochs:** 30
- **Batch Size:** 32
- **Training on GPU (if available)**
- **Performance Tracking:** Accuracy and Loss recorded per epoch

### Phase 2: Fine-Tuning
- **Layer Freezing:** Early layers frozen, only last 10 layers and classifier trained
- **Lower Learning Rate:** `0.0001` to refine features
- **Additional Augmentations:** To improve generalization
- **Model Saving:** Best model checkpoint stored

---

## 6. Evaluation and Results
Final evaluation performed on the **test set**:
- **Overall Accuracy:** ~62%
- **F1-Score, Precision, Recall:** Analyzed per class
- **Confusion Matrix:** Visualized class-wise predictions
- **Explainability:** **Grad-CAM** used for model interpretability

---

## 7. Conclusion
The **EfficientNet-B3-based DR detection model** successfully achieved high classification accuracy while maintaining computational efficiency. Future work could explore:
- **Using Vision Transformers (ViTs)** for improved feature extraction
- **Semi-supervised learning** to leverage unlabeled data
- **Model deployment** via Flask/FastAPI for real-time diagnosis

**Trained model and results are available in the GitHub repository**, along with detailed code documentation.

---

## Hackathon Submission Includes:
- 🐍 **Python Script**
- 📂 **Model Weights (.pt file) [Google Drive Link]**
- 📄 **Detailed Report (this document)**
- 🎥 **Demo Video [Google Drive Link]**

---

### 🔗 **Repository Structure**
```bash
📂 Diabetic-Retinopathy-Detection
├── 📁 notebooks
│   ├── full_acc.ipynb
│   ├── s_model.py
├── README.md

```
📌 **Installation & Usage:**
```bash
# Clone the repository
git clone https://github.com/your-repo/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection

# Install dependencies
pip install torch torchvision tqdm matplotlib numpy scikit-learn seaborn


```

📢 **For deployment, refer to deployment.md in the repository.**

---

🚀 **Developed by DroidMinds | Infyma AI Hackathon 25'**
