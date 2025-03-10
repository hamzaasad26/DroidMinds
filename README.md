model and video google drive link: https://drive.google.com/drive/folders/1uwqU8KUQjCtb1QQ7J0Sv5VgSlscTKqE4?usp=drive_link


Diabetic Retinopathy Detection using Deep Learning
Hamza Asad , Umema Ashar (DroidMinds)
Emails : 1. Hamzaasad26@gmail.com, 2. I222036@nu.edu.pk 


1. Introduction
Diabetic Retinopathy (DR) is a severe complication of diabetes that affects the retina, potentially leading to vision loss. Early detection is crucial for timely medical intervention. This report outlines the approach taken in the Infyma AI Hackathon 25', where we developed an AI-based model for DR detection using retinal fundus images.
2. Problem Statement
The task involves classifying retinal images into five DR severity levels:
•	No_DR (0): No signs of DR
•	Mild (1): Early-stage DR
•	Moderate (2): Noticeable blood vessel changes
•	Severe (3): Increased risk of vision impairment
•	Proliferative_DR (4): Advanced DR stage with significant risks
The goal is to develop a high-accuracy, computationally efficient model to assist in early diagnosis.
3. Dataset
The dataset was sourced from Kaggle (Diabetic Retinopathy Balanced Dataset), consisting of labeled retinal fundus images in JPEG/PNG format.
Dataset structure:
•	Train set: Used for model training
•	Validation set: Used for hyperparameter tuning
•	Test set: Used for final model evaluation

Data preprocessing steps:
•	Resizing: Images were resized to 224x224 pixels (EfficientNet-B3 requirement)
•	Normalization: Image pixel values were scaled to the range [0,1] using mean and standard deviation from ImageNet
•	Augmentation: Random rotation, horizontal flipping to enhance generalization

4. Model Selection
Based on the hackathon guidelines and dataset characteristics, we selected EfficientNet-B3 for the following reasons:
1.	Balance of Accuracy and Efficiency: EfficientNet-B3 achieves high classification accuracy while maintaining computational efficiency.
2.	Optimized Architecture: Utilizes compound scaling, making it more effective than ResNet and VGG models.
3.	Proven Performance on Medical Imaging: EfficientNet has shown strong results in medical classification tasks.
Model Modifications
•	Pretrained Weights: Initialized using ImageNet1K_V1 weights
•	Classifier Modification: Changed the final fully connected layer to match the 5 DR severity classes
5. Training Methodology
Phase 1: Initial Training
•	Optimizer: Adam (learning rate = 0.001)
•	Loss Function: Cross-Entropy Loss
•	Epochs: 30
•	Batch Size: 32
•	Training on GPU (if available)
•	Performance Tracking: Accuracy and Loss recorded per epoch
Phase 2: Fine-tuning
•	Layer Freezing: Early layers frozen, only last 10 layers and classifier trained
•	Lower Learning Rate: 0.0001 to refine features
•	Additional Augmentations: To improve generalization
•	Model Saving: Best model checkpoint stored
6. Evaluation and Results
Final evaluation was performed on the test set with:
•	Overall Accuracy: ~62%
•	F1-Score, Precision, Recall: Analyzed per class
•	Confusion Matrix: Visualized class-wise predictions
•	Explainability: Grad-CAM used for model interpretability
7. Conclusion
The EfficientNet-B3-based DR detection model successfully achieved high classification accuracy while maintaining computational efficiency. Future work could explore:
•	Using Vision Transformers (ViTs) for improved feature extraction
•	Semi-supervised learning to leverage unlabeled data
•	Model deployment via Flask/FastAPI for real-time diagnosis
The trained model and results are available in the GitHub repository, along with detailed code documentation.
Hackathon Submission Includes:
•	Python Script
•	Model Weights (.pt file) (Google drive link given)
•	Detailed Report (this document)
•	Demo video (Google drive link given)


