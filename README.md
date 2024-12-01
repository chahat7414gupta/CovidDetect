# **CovAid: AI-Based COVID-19 Detection from Chest X-rays**

## **Overview**
**CovAid** is an AI-powered COVID-19 recognition system that leverages machine learning (ML) techniques to classify chest X-ray images into two categories:  
- **Healthy**  
- **(COVID-19 Positive)**  

This system aims to assist healthcare professionals by providing a fast and reliable diagnostic tool for detecting COVID-19 infections using radiological imaging.

---

## **Objective**
The primary objective of CovAid is to aid in the detection of COVID-19 infections through chest X-ray images using a machine learning model, thereby accelerating the diagnostic process in healthcare systems.  

- Enable real-time COVID-19 detection with an accuracy rate of **90-95%**.  
- Provide an easy-to-use web interface where users can upload chest X-rays for diagnosis.  
- Support healthcare professionals and authorities in pandemic response efforts by integrating AI-based systems into medical infrastructures.  

---

## **Features**
- Automated classification of chest X-rays as **Healthy** or **COVID-19 Positive**.  
- Real-time predictions with a simple and intuitive web interface.  
- High accuracy and fast inference time to assist healthcare professionals.  
- Portable and scalable AI-based solution for COVID-19 detection.  

---

## **Workflow**

### **1. Data Collection and Preparation**
1. **Dataset:**
   - Utilize publicly available datasets such as the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).
   - Ensure the dataset includes two categories:
     - **Healthy (Normal X-rays)**
     - **Pneumonia (COVID-19 Positive)**

2. **Data Preprocessing:**
   - Normalize the images: Scale pixel values to the range `[0, 1]`.  
   - Resize all images to a consistent dimension, e.g., \( 224 \times 224 \), for model compatibility.  
   - Perform data augmentation (rotation, flipping, zooming) to increase variability and address class imbalance.  

---

### **2. Model Development**
1. **Architecture:**
   - Build a Convolutional Neural Network (CNN) or use **transfer learning** with pre-trained models such as:
     - **ResNet50**
     - **VGG16**
     - **EfficientNet**

2. **Implementation:**
   - Include convolutional, pooling, and dense layers in the model architecture.  
   - Use **ReLU activation** for hidden layers and **Softmax activation** for the output layer.  
   - Compile the model with:
     - **Loss Function**: Categorical Cross-Entropy  
     - **Optimizer**: Adam  
     - **Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC  

---

### **3. Training the Model**
1. **Dataset Splitting:**
   - Split the dataset into:
     - **Training Set**: 70%
     - **Validation Set**: 20%
     - **Testing Set**: 10%

2. **Training Process:**
   - Train the model for **50 epochs** with a batch size of **32**.  
   - Use a checkpoint callback to save the best-performing model based on validation loss.  

3. **Visualization:**
   - Plot training and validation accuracy/loss using matplotlib.  

---

### **4. Evaluation**
Evaluate the model on the test set and compute the following metrics:  
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **ROC-AUC Curve**

---

### **5. Deployment**
1. **Model Deployment:**
   - Save the trained model in `.h5` format.  
   - Create a web application using **Flask** or **FastAPI** for real-time predictions.  

2. **Web Interface:**
   - Build a simple UI where users can upload chest X-rays.  
   - Display the result (Healthy or COVID-19 Positive) along with confidence scores.  

---

## **System Requirements**
- Python 3.8 or above  
- TensorFlow/Keras  
- NumPy  
- Matplotlib  
- Flask or FastAPI  

---

## **How to Run the Project**
1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-repo/CovAid.git](https://github.com/chahat7414gupta/CovidDetect)
   cd CovAid
