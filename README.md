### **BrainWasher2024 - Project Workflow Summary**  
The **BrainWasher2024** project implements a **machine unlearning framework** based on **ResNet-50** for **face recognition**. The primary goal is to allow a deep learning model to **"forget" specific data** while retaining general knowledge, which is useful for **privacy compliance and adaptive learning**.  

---

# **📌 Workflow of the Project**
The project consists of several key steps:  

## **1️⃣ Dataset Preparation**
- **Files involved**:  
  - `data_loader.py`
  - `unlearner_data_loader.py`
  - `casia_csv_maker.py`
  - `write_csv_for_making_dataset.py`

### **Workflow**:
1. **Data is loaded using PyTorch’s `DataLoader`** from CSV-based image datasets.  
2. **Preprocessing is applied**, including resizing and normalizing the images.  
3. **Dataset is split** into:
   - **Forget Dataset** (data the model must forget)
   - **Retain Dataset** (data that should not be forgotten)
   - **Validation Dataset** (used to test the model's performance).  

---

## **2️⃣ Model Architecture**
- **Files involved**:  
  - `models.py`
  - `FaceNetModel` (ResNet-50 based)
  - `utils_inceptionresnetv2.py`  

### **Workflow**:
1. **ResNet-50 is used as a feature extractor**, removing its original classification head.  
2. **Fully connected layers (FC layers) are modified** to generate **512-dimensional embeddings** instead of classifying into ImageNet classes.  
3. **Face embeddings are L2 normalized** to ensure better similarity comparison.  
4. **A classifier head (linear layer) is added**, mapping embeddings to **105 classes (for face recognition)**.  

### **Model Used:**
✅ **FaceNet-style architecture based on ResNet-50**  
✅ **Feature extractor:** Pretrained **ResNet-50 backbone**  
✅ **Embedding layer:** Converts deep features to **512D L2-normalized vectors**  
✅ **Classifier head:** Maps embeddings to **105 classes**  

---

## **3️⃣ Model Training**
- **Files involved**:
  - `train.py`
  - `train_fc.py`
  - `train_retain.py`

### **Workflow**:
1. **Training is done in two stages**:
   - **Train the feature extractor** (ResNet-50) with **triplet loss** for better separation between face embeddings.
   - **Train the classification head** to map embeddings to face classes.

2. **Triplet Loss is used** for training:
   - **Encourages similar faces to be closer together** in embedding space.
   - **Encourages different faces to be farther apart**.

3. **Cross-entropy loss is used** for classification.

4. **Optimizer used**: **SGD (Stochastic Gradient Descent) with momentum**.
5. **Learning rate scheduling**:
   - **CosineAnnealingLR**: Gradually reduces learning rate for stable convergence.
   - **StepLR**: Reduces learning rate at specific epochs.

---

## **4️⃣ Unlearning Process**
- **Files involved**:  
  - `BrainWasher_algorithm.py`
  - `unlearner.py`
  - `unlearner_data_loader.py`

### **Workflow**:
The **unlearning process is executed in two stages**:

### **✅ Stage 1: Forgetting Stage**
- Uses **KL Divergence Loss** to **force the model's output distribution to be uniform** for the "forget" dataset.  
- **Why?**  
  - If the model produces **uniform probabilities**, it means it has "forgotten" any meaningful information about the data.  

#### **Code Snippet**
```python
uniform_label = torch.ones_like(outputs).to(DEVICE) / outputs.shape[1]
loss = self.kl_loss_sym(outputs, uniform_label)
loss.backward()
```

### **✅ Stage 2: Retaining Stage**
- **Contrastive learning loss** ensures that the model **retains knowledge of non-forgotten data**.
- **Dot product similarity** between retained embeddings is maximized.
- **Forget embeddings are pushed away from retained embeddings**.

#### **Code Snippet**
```python
loss = (-1 * nn.LogSoftmax(dim=-1)(outputs_forget @ outputs_retain.T / t)).mean()
```

---

## **5️⃣ Model Evaluation**
- **Files involved**:  
  - `eval_metrics.py`
  - `accuracies.py`

### **Metrics Used:**
✅ **Accuracy** - Measures the **classification performance** before and after unlearning.  
✅ **Loss Reduction** - Tracks the **change in KL divergence loss** to confirm forgetting.  
✅ **Cosine Similarity** - Ensures that similar face embeddings remain close.  
✅ **F1-score, Precision, Recall** - Evaluates model generalization after forgetting. 

---

# **📌 Final Summary**
## **✅ What Does This Project Do?**
1. **Trains a Face Recognition Model** based on **ResNet-50**.  
2. **Uses triplet loss for embedding learning** and **cross-entropy loss for classification**.  
3. **Implements unlearning** to remove specific user data while retaining other knowledge.  
4. **Uses KL Divergence Loss and Contrastive Learning** for controlled forgetting.  
5. **Evaluates the model before and after unlearning** to track performance changes.  

## **✅ Models Used**
- **ResNet-50** as the backbone.
- **Fully connected layers modified** to produce **512D face embeddings**.
- **Classifier head added** to predict **105 face classes**.

## **✅ How is the Model Trained?**
1. **Initial training using triplet loss** (FaceNet-style).
2. **Fine-tuning classification layers** with cross-entropy loss.
3. **SGD optimizer with momentum** + **CosineAnnealingLR scheduler**.

## **✅ What Metrics Are Used for Evaluation?**
- **Accuracy** (before and after unlearning).
- **KL Divergence loss** (to check if knowledge was removed).
- **Cosine Similarity** (for embedding consistency).
- **Precision, Recall, F1-score** (for general performance tracking).

---

# **🚀 Conclusion**
The **BrainWasher2024** project is an **advanced face recognition system** that integrates **machine unlearning** techniques. It enables **controlled forgetting of specific data** while ensuring **robust model performance**. The model is trained with **triplet loss, cross-entropy loss, and contrastive loss**, leveraging **ResNet-50 embeddings**. Evaluation is done using **accuracy, cosine similarity, and KL divergence loss tracking**.
