# Brain-Hemorrhage-Classification-DeepLearning

##  Overview  
This repository accompanies the work described in the paper *“Brain Hemorrhage Classification Using Deep Learning Techniques”* submitted to JETIR (April 2024, Vol. 11, Issue 4). The goal of the project is to classify brain hemorrhage types (Bleed, Calcified, Bleed/Calcified) from MRI scan images using various deep-learning models (CNN, U-Net Autoencoder, Transfer-learning with VGG-19). :contentReference[oaicite:0]{index=0}

##  Motivation  
- Brain hemorrhage is a critical medical emergency — early detection and classification can be life-saving. :contentReference[oaicite:1]{index=1}  
- Traditional diagnosis requires expert radiologists and considerable time; an automated model can assist doctors for quicker, more reliable detection — especially when data is limited. :contentReference[oaicite:2]{index=2}  
- Among multiple DL approaches, this work compares simple CNN, U-Net Autoencoder (for segmentation + classification), and transfer-learning (VGG-19) to evaluate efficacy on a modest-sized MRI dataset. :contentReference[oaicite:3]{index=3}

##  Dataset & Preprocessing  
- The dataset consists of **MRI brain scan images** (initially 397 images, with a subset of 368 used). :contentReference[oaicite:4]{index=4}  
- Preprocessing steps:  
  1. Resize images to 256 × 256 pixels. :contentReference[oaicite:5]{index=5}  
  2. Noise removal using median filter. :contentReference[oaicite:6]{index=6}  
  3. Data augmentation: flipping (horizontal/vertical), rescaling, shearing, zooming, rotation, etc. to artificially increase training set size. :contentReference[oaicite:7]{index=7}  
  4. Dataset split: ~90% for training, ~10% for testing. :contentReference[oaicite:8]{index=8}  

##  Methods / Models  

| Model | Description |
|-------|-------------|
| **CNN** | A basic convolutional neural network — standard conv + pooling + fully connected layers for classification. :contentReference[oaicite:9]{index=9} |
| **U-Net Autoencoder** | A segmentation-based architecture: encode input via conv layers + pooling, then decode + upsample to reconstruct, followed by classification. Useful when annotated data is scarce. :contentReference[oaicite:10]{index=10} |
| **Transfer-Learning (VGG-19)** | Using pretrained VGG-19 (on ImageNet) and fine-tuning / feature-extraction for hemorrhage classification from MRI images. :contentReference[oaicite:11]{index=11} |

Implementation is done in Python (tested with Python 3.10.11) and uses a deep-learning framework (e.g. TensorFlow or PyTorch, as per authors’ environment) to build and train the models. :contentReference[oaicite:12]{index=12}

## ✅ Results  

- **U-Net Autoencoder**: ~85% accuracy on test set. :contentReference[oaicite:13]{index=13}  
- **VGG-19 (Transfer Learning)**: ~80% accuracy. :contentReference[oaicite:14]{index=14}  
- **Simple CNN**: ~60% accuracy. :contentReference[oaicite:15]{index=15}  

The U-Net based model outperformed the others, suggesting segmentation + autoencoder workflow helps when dataset is small.

##  How to Use / Reproduce  

1. Clone this repository.  
2. Ensure you have Python 3.10 (or compatible) + required DL libraries (TensorFlow / PyTorch, plus any dependencies like OpenCV, NumPy, etc.).  
3. Place the MRI image dataset in a folder (e.g. `data/`).  
4. Run the preprocessing script to resize, denoise, and augment images.  
5. Run training scripts for the desired model (CNN / U-Net / VGG-19).  
6. Evaluate using test split and inspect metrics (accuracy, precision, recall, F1-score, confusion matrix).  

##  Insights & Conclusion  

- For limited medical datasets, segmentation-based architectures (U-Net Autoencoder) yield better classification performance vs naive CNN or even transfer learning.  
- Augmentation + preprocessing is essential to compensate for small dataset size.  
- There is potential to improve with larger datasets — increasing training data volume could boost model generalization and reliability for real-world diagnostic support. :contentReference[oaicite:16]{index=16}  

##  Future Work  

As proposed in the original paper:  
- Collect a larger and more diverse MRI image dataset for better generalization. :contentReference[oaicite:17]{index=17}  
- Explore more advanced architectures, ensemble models, or hybrid models to improve classification performance and reduce false negatives/positives.  

##  Reference  

Kulkarni, N. G., Itagi, A., Shirol, P., Chavhan, S., & Yaragatti, V. (2024). *Brain Hemorrhage Classification Using Deep Learning Techniques*. JETIR, April 2024, Volume 11, Issue 4. :contentReference[oaicite:18]{index=18}  

##  Contact / Authors  

- Original Authors: Nita G. Kulkarni, Abhay Itagi, Pavankumar Shirol, Shashidhar Chavhan, Vijay Yaragatti. :contentReference[oaicite:19]{index=19}  
- Affiliation: Department of Computer Science & Engineering, SDM College of Engineering and Technology, Dharwad, Karnataka, India. :contentReference[oaicite:20]{index=20}  

---

 
