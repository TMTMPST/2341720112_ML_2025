# 🤖 Machine Learning

Repository ini berisi kumpulan praktikum, tugas, dan proyek machine learning untuk mata kuliah Machine Learning di tingkat perguruan tinggi.

## 🎯 Tujuan Repository

Repository ini dibuat untuk:

- 📚 Dokumentasi pembelajaran machine learning
- 🔬 Eksperimen dengan berbagai algoritma ML
- 📊 Analisis dataset klasik dan real-world
- 🏗️ Implementasi proyek machine learning end-to-end

## 📁 Struktur Repository

```
Machine-Learning/
├── JS01/                                          # Etika dan Dampak AI
│   ├── JS01_2341720112_Vidi-Joshubzky-Saviola.ipynb
│   └── README.md
├── JS02/                                          # Data Preprocessing Fundamentals
│   ├── JS02_Praktikum1.ipynb                     # Pengenalan Data
│   ├── JS02.Praktikum2.ipynb                     # Data Cleaning
│   ├── JS02.Praktikum3.ipynb                     # Data Transformation
│   ├── JS02_Praktikum4.ipynb                     # Feature Engineering
│   ├── JS02_2341720112_Vidi-Joshubzky-Saviola.ipynb # Tugas Utama
│   └── README.md
├── JS03/                                          # Machine Learning Pipeline
│   ├── Praktikum1.ipynb                          # Pengenalan ML
│   ├── Praktikum2.ipynb                          # Data Preprocessing
│   ├── Praktikum3.ipynb                          # Exploratory Data Analysis
│   ├── Praktikum4.ipynb                          # Feature Engineering
│   ├── TP.ipynb                                  # Tugas Praktikum
│   ├── data/                                     # Dataset folder
│   │   ├── wbc.csv                               # Wisconsin Breast Cancer
│   │   ├── Titanic-Dataset*.csv                  # Titanic datasets
│   │   └── Lenna.png                             # Image dataset
│   └── README.md
├── JS04/                                          # Clustering - K-Means
│   ├── P1_JS04.ipynb                             # K-Means Basics
│   ├── P2_JS04.ipynb                             # Elbow Method
│   ├── P3_JS04.ipynb                             # Silhouette Score
│   ├── TP_JS04.ipynb                             # Tugas Praktikum
│   ├── Kmeans.ipynb                              # K-Means Complete
│   └── data/
│       ├── Iris.csv
│       └── Mall_Customers.csv
├── JS05/                                          # Clustering - HDBSCAN
│   ├── P1_JS05.ipynb                             # HDBSCAN Introduction
│   ├── P2_JS05.ipynb                             # HDBSCAN Advanced
│   └── TP_JS05.ipynb                             # Tugas Praktikum
├── JS06/                                          # Regression - Linear & Polynomial
│   ├── P1_JS06.ipynb                             # Linear Regression
│   ├── P2_JS06.ipynb                             # Polynomial Regression
│   ├── TP_JS06.ipynb                             # Tugas Praktikum
│   ├── dataset.csv
│   ├── insurance.csv
│   └── Posisi_gaji.csv
├── JS07/                                          # Classification - Multiple Algorithms
│   ├── P1_JS07.ipynb                             # Logistic Regression
│   ├── P2_JS07.ipynb                             # K-Nearest Neighbors
│   ├── P3_JS07.ipynb                             # Support Vector Machine
│   ├── P4_JS07.ipynb                             # Decision Tree
│   ├── P5_JS07.ipynb                             # Random Forest
│   ├── P6_JS07.ipynb                             # Naive Bayes
│   └── TP_JS07.ipynb                             # Tugas Praktikum
├── JS08/                                          # Ensemble Learning
│   └── TP_JS08.ipynb                             # Ensemble Methods
├── JS09/                                          # Advanced Classification
│   ├── P1_JS09.ipynb                             # Classification Metrics
│   ├── P2_JS09.ipynb                             # Model Evaluation
│   ├── P3_JS09.ipynb                             # Hyperparameter Tuning
│   ├── TP1_JS09.ipynb                            # Tugas Praktikum 1
│   ├── TP2_JS09.ipynb                            # Tugas Praktikum 2
│   ├── iris.csv
│   ├── spam.csv
│   └── voice.csv
├── JS11/                                          # Neural Networks Basics
│   ├── P1_JS11.ipynb                             # Perceptron
│   ├── P2_JS11.ipynb                             # Multi-Layer Perceptron
│   ├── P3_JS11.ipynb                             # Backpropagation
│   ├── P4_JS11.ipynb                             # Activation Functions
│   ├── P5_JS11.ipynb                             # Neural Network Training
│   └── TP_JS11.ipynb                             # Tugas Praktikum
├── JS13/                                          # Deep Learning with Keras/TensorFlow
│   ├── P1_JS13.ipynb                             # Keras Introduction
│   ├── P2_JS13.ipynb                             # Deep Neural Networks
│   ├── P3_JS13.ipynb                             # Model Optimization
│   └── TP_JS13.ipynb                             # Tugas Praktikum
├── JS14/                                          # Convolutional Neural Networks (CNN)
│   ├── P1_JS14.ipynb                             # CNN Basics
│   ├── P2_JS14.ipynb                             # CIFAR-10 Classification
│   ├── TP_JS14.ipynb                             # MNIST Classification
│   └── dataset/
│       ├── single_prediction/
│       ├── test_set/
│       │   ├── cats/
│       │   └── dogs/
│       └── training_set/
│           ├── cats/
│           └── dogs/
├── JS15/                                          # ML Pipeline & Deployment
│   ├── P1_JS15.ipynb                             # Day/Night Classifier Training
│   ├── P2_JS15/                                  # Deployment Project
│   │   ├── README.MD                             # Deployment Documentation
│   │   └── daynight-project/
│   │       ├── daynight-classifier-Vidi/
│   │       │   ├── app.py                        # Flask Application
│   │       │   ├── Dockerfile                    # Docker Configuration
│   │       │   ├── requirements.txt              # Python Dependencies
│   │       │   ├── README.md
│   │       │   └── model/
│   │       │       └── day_night_model.h5
│   │       └── IMG/                              # Screenshots
│   ├── TP_JS15.ipynb                             # Tugas Praktikum
│   ├── dataset/                                  # Training Images
│   └── model/                                    # Saved Models
├── Journal/                                       # Experimental Notebooks
│   ├── db_sportify.ipynb                         # Spotify Clustering DBSCAN
│   ├── KMeans_Clustering.ipynb                   # K-Means Experiments
│   ├── kmeans_sportify*.ipynb                    # Spotify K-Means Variations
│   ├── data/
│   ├── outputs/                                  # K-Means Results
│   ├── outputs_v2_dbscan/                        # DBSCAN Results
│   └── outputs_v2_spherical*/                    # Spherical K-Means Results
├── UTS/                                           # Mid-Term Projects
│   ├── TUGAS2_CreditCard_Clustering_ANN.ipynb
│   ├── TUGAS2_CreditCard_Clustering_ANN2.ipynb
│   ├── TUGAS2_CreditCard_Clustering_ANN3_detailed.ipynb
│   └── data/
│       └── CC GENERAL.csv
├── KMeans_Song_Mood_Clustering.ipynb              # Main Project
├── venv/                                          # Virtual environment Python
├── .git/                                         # Git version control
└── README.md                                     # File dokumentasi ini
```

## 🔬 Topik yang Dipelajari

### JS01 - Etika dan Dampak Kecerdasan Buatan

- **Etika AI**: Bias, privasi, transparansi, akuntabilitas
- **Aspek Hukum**: Regulasi, hak cipta, tanggung jawab hukum
- **Dampak Lingkungan**: Carbon footprint, green AI, sustainable computing
- **Kasus Nyata**: Deepfake, pelanggaran hak cipta, penggunaan AI dalam hukum

### JS02 - Data Preprocessing dan ML Fundamentals

- **Data Understanding**: Eksplorasi dan analisis dataset
- **Data Cleaning**: Handling missing values, outlier detection
- **Data Transformation**: Encoding, normalization, standardization
- **Feature Engineering**: Selection, creation, dimensionality reduction
- **Preprocessing Pipeline**: End-to-end data preparation

### JS03 - Machine Learning Pipeline

- **Data Preprocessing**: Cleaning, transformation, encoding
- **Exploratory Data Analysis (EDA)**: Statistical analysis, visualization
- **Feature Engineering**: Selection, extraction, transformation
- **Model Evaluation**: Cross-validation, metrics, performance analysis

### JS04 - Clustering dengan K-Means

- **K-Means Algorithm**: Centroid-based clustering
- **Elbow Method**: Determining optimal number of clusters
- **Silhouette Score**: Cluster quality evaluation
- **Dataset**: Iris, Mall Customers
- **Applications**: Customer segmentation, pattern recognition

### JS05 - Clustering dengan HDBSCAN

- **HDBSCAN Algorithm**: Hierarchical density-based clustering
- **Hyperparameter Tuning**: min_cluster_size, min_samples, cut_distance
- **Noise Detection**: Identifying outliers automatically
- **Comparison**: HDBSCAN vs DBSCAN vs K-Means
- **Applications**: Handling complex data structures

### JS06 - Regression Analysis

- **Linear Regression**: Simple and multiple regression
- **Polynomial Regression**: Non-linear relationships
- **Model Evaluation**: R², MSE, RMSE, MAE
- **Dataset**: Insurance, Salary prediction
- **Applications**: Price prediction, trend analysis

### JS07 - Classification Algorithms

- **Logistic Regression**: Binary and multiclass classification
- **K-Nearest Neighbors (KNN)**: Distance-based classification
- **Support Vector Machine (SVM)**: Kernel methods, margin optimization
- **Decision Tree**: Tree-based classification
- **Random Forest**: Ensemble of decision trees
- **Naive Bayes**: Probabilistic classification
- **Applications**: Spam detection, medical diagnosis

### JS08 - Ensemble Learning

- **Bagging**: Bootstrap aggregating
- **Boosting**: Adaptive boosting (AdaBoost, Gradient Boosting)
- **Stacking**: Meta-learning approaches
- **Voting Classifiers**: Hard and soft voting
- **Applications**: Improving model accuracy and robustness

### JS09 - Advanced Classification & Model Evaluation

- **Classification Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Error analysis
- **ROC Curve & AUC**: Model performance visualization
- **Cross-Validation**: K-fold, stratified
- **Hyperparameter Tuning**: Grid search, random search
- **Dataset**: Iris, Spam detection, Voice classification

### JS11 - Neural Networks Fundamentals

- **Perceptron**: Basic neural unit
- **Multi-Layer Perceptron (MLP)**: Feedforward networks
- **Backpropagation**: Gradient descent, weight updates
- **Activation Functions**: Sigmoid, ReLU, tanh, softmax
- **Training Techniques**: Learning rate, epochs, batch size
- **Applications**: Pattern recognition, function approximation

### JS13 - Deep Learning dengan Keras/TensorFlow

- **Keras API**: Sequential and Functional models
- **Deep Neural Networks**: Multiple hidden layers
- **Regularization**: Dropout, L1/L2, batch normalization
- **Optimization**: Adam, SGD, RMSprop
- **Model Callbacks**: Early stopping, model checkpointing
- **Applications**: Complex pattern recognition

### JS14 - Convolutional Neural Networks (CNN)

- **CNN Architecture**: Convolutional layers, pooling, flatten
- **Image Classification**: CIFAR-10 (10 classes), MNIST (digits)
- **Data Augmentation**: Rotation, flip, zoom
- **Transfer Learning**: Pre-trained models
- **Model Optimization**: Achieving >80% accuracy
- **Applications**: Image recognition, computer vision

### JS15 - ML Pipeline & Deployment

- **Feature Extraction**: HOG (Histogram of Oriented Gradients)
- **Model Training**: Day vs Night classifier
- **Flask Application**: REST API for predictions
- **Docker Containerization**: Reproducible environments
- **Cloud Deployment**: Hugging Face Spaces
- **MLOps**: Version control, CI/CD for ML models
- **Applications**: Production-ready ML systems

## 📊 Dataset yang Digunakan

| Dataset                 | Deskripsi                             | Tipe                  | Jumlah Data | Sesi         |
| ----------------------- | ------------------------------------- | --------------------- | ----------- | ------------ |
| Wisconsin Breast Cancer | Klasifikasi diagnosis kanker payudara | Binary Classification | 569         | JS02, JS03   |
| Titanic Dataset         | Prediksi kelangsungan hidup penumpang | Binary Classification | ~800        | JS03         |
| Lenna Image             | Pengolahan citra digital standar      | Image Processing      | 1 image     | JS03         |
| AI Ethics Cases         | Kasus nyata pelanggaran etika AI      | Text Analysis         | Multiple    | JS01         |
| Iris Dataset            | Klasifikasi spesies bunga iris        | Multiclass/Clustering | 150         | JS04, JS09   |
| Mall Customers          | Segmentasi pelanggan berdasarkan data | Clustering            | 200         | JS04         |
| Insurance Dataset       | Prediksi biaya asuransi kesehatan     | Regression            | ~1300       | JS06         |
| Salary/Position Dataset | Prediksi gaji berdasarkan posisi      | Regression            | Variable    | JS06         |
| Spam Dataset            | Klasifikasi email spam                | Binary Classification | ~5000       | JS09         |
| Voice Dataset           | Klasifikasi gender dari suara         | Binary Classification | ~3000       | JS09         |
| CIFAR-10                | Klasifikasi gambar 10 kategori        | Multiclass (CNN)      | 60000       | JS14         |
| MNIST                   | Klasifikasi digit tulisan tangan      | Multiclass (CNN)      | 70000       | JS14         |
| Cats vs Dogs            | Klasifikasi gambar kucing dan anjing  | Binary (CNN)          | ~25000      | JS14         |
| Day vs Night Images     | Klasifikasi gambar siang dan malam    | Binary Classification | Custom      | JS15         |
| Credit Card Dataset     | Clustering kartu kredit               | Clustering            | ~9000       | UTS          |
| Spotify Dataset         | Clustering mood lagu                  | Clustering            | ~10000      | Journal/Main |

## 🛠️ Teknologi dan Tools

### Programming Language

- **Python 3.x**: Bahasa pemrograman utama

### Libraries & Frameworks

```python
# Data Manipulation & Analysis
pandas              # Data manipulation
numpy               # Numerical computing

# Visualization
matplotlib          # Basic plotting
seaborn            # Statistical visualization
plotly             # Interactive plots

# Machine Learning - Classical
scikit-learn       # ML algorithms & tools
scipy              # Scientific computing
hdbscan            # Density-based clustering

# Deep Learning
tensorflow         # Deep learning framework
keras              # High-level neural networks API

# Computer Vision
opencv-python      # Computer vision
PIL/Pillow         # Image manipulation
scikit-image       # Image processing (HOG features)

# Web Framework & Deployment
flask              # Web application framework
gunicorn           # WSGI HTTP server
docker             # Containerization

# Signal Processing & Bioinformatics (JS01)
pyprep             # EEG preprocessing
wandb              # ML experiment tracking
pyecg              # ECG signal analysis

# Data Preprocessing
sklearn.preprocessing.LabelEncoder      # Categorical encoding
sklearn.preprocessing.StandardScaler    # Feature scaling
sklearn.preprocessing.MinMaxScaler      # Normalization

# Model Evaluation
sklearn.metrics    # Classification/regression metrics
sklearn.model_selection  # Cross-validation, train-test split

# Environment
google.colab.files # File handling (Colab)
jupyter            # Interactive development
ipython            # Enhanced Python shell
```

### Development Environment

- **Jupyter Notebook**: Interactive development
- **VS Code**: Code editor dengan Python extension
- **Google Colab**: Cloud-based notebook
- **Git**: Version control system
- **Docker**: Containerization platform
- **Hugging Face Spaces**: ML model deployment

## 💻 Environment Compatibility

| Sesi | Local Jupyter | Google Colab  | VS Code | Requirements                    |
| ---- | ------------- | ------------- | ------- | ------------------------------- |
| JS01 | ✅ Preferred  | ⚠️ Limited    | ✅ Yes  | pyprep, scipy, wandb            |
| JS02 | ✅ Yes        | ✅ Optimized  | ✅ Yes  | pandas, sklearn                 |
| JS03 | ✅ Preferred  | ✅ Compatible | ✅ Yes  | Full ML stack                   |
| JS04 | ✅ Yes        | ✅ Yes        | ✅ Yes  | sklearn, pandas, matplotlib     |
| JS05 | ✅ Yes        | ✅ Yes        | ✅ Yes  | hdbscan, sklearn                |
| JS06 | ✅ Yes        | ✅ Yes        | ✅ Yes  | sklearn, numpy, matplotlib      |
| JS07 | ✅ Yes        | ✅ Yes        | ✅ Yes  | sklearn (multiple algorithms)   |
| JS08 | ✅ Yes        | ✅ Yes        | ✅ Yes  | sklearn (ensemble methods)      |
| JS09 | ✅ Yes        | ✅ Yes        | ✅ Yes  | sklearn, pandas                 |
| JS11 | ✅ Yes        | ✅ Yes        | ✅ Yes  | numpy, matplotlib, sklearn      |
| JS13 | ✅ Yes        | ✅ Preferred  | ✅ Yes  | tensorflow, keras               |
| JS14 | ✅ Yes        | ✅ Preferred  | ✅ Yes  | tensorflow, keras, GPU optional |
| JS15 | ✅ Yes        | ⚠️ Partial    | ✅ Yes  | flask, docker, scikit-image     |

**Notes:**

- JS01: Beberapa library signal processing mungkin terbatas di Colab
- JS02: Dioptimalkan untuk Google Colab dengan file upload
- JS13-JS14: Google Colab menyediakan GPU gratis untuk training CNN
- JS15: Deployment memerlukan Docker dan Git (lokal/server)

## 🚀 Setup dan Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/TMTMPST/Machine-Learning.git
cd Machine-Learning
```

### 2. Setup Virtual Environment

```bash
# Membuat virtual environment
python -m venv venv

# Aktivasi virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Core packages untuk semua sesi
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# JS01 - Signal processing & ML tracking
pip install pyprep scipy wandb

# JS04 - Clustering
pip install scikit-learn matplotlib

# JS05 - HDBSCAN clustering
pip install hdbscan

# JS06-JS09 - Classical ML
pip install scikit-learn pandas numpy matplotlib seaborn

# JS11 - Neural Networks
pip install numpy matplotlib scikit-learn

# JS13-JS14 - Deep Learning (with GPU support)
pip install tensorflow[and-cuda]  # For GPU
# OR for CPU only:
pip install tensorflow

# JS15 - Deployment
pip install flask scikit-image opencv-python-headless tensorflow-cpu
pip install gunicorn  # For production server

# Optional: Untuk compatibility dengan Google Colab
pip install google-colab-tools

# Install semua sekaligus (recommended)
pip install pandas numpy matplotlib seaborn scikit-learn jupyter \
    tensorflow pillow opencv-python-headless hdbscan \
    flask scikit-image gunicorn
```

### 4. Menjalankan Jupyter Notebook

```bash
jupyter notebook
```

## 📚 Cara Penggunaan

### Urutan Pembelajaran yang Disarankan

1. **JS01 - Etika AI (Foundation)**

   - Mulai dengan memahami aspek etika dan dampak AI
   - Buka `JS01/JS01_2341720112_Vidi-Joshubzky-Saviola.ipynb`
   - Pelajari kasus-kasus nyata dan solusi Green AI

2. **JS02 - Data Preprocessing (Core Skills)**

   - Pelajari fundamental preprocessing data
   - Ikuti urutan: Praktikum1 → Praktikum2 → Praktikum3 → Praktikum4
   - Selesaikan tugas utama untuk hands-on experience

3. **JS03 - ML Pipeline (Application)**

   - Terapkan skills preprocessing untuk machine learning
   - Ikuti urutan: Praktikum1 → Praktikum2 → Praktikum3 → Praktikum4
   - Kerjakan `TP.ipynb` untuk Wisconsin Breast Cancer classification

4. **JS04 - Clustering (K-Means)**

   - Pelajari unsupervised learning dengan K-Means
   - Praktikum: P1 → P2 → P3 → TP
   - Eksperimen dengan Iris dan Mall Customers dataset

5. **JS05 - Advanced Clustering (HDBSCAN)**

   - Pelajari density-based clustering
   - Bandingkan HDBSCAN vs DBSCAN vs K-Means
   - Praktikum: P1 → P2 → TP

6. **JS06 - Regression**

   - Linear dan Polynomial Regression
   - Praktikum: P1 (Linear) → P2 (Polynomial) → TP
   - Dataset: Insurance, Salary prediction

7. **JS07 - Classification Algorithms**

   - Pelajari 6 algoritma klasifikasi
   - Praktikum: P1 (Logistic) → P2 (KNN) → P3 (SVM) → P4 (Decision Tree) → P5 (Random Forest) → P6 (Naive Bayes) → TP
   - Bandingkan performa setiap algoritma

8. **JS08 - Ensemble Learning**

   - Pelajari teknik ensemble untuk meningkatkan akurasi
   - Fokus pada `TP_JS08.ipynb`

9. **JS09 - Model Evaluation & Tuning**

   - Metrics, confusion matrix, ROC curve
   - Hyperparameter tuning
   - Praktikum: P1 → P2 → P3 → TP1 → TP2

10. **JS11 - Neural Networks Basics**

    - Perceptron, MLP, Backpropagation
    - Praktikum: P1 → P2 → P3 → P4 → P5 → TP

11. **JS13 - Deep Learning**

    - Keras/TensorFlow introduction
    - Deep neural networks
    - Praktikum: P1 → P2 → P3 → TP

12. **JS14 - Convolutional Neural Networks**

    - CNN untuk image classification
    - CIFAR-10 dan MNIST
    - Praktikum: P1 → P2 → TP

13. **JS15 - Deployment**
    - End-to-end ML pipeline
    - Training → Flask API → Docker → Deployment
    - Praktikum: P1 (Training) → P2 (Deployment) → TP

### Setup per Sesi

#### JS01 Setup

```bash
pip install pyprep scipy wandb
# Note: pyecg mungkin memerlukan versi Python tertentu
```

#### JS02-JS03 Setup

```bash
pip install pandas scikit-learn matplotlib seaborn
```

#### JS04-JS05 Setup

```bash
pip install scikit-learn matplotlib hdbscan
```

#### JS06-JS09 Setup

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

#### JS11 Setup

```bash
pip install numpy matplotlib scikit-learn
```

#### JS13-JS14 Setup

```bash
pip install tensorflow[and-cuda]  # GPU support
# OR
pip install tensorflow  # CPU only
```

#### JS15 Setup

```bash
pip install flask scikit-image opencv-python-headless tensorflow-cpu gunicorn
```

## 📈 Progress Pembelajaran

### JS01 - Etika dan Dampak AI

- [x] Setup environment (PyPREP, SciPy, wandb)
- [x] Analisis kasus pelanggaran etika AI
- [x] Evaluasi dampak lingkungan AI
- [x] Penelitian solusi Green AI
- [x] Diskusi regulasi dan aspek hukum AI

### JS02 - Data Preprocessing Fundamentals

- [x] Praktikum 1: Pengenalan dan eksplorasi data
- [x] Praktikum 2: Data cleaning dan handling missing values
- [x] Praktikum 3: Data transformation dan encoding
- [x] Praktikum 4: Feature engineering dan selection
- [x] Tugas Utama: Complete preprocessing pipeline
  - [x] Data loading dan eksplorasi
  - [x] Feature selection (drop unuseful columns)
  - [x] Label encoding untuk target variable
  - [x] Standardization untuk numerical features

### JS03 - Machine Learning Pipeline

- [x] Setup environment dan dependencies
- [x] Praktikum 1: Pengenalan Machine Learning
- [x] Praktikum 2: Data Preprocessing
- [x] Praktikum 3: Exploratory Data Analysis
- [x] Praktikum 4: Feature Engineering
- [x] Tugas Praktikum: Wisconsin Breast Cancer
  - [x] Pemisahan variabel
  - [x] Encoding target variable
  - [x] Standarisasi fitur
  - [x] Train-test split

### JS04 - Clustering dengan K-Means

- [x] Praktikum 1: K-Means basics dengan Iris dataset
- [x] Praktikum 2: Elbow method untuk optimal K
- [x] Praktikum 3: Silhouette score evaluation
- [x] Tugas Praktikum: Customer segmentation
- [x] Dataset: Iris, Mall Customers

### JS05 - HDBSCAN Clustering

- [x] Praktikum 1: HDBSCAN introduction
- [x] Praktikum 2: Hyperparameter tuning
- [x] Tugas Praktikum: Density-based clustering
- [x] Comparison: HDBSCAN vs DBSCAN

### JS06 - Regression Analysis

- [x] Praktikum 1: Linear Regression
- [x] Praktikum 2: Polynomial Regression
- [x] Tugas Praktikum: Insurance/Salary prediction
- [x] Model evaluation: R², MSE, RMSE

### JS07 - Classification Algorithms

- [x] Praktikum 1: Logistic Regression
- [x] Praktikum 2: K-Nearest Neighbors
- [x] Praktikum 3: Support Vector Machine
- [x] Praktikum 4: Decision Tree
- [x] Praktikum 5: Random Forest
- [x] Praktikum 6: Naive Bayes
- [x] Tugas Praktikum: Algorithm comparison

### JS08 - Ensemble Learning

- [x] Tugas Praktikum: Ensemble methods
- [x] Bagging, Boosting, Stacking
- [x] Model performance optimization

### JS09 - Advanced Classification

- [x] Praktikum 1: Classification metrics
- [x] Praktikum 2: Model evaluation techniques
- [x] Praktikum 3: Hyperparameter tuning
- [x] Tugas Praktikum 1: Iris classification
- [x] Tugas Praktikum 2: Spam/Voice classification
- [x] ROC curve, confusion matrix, cross-validation

### JS11 - Neural Networks Fundamentals

- [x] Praktikum 1: Perceptron
- [x] Praktikum 2: Multi-Layer Perceptron
- [x] Praktikum 3: Backpropagation
- [x] Praktikum 4: Activation functions
- [x] Praktikum 5: Neural network training
- [x] Tugas Praktikum: Complete NN implementation

### JS13 - Deep Learning dengan Keras

- [x] Praktikum 1: Keras introduction
- [x] Praktikum 2: Deep neural networks
- [x] Praktikum 3: Model optimization
- [x] Tugas Praktikum: Advanced DNN
- [x] Regularization techniques

### JS14 - Convolutional Neural Networks

- [x] Praktikum 1: CNN basics
- [x] Praktikum 2: CIFAR-10 classification (>80% accuracy)
- [x] Tugas Praktikum: MNIST classification (>99% accuracy)
- [x] Model architecture: Conv2D, MaxPooling, Dropout, BatchNorm
- [x] Dataset: CIFAR-10, MNIST, Cats vs Dogs

### JS15 - ML Pipeline & Deployment

- [x] Praktikum 1: Day/Night classifier training
  - [x] HOG feature extraction
  - [x] Neural network training
  - [x] Model export (.h5, .pkl)
- [x] Praktikum 2: Cloud deployment
  - [x] Flask application development
  - [x] Docker containerization
  - [x] Hugging Face Spaces deployment
  - [x] Git version control
- [x] Tugas Praktikum: Complete MLOps pipeline

### Projects & Special Topics

- [x] UTS: Credit Card Clustering dengan ANN
- [x] Journal: Spotify Song Mood Clustering
  - [x] K-Means experiments
  - [x] DBSCAN clustering
  - [x] Spherical K-Means
- [x] Main Project: KMeans Song Mood Clustering

---

_Repository ini dibuat untuk keperluan pembelajaran Machine Learning di Kampus._

---

**Happy Learning! 🚀**
