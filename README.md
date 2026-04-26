🧠 NeuroScan: DL-Based Brain Tumor Detection🔬 Project Purpose & OverviewBrain tumors are highly critical neurological conditions where early and accurate detection is a matter of life and death. While Magnetic Resonance Imaging (MRI) is the gold standard for scanning, manual interpretation of these complex images is time-consuming, highly subjective, and prone to human error—especially when identifying diffuse tumor boundaries like Gliomas.NeuroScan is a complete, end-to-end Deep Learning pipeline and web application designed to automate this diagnosis. Moving beyond simple "black-box" AI classification, this project focuses heavily on clinical interpretability and accessibility.It classifies MRI scans into four distinct categories (Glioma, Meningioma, Pituitary, and No Tumor) while simultaneously applying mathematical edge enhancement and generating localized heatmaps to prove why the model made its prediction.✨ Key FeaturesAdvanced Deep Learning: Utilizes EfficientNet-B0, a state-of-the-art Convolutional Neural Network that balances high accuracy with a lightweight parameter count.Mathematical Edge Enhancement: Implements Inverse Fast Fourier Transform (IFFT) to filter low-frequency noise and highlight critical, high-frequency structural boundaries in the brain tissue.Explainable AI (XAI): Employs GradCAM (Gradient-weighted Class Activation Mapping) to dynamically generate heatmaps, visually highlighting the suspected tumor region."Potato-PC" Friendly: The heavy lifting (model training) is decoupled to Google Colab (GPU). The local Flask deployment is heavily optimized to run inference entirely on standard consumer CPUs.Dark Medical UI: A custom, fully responsive web dashboard mimicking high-end medical terminals.⚙️ System Architecture & WorkflowThe system is logically split into a frontend UI, a Flask backend API, and a PyTorch inference pipeline.graph TD
    A[User Uploads MRI Scan] -->|Drag & Drop UI| B(Flask Backend 'app.py')
    B -->|Bytes in Memory| C{Inference Pipeline}
    
    C --> D[EfficientNet-B0 Classification]
    C --> E[IFFT Frequency Domain Enhancement]
    
    D -->|Softmax Probabilities| F{Tumor Detected?}
    
    F -- Yes --> G[Trigger GradCAM Hooks]
    G --> H[Generate Localized Heatmap]
    F -- No --> I[Skip Heatmap]
    
    E --> J[Spatial Domain Reconstruction]
    H --> K[Blend Heatmap over Enhanced MRI]
    I --> K
    
    J --> L[Base64 Encoding]
    K --> L
    
    L -->|JSON Response| M[Frontend Dashboard Updated]
    
    classDef frontend fill:#080f17,stroke:#00c8ff,stroke-width:2px,color:#fff;
    classDef backend fill:#1f2937,stroke:#00e5a0,stroke-width:2px,color:#fff;
    classDef ai fill:#3730a3,stroke:#ffc840,stroke-width:2px,color:#fff;
    
    class A,M frontend;
    class B,L backend;
    class C,D,E,F,G,H,I,J,K ai;
📁 File StructureThe repository is neatly organized to separate training scripts from the deployment application.brain-tumor-detection/
│
├── notebook/
│   └── train_model.ipynb        ← Google Colab training notebook
│
├── app/
│   ├── app.py                   ← Flask backend (API router)
│   ├── pipeline.py              ← Inference + IFFT + GradCAM logic
│   ├── model_utils.py           ← PyTorch model loader utility
│   │
│   ├── models/                  ← Drop your trained .pth files here
│   │   └── efficientnet_b0_brain_tumor.pth
│   │
│   ├── uploads/                 ← Temp buffer directory
│   │
│   ├── static/
│   │   ├── css/style.css        ← "Dark Medical" UI Styling
│   │   └── js/main.js           ← Async fetch & DOM logic
│   │
│   └── templates/
│       └── index.html           <- Main dashboard structure
│
└── requirements.txt             <- Python dependencies
🚀 Setup & Installation GuidePrerequisitesPython 3.8 or higherGit1. Local Deployment (Running the Dashboard)Follow these steps to run the inference dashboard on your personal computer:# 1. Clone the repository
git clone [https://github.com/yourusername/brain-tumor-detection.git](https://github.com/yourusername/brain-tumor-detection.git)
cd brain-tumor-detection

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install the required dependencies
pip install -r requirements.txt

# 4. Add the trained weights
# Ensure your trained EfficientNet model (.pth) is placed inside the 'app/models/' directory.

# 5. Start the local Flask server
cd app
python app.py
Access the App: Open your web browser and navigate to http://127.0.0.1:50002. Training the Model (Google Colab)If you wish to retrain the model from scratch:Download the Brain Tumor MRI Dataset from Kaggle as a .zip file.Open notebook/train_model.ipynb in Google Colab.Change the runtime to GPU (T4).Upload your dataset zip file and follow the notebook instructions to extract, train, and save the .pth weights.📊 Dataset & ResultsThe model was trained on a robust dataset of over 7,000 MRI scans, segmented into a 5,600-image training set and a 1,600-image testing set.Overall Accuracy: ~94% (Validated over 15 Epochs)Classes Detected: Glioma (0.89 F1), Meningioma (0.93 F1), Pituitary (0.98 F1), and No Tumor (0.95 F1).Disclaimer: This tool is for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.
