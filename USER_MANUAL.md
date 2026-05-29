# Pet Breed Classifier — Detailed User Manual

**System:** AI-powered Pet Breed Classification using Deep Learning & Transfer Learning  
**Model:** EfficientNetV2B0 (37 breed classes)  
**Expected Accuracy:** ~90–95%  
**Estimated Total Time:** ~1.5 hours (mostly training)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Getting the Code from GitHub](#2-getting-the-code-from-github)
3. [Python Installation](#3-python-installation)
4. [GPU Setup (NVIDIA CUDA & cuDNN)](#4-gpu-setup-nvidia-cuda--cudnn)
5. [Setting Up the Virtual Environment](#5-setting-up-the-virtual-environment)
6. [Installing Dependencies](#6-installing-dependencies)
7. [Verifying Your Setup](#7-verifying-your-setup)
8. [Training the Model](#8-training-the-model)
9. [Evaluating the Model & Generating Figures](#9-evaluating-the-model--generating-figures)
10. [Understanding the Output Files](#10-understanding-the-output-files)
11. [Troubleshooting](#11-troubleshooting)
12. [Maintenance Tips](#12-maintenance-tips)
13. [Quick Run Cheat Sheet](#13-quick-run-cheat-sheet)
14. [Contact & Support](#14-contact--support)

---

## 1. Prerequisites

Before starting, make sure your system meets the following requirements.

### Minimum Requirements
| Component | Requirement |
|-----------|------------|
| Operating System | Windows 10 or Windows 11 |
| Python Version | 3.10 or 3.11 (do **not** use 3.12+) |
| RAM | 8 GB |
| GPU | NVIDIA GPU (strongly recommended) |
| Storage | At least 5 GB of free disk space |

### Recommended Requirements
| Component | Recommendation |
|-----------|---------------|
| GPU | NVIDIA RTX 4050 (6 GB VRAM) |
| CUDA Version | 11.8 |
| RAM | 16 GB |
| Internet | Stable connection (~800 MB dataset download) |

### Required Software (installed during this guide)
- Python 3.10 or 3.11
- Git (for cloning/forking from GitHub)
- NVIDIA GPU Drivers
- CUDA Toolkit 11.8
- cuDNN 8.6

---

## 2. Getting the Code from GitHub

You have two options: **Cloning** (direct copy, no GitHub account needed) or **Forking** (creates your own copy on GitHub, requires an account). Choose one.

---

### Option A — Clone the Repository (Recommended for most users)

Cloning gives you a local copy of the project so you can run it on your machine.

**Step 1: Install Git**

If you don't have Git installed:
1. Go to [https://git-scm.com/downloads](https://git-scm.com/downloads)
2. Download the Windows installer and run it.
3. During installation, accept all defaults and click **Next** through each screen.
4. Verify Git is installed by opening **Command Prompt** and running:
   ```cmd
   git --version
   ```
   You should see something like `git version 2.x.x`.

**Step 2: Navigate to where you want the project**

Open **Command Prompt** (press `Windows key`, type `cmd`, press `Enter`):
```cmd
cd Desktop
```

**Step 3: Clone the repository**
```cmd
git clone https://github.com/lhanze27/pet_classifier.git
```

**Step 4: Move into the project folder**
```cmd
cd pet_classifier
```

Your project folder is now ready at `Desktop\pet_classifier`.

---

### Option B — Fork the Repository (Recommended for contributors)

Forking creates your own copy of the repository on your GitHub account, allowing you to make changes and push them independently.

**Step 1: Create a GitHub account (if you don't have one)**
1. Go to [https://github.com](https://github.com)
2. Click **Sign up** and follow the on-screen steps.

**Step 2: Fork the repository**
1. Go to the original repository page on GitHub.
2. Click the **Fork** button in the top-right corner of the page.
3. Under "Owner", select your GitHub username.
4. Click **Create fork**.

**Step 3: Install Git** (same as Option A, Step 1 above)

**Step 4: Clone your fork to your local machine**

Open **Command Prompt**:
```cmd
cd Desktop
git clone https://github.com/YOUR-USERNAME/pet_classifier.git
cd pet_classifier
```

Replace `YOUR-USERNAME` with your actual GitHub username.

**Step 5: (Optional) Connect to the original repo for future updates**
```cmd
git remote add upstream https://github.com/lhanze27/pet_classifier.git
```

This lets you pull future updates from the original project using:
```cmd
git fetch upstream
git merge upstream/main
```

---

## 3. Python Installation

**IMPORTANT:** Only use Python **3.10 or 3.11**. Python 3.12+ is NOT compatible with TensorFlow 2.10.

**Step 1: Download Python**
1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Under "Looking for a specific release?", scroll down and download **Python 3.11.x** or **Python 3.10.x** (the latest patch version of either is fine).
3. Click the Windows installer link (e.g., `Windows installer (64-bit)`).

**Step 2: Run the installer**
1. Double-click the downloaded `.exe` file.
2. On the first screen, **check the box** that says **"Add Python to PATH"** — this is critical.
3. Click **Install Now**.
4. Wait for the installation to complete and click **Close**.

**Step 3: Verify Python is installed correctly**

Open a new **Command Prompt** window and run:
```cmd
python --version
```
You should see:
```
Python 3.11.x
```
or
```
Python 3.10.x
```

If you see `Python 3.12.x` or higher, uninstall and reinstall the correct version.

---

## 4. GPU Setup (NVIDIA CUDA & cuDNN)

This section configures your NVIDIA GPU to work with TensorFlow for accelerated training. Skip this section only if you plan to train on CPU (which will be significantly slower).

### Step 1: Update NVIDIA GPU Drivers

1. Download **GeForce Experience** from [https://www.nvidia.com/en-us/geforce/geforce-experience/](https://www.nvidia.com/en-us/geforce/geforce-experience/) and install it.
2. Open GeForce Experience.
3. Go to the **Drivers** tab and click **Check for updates**.
4. Download and install the latest driver.
5. Restart your computer after the driver installs.

### Step 2: Install CUDA Toolkit 11.8

1. Go to the CUDA 11.8 archive: [https://developer.nvidia.com/cuda-11-8-0-download-archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Select:
   - **Operating System:** Windows
   - **Architecture:** x86_64
   - **Version:** 11
   - **Installer Type:** exe (local)
3. Click **Download** (the file is approximately 2.5 GB).
4. Run the downloaded installer.
5. Choose **Express (Recommended)** installation and click **Next**.
6. Wait for the installation to complete (~10–15 minutes).
7. Restart your computer.

### Step 3: Install cuDNN 8.6 for CUDA 11.8

1. Go to the cuDNN archive: [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)
2. You will need a free **NVIDIA Developer Account** to download. Click **Join** or **Sign In** to create one.
3. Find **cuDNN v8.6.x for CUDA 11.x** and download the **Windows zip file**.
4. Extract the downloaded zip file.
5. Inside the extracted folder, you will see three folders: `bin`, `include`, and `lib`.
6. Copy the **contents** of each folder into the corresponding folder at:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\
   ```
   - Copy contents of `bin` → `C:\...\CUDA\v11.8\bin\`
   - Copy contents of `include` → `C:\...\CUDA\v11.8\include\`
   - Copy contents of `lib` → `C:\...\CUDA\v11.8\lib\`
7. Restart your computer.

---

## 5. Setting Up the Virtual Environment

A virtual environment isolates this project's dependencies from other Python projects on your machine.

**Step 1: Open Command Prompt and navigate to the project folder**
```cmd
cd Desktop\pet_classifier
```

**Step 2: Create the virtual environment**
```cmd
python -m venv venv
```

This creates a `venv` folder inside your project directory.

**Step 3: Activate the virtual environment**
```cmd
venv\Scripts\activate
```

You will know it is active when you see `(venv)` at the beginning of your command prompt line, like this:
```
(venv) C:\Users\YourName\Desktop\pet_classifier>
```

> **Important:** You must activate the virtual environment every time you open a new Command Prompt window before running any project scripts.

---

## 6. Installing Dependencies

With the virtual environment active, install all required Python libraries.

**Step 1: Upgrade pip to the latest version**
```cmd
pip install --upgrade pip
```

**Step 2: Install all project dependencies**
```cmd
pip install -r requirements.txt
```

This will install the following libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.10.0 | Deep learning framework |
| tensorflow-datasets | 4.8.3 | Downloads and manages the pet dataset |
| matplotlib | ≥ 3.7.0 | Generates training curve graphs |
| seaborn | ≥ 0.12.0 | Generates the confusion matrix |
| scikit-learn | ≥ 1.3.0 | Classification metrics (precision, recall, F1) |
| numpy | ≥ 1.24.0 | Numerical operations |
| Pillow | ≥ 10.0.0 | Image loading and processing |

> **Note:** This download is approximately 2 GB and may take 5–10 minutes depending on your internet speed.

---

## 7. Verifying Your Setup

Before training, verify that Python and the GPU are configured correctly.

**Step 1: Verify TensorFlow can see your GPU**

With the virtual environment active, run:
```cmd
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**Expected output (GPU working correctly):**
```
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Output if GPU is not detected:**
```
GPUs: []
```

If the GPU is not detected, revisit [Section 4](#4-gpu-setup-nvidia-cuda--cudnn) and ensure CUDA and cuDNN are installed correctly.

**Step 2: Verify all required files are present**

Your project folder should contain:
```
pet_classifier/
├── 01_train_model.py
├── 02_evaluate_model.py
├── requirements.txt
├── README.md
└── venv/
```

If any of the `.py` files are missing, re-clone or re-fork the repository (see [Section 2](#2-getting-the-code-from-github)).

---

## 8. Training the Model

**Before you begin training:**
- Plug your laptop into a power outlet (training uses significant power)
- Use a cooling pad if available (the GPU will run hot)
- Close any heavy applications (games, video editors, browsers with many tabs)
- Ensure you have a stable internet connection for the initial dataset download

**Step 1: Make sure your virtual environment is active**

Open **Command Prompt** and run:
```cmd
cd Desktop\pet_classifier
venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt.

**Step 2: Start the training script**
```cmd
python 01_train_model.py
```

**Step 3: What to expect during training**

The script runs in three stages:

| Stage | Description | Estimated Time |
|-------|-------------|----------------|
| Dataset Download | Downloads the Oxford-IIIT Pet Dataset (~800 MB) | 5–10 minutes (first run only) |
| Phase 1 Training | Trains only the classifier head (base model frozen) for 10 epochs | ~15 minutes |
| Phase 2 Training | Fine-tunes the entire model for additional epochs | ~20–30 minutes |

- During Phase 1, accuracy should climb from ~3% up to ~85–90%.
- During Phase 2, accuracy should improve to ~92–95%.
- Training is complete when you see the message: **"TRAINING COMPLETE!"**

**Step 4: Do not close the terminal**

Do not close the Command Prompt window while training is running. You may minimize it, but closing it will terminate the training process.

> **If training crashes with an Out of Memory error:**
> 1. Open `01_train_model.py` in any text editor (Notepad, VS Code, etc.).
> 2. Find the line: `BATCH_SIZE = 16`
> 3. Change it to: `BATCH_SIZE = 8`
> 4. Save the file and re-run `python 01_train_model.py`.

---

## 9. Evaluating the Model & Generating Figures

After training completes successfully, run the evaluation script to generate all output figures and reports.

**Step 1: Run the evaluation script**

With the virtual environment still active:
```cmd
python 02_evaluate_model.py
```

**Step 2: What the evaluation script produces**

| Output File | Location | Purpose |
|-------------|----------|---------|
| `training_curves.png` | `outputs/figures/` | Shows training and validation accuracy per epoch |
| `confusion_matrix.png` | `outputs/figures/` | Visual grid of prediction performance across all 37 breeds |
| `sample_predictions.png` | `outputs/figures/` | Side-by-side examples of predicted vs. actual breed labels |
| `per_class_accuracy.png` | `outputs/figures/` | Bar chart of accuracy per breed class |
| `classification_report.txt` | `outputs/` | Full table of precision, recall, and F1-score per breed |
| `results_summary.txt` | `outputs/` | High-level summary of overall accuracy and key metrics |

---

## 10. Understanding the Output Files

### Training Curves (`training_curves.png`)
A line graph showing how training accuracy and validation accuracy changed across each epoch. Ideally, both lines should increase and converge close together, indicating the model is learning without overfitting.

### Confusion Matrix (`confusion_matrix.png`)
A grid where rows represent the true breed labels and columns represent the predicted labels. A perfect classifier would show a bright diagonal from top-left to bottom-right. Off-diagonal cells indicate misclassifications between visually similar breeds.

### Per-Class Accuracy (`per_class_accuracy.png`)
A horizontal bar chart showing how accurately the model identifies each of the 37 breeds individually. Breeds with shorter bars may benefit from additional training data.

### Sample Predictions (`sample_predictions.png`)
A grid of example images showing the pet photo, the predicted breed, and the actual breed. Green labels indicate correct predictions; red labels indicate incorrect ones.

### Classification Report (`classification_report.txt`)
A detailed text table showing three metrics for each breed class:
- **Precision:** Of all times the model predicted this breed, what percentage was correct.
- **Recall:** Of all actual images of this breed, what percentage the model correctly identified.
- **F1-score:** The harmonic mean of precision and recall (higher is better; max is 1.0).

### Final Output Folder Structure
```
pet_classifier/
├── 01_train_model.py
├── 02_evaluate_model.py
├── requirements.txt
├── README.md
├── venv/
└── outputs/
    ├── models/
    │   ├── best_phase1.keras
    │   ├── best_final.keras
    │   └── final_model.keras        ← Main trained model file
    ├── figures/
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   ├── sample_predictions.png
    │   └── per_class_accuracy.png
    ├── class_names.txt
    ├── classification_report.txt
    └── results_summary.txt
```

---

## 11. Troubleshooting

### "No module named 'tensorflow'"
**Cause:** The virtual environment is not activated.  
**Fix:** Run `venv\Scripts\activate` before running any script.

---

### "Could not load dynamic library 'cudart64_110.dll'"
**Cause:** CUDA is not installed or is installed incorrectly.  
**Fix:** Reinstall CUDA Toolkit 11.8 following [Section 4, Step 2](#step-2-install-cuda-toolkit-118) exactly.

---

### "ResourceExhaustedError: OOM" (Out of Memory)
**Cause:** The GPU does not have enough VRAM for the current batch size.  
**Fix:**
1. Open `01_train_model.py` in a text editor.
2. Locate the line `BATCH_SIZE = 16`.
3. Change it to `BATCH_SIZE = 8` (or `BATCH_SIZE = 4` if the error persists).
4. Save and rerun `python 01_train_model.py`.

---

### GPU not detected (`GPUs: []`)
**Cause:** NVIDIA drivers, CUDA, or cuDNN are missing or mismatched.  
**Fix:**
1. Reinstall the latest NVIDIA drivers via GeForce Experience.
2. Reinstall CUDA Toolkit 11.8.
3. Reinstall cuDNN 8.6 and ensure the files are copied to the correct CUDA directory.
4. Restart your computer and retest.

---

### Training stuck at low accuracy (below 50%)
**Cause:** The model may be training on CPU instead of GPU.  
**Fix:**
1. Open **Task Manager** (Ctrl + Shift + Esc).
2. Click the **Performance** tab and then **GPU**.
3. If GPU usage is 0% during training, CUDA is not configured correctly — redo [Section 4](#4-gpu-setup-nvidia-cuda--cudnn).

---

### "Kernel died" or laptop freezes during training
**Cause:** Excessive VRAM or RAM usage.  
**Fix:**
1. Lower `BATCH_SIZE` in `01_train_model.py` (try 8, then 4).
2. Close all browser tabs and background applications before training.

---

### Dataset download fails or hangs
**Cause:** Slow or interrupted internet connection.  
**Fix:**
1. Check your internet connection.
2. Delete the partial download folder:  
   `C:\Users\YourName\tensorflow_datasets`
3. Rerun `python 01_train_model.py`.

---

### Missing dependencies after pulling updates from GitHub
**Cause:** New packages may have been added to `requirements.txt`.  
**Fix:** With the virtual environment active, run:
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 12. Maintenance Tips

- **Keep TensorFlow updated** within the compatible version range (2.10.x) by checking for patch releases.
- **Back up trained models** regularly — copy the `outputs/models/` folder to an external drive or cloud storage. The main model file (`final_model.keras`) is approximately 30 MB.
- **Store datasets securely** — do not delete the `tensorflow_datasets` folder in your home directory if you plan to retrain, as re-downloading takes 5–10 minutes.
- **Monitor GPU temperatures** during training using MSI Afterburner or GPU-Z. Sustained temperatures above 90°C may indicate insufficient cooling.
- **Pull the latest updates** from GitHub periodically to get bug fixes and improvements:
  ```cmd
  git pull origin main
  ```
  Or if you are on a fork:
  ```cmd
  git fetch upstream
  git merge upstream/main
  ```

---

## 13. Quick Run Cheat Sheet

Once your one-time setup is complete, running the full pipeline again is as simple as:

```cmd
cd Desktop\pet_classifier
venv\Scripts\activate
python 01_train_model.py
python 02_evaluate_model.py
```

To update your local code from GitHub before running:
```cmd
cd Desktop\pet_classifier
git pull origin main
venv\Scripts\activate
pip install -r requirements.txt
python 01_train_model.py
python 02_evaluate_model.py
```

---

## 14. Contact & Support

For technical issues or questions, contact the development team:

| Name | Contact Number |
|------|---------------|
| Enano, Arcejay | 0961 122 7283 |
| Abunda, Lhanze Tyler | 0908 653 7095 |

When reporting an issue, please include:
- A screenshot of the full error message in the Command Prompt
- Your Python version (`python --version`)
- Whether your GPU was detected (`tf.config.list_physical_devices('GPU')`)
- Which script produced the error (`01_train_model.py` or `02_evaluate_model.py`)

---

*Pet Breed Classifier — User Manual v1.0*
