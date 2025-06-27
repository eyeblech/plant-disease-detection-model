# Plant Disease Detector 🔍🌱

A no-bullshit Flask app for detecting plant diseases from leaf images. Works on Linux, Windows, and Mac.

![Demo Screenshot]

(https://github.com/user-attachments/assets/7af8b3d9-f5d2-4899-8e97-58b762b22681)

## System Requirements 💻
- **Minimum**:
  - 4GB RAM
  - 2GB free disk space

- **Recommended**:
  - NVIDIA GPU with CUDA support
  - 8GB+ RAM

## Installation 🛠️

### Linux (Debian/Ubuntu)
```bash
# 1. Install dependencies
sudo apt update && sudo apt install -y python3-pip python3-venv libjpeg-dev

# 2. Clone repo
git clone https://github.com/eyeblech/plant-disease-detection-model.git
cd plant-disease-detection-model

# 3. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```
### Windows 10/11
```bash
# 1. Install Python 3.9+ from Microsoft Store
# 2. Open PowerShell as Admin
Set-ExecutionPolicy Bypass -Scope Process

# 3. Clone repo
git clone https://github.com/eyeblech/plant-disease-detection-model.git
cd plant-disease-detection-model

# 4. Set up virtual environment
python -m venv venv
.\venv\Scripts\activate

# 5. Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### MacOS (Intel/Apple Silicon)
```bash
# 1. Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install dependencies
brew install python

# 3. Clone repo
git clone https://github.com/eyeblech/plant-disease-detection-model.git
cd plant-disease-detection-model

# 4. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### How to Run 🚀
```bash

# Start the web app
python app.py
```

### Then open http://localhost:5000 in your browser.


### File Structure 📁
```
.
├── app.py                # Main application
├── plant_disease_model.pt # Pretrained weights
├── class_names.txt       # Disease labels
├── static/
│   └── uploads/          # Where uploaded images go
└── templates/
    ├── index.html        # Main page
    └── result.html       # Results page
```
