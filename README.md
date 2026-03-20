# RhizoAI: AI-ready datasets and workflows for rhizosphere imaging and classification of roots

## What is RhizoAI?
RhizoAI leverages machine learning to analyze OCT images of plant roots, enabling accurate classification and trait extraction in the complex rhizosphere environment. This repository provides tools to preprocess tomography data into machine-learning-ready datasets, as well as provides baseline tools for root classification using RandomForestClassifier.

## Project Structure
```
├── src
│   ├── __init__.py
│   ├── preprocess.py          # Functions for preprocessing image data
│   ├── method1_classifier.py  # Random Forest classification method 1
│   ├── method2_classifier.py  # Random Forest classification method 2
│   ├── feature_extraction.py  # Functions for feature extraction from images
│   └── utils.py               # Utility functions for data handling
├── legacy_notebooks           # Jupyter Notebooks for experimentation
│   ├── image_similarity_tests.ipynb
│   ├── preprocess_image.ipynb
│   ├── OCT_RF_Classification_Method_1.ipynb
│   └── OCT_RF_Classification_Method_2.ipynb
├── requirements.txt           # Project dependencies
├── setup.py                   # Project packaging information
└── README.md                  # Project documentation
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/aramyxt/RhizoAI.git
cd RhizoAI
pip install ".[method1,method2]"
```

Depending on the desired methods, you can install their respective requirements separately:

```bash
pip install ".[method1]"

pip install ".[method2]"
```

## Usage
1. **Preprocessing**: Use the `preprocess.py` module to preprocess the OCT images and prepare the data for classification.
2. **Feature Extraction**: Utilize the `feature_extraction.py` module to extract features from the images.
3. **Model Training**: 
   - For Method 1, run the `method1_classifier.py` to train the Random Forest model using flattened data data.
   - For Method 2, run the `method2_classifier.py` to train the model using HOG features extracted from the images.
4. **Evaluation**: Evaluate the model performance using the provided Jupyter Notebooks.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
