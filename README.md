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

## License and Disclaimer
DISCLAIMER:

This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830

LICENSE:

Copyright Battelle Memorial Institute 2026

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
