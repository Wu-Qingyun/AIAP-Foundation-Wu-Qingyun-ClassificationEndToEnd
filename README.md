# HDB Resale Flats Classification

This project implements machine learning models to classify HDB resale flats as either above or below the median price. The classification helps potential buyers and sellers quickly understand the relative value of a flat in the resale market.

## Project Structure

```
root/
    |- eda.ipynb              # Jupyter notebook for exploratory data analysis
    |- README.md              # Project documentation
    |- requirements.txt       # Python package dependencies
    |- data/
           |- data.csv        # Raw HDB resale flats data
    |- src/
           |- data_preparation.py  # Data cleaning and preprocessing
           |- model_training.py    # Model training and evaluation
           |- config.yaml          # Configuration settings
    |- main.py               # Main pipeline script
```

## Setup Instructions

1. Create and activate a conda environment:
```bash
conda create -n aiap python=3.8
conda activate aiap
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the pipeline:
```bash
python main.py
```

## Project Components

### Data Preparation (`src/data_preparation.py`)
- Loads and cleans the raw HDB resale flats data
- Performs feature engineering (e.g., calculating remaining lease)
- Handles categorical and numerical features
- Splits data into training and testing sets

### Model Training (`src/model_training.py`)
Implements three classification models:
1. Random Forest Classifier
2. XGBoost Classifier
3. LightGBM Classifier

Each model is evaluated using:
- Accuracy score
- Cross-validation scores
- Classification report
- Confusion matrix

### Configuration (`src/config.yaml`)
Contains configurable parameters for:
- Data paths
- Preprocessing settings
- Model hyperparameters
- Training settings

## Model Evaluation

The models are evaluated based on:
1. Test accuracy
2. Cross-validation performance
3. Classification metrics (precision, recall, F1-score)

Results are visualized in `model_comparison.png`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 