# Red Wine Quality Prediction

## Project Overview
This project aims to predict the quality of red wine based on its physicochemical properties using machine learning. The dataset used for training and testing the model is sourced from the **UCI Machine Learning Repository**.

## Dataset Description
The dataset consists of various physicochemical features such as:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (target variable)

## Tools and Libraries Used
- **Python**: Main programming language
- **Pandas**: Data manipulation and preprocessing
- **NumPy**: Numerical computations
- **Scikit-Learn**: Machine learning model training and evaluation

## Project Workflow
1. **Importing Dependencies**
   - Load required libraries such as Pandas, NumPy, and Scikit-Learn.
2. **Data Loading and Preprocessing**
   - Read the dataset from a CSV file.
   - Handle missing values if any.
   - Normalize or scale features if needed.
3. **Train-Test Split**
   - Divide the dataset into training and testing sets.
4. **Model Training**
   - Train a **Random Forest Classifier** on the training dataset.
5. **Model Evaluation**
   - Use accuracy score and other metrics to evaluate model performance.
6. **Predictions**
   - Predict wine quality based on input features.

## How to Use
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/wine-quality-prediction.git
   ```
2. Navigate to the project directory:
   ```sh
   cd wine-quality-prediction
   ```
3. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook or script to train and test the model:
   ```sh
   jupyter notebook Wine(Red)_Quality_Prediction.ipynb
   ```

## Results
The trained model provides an accuracy score of approximately **91.8%**, depending on data preprocessing and hyperparameter tuning.

## Future Improvements
- Experiment with different machine learning models such as **XGBoost** or **Neural Networks**.
- Perform **feature selection** to improve model performance.
- Fine-tune hyperparameters for better accuracy.
- Deploy the model as a **web application** for user interaction.

## Contribution
Feel free to contribute by opening an issue or submitting a pull request.

## License
This project is licensed under the MIT License.
