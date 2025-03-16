# Customer Churn Prediction

## Project Overview
This project aims to predict customer churn using machine learning techniques. By analyzing customer data, we identify key factors influencing churn and provide actionable insights to help businesses retain customers.

## Key Features
- Exploratory Data Analysis (EDA) to identify churn patterns.
- Machine learning models (Random Forest, Logistic Regression, etc.) for churn prediction.
- Feature engineering techniques to improve model accuracy.
- Performance evaluation using metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Business recommendations to reduce churn.

## Dataset
The dataset contains customer-related attributes such as:
- **Demographics**: Age, Gender, etc.
- **Account Information**: Contract type, Monthly charges, Tenure, etc.
- **Service Usage**: Internet service, Streaming services, etc.
- **Target Variable**: Churn (Yes/No)

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd Customer-Churn-prediction-main
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook to explore the project:
   ```bash
   jupyter notebook
   ```

## Model Performance
- **Best Model**: Random Forest (ROC-AUC score: 0.85)
- **Key Metrics**:
  - Accuracy: XX%
  - Precision: XX%
  - Recall: XX%
  - F1-Score: XX%

## Insights & Recommendations
- Customers with **short tenure** and **high monthly charges** are more likely to churn.
- Contract type plays a significant role; long-term contracts reduce churn.
- Businesses can reduce churn by offering **loyalty programs** and **personalized discounts**.

## Visualizations
### Churn Distribution
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('data/customer_churn.csv')
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=data, palette='coolwarm')
plt.title('Churn Distribution')
plt.show()
```

### Monthly Charges vs Churn
```python
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=data, palette='coolwarm')
plt.title('Monthly Charges vs Churn')
plt.show()
```

### Feature Importance (Random Forest)
```python
from sklearn.ensemble import RandomForestClassifier

X = data.drop(columns=['Churn'])
y = data['Churn']
model = RandomForestClassifier()
model.fit(X, y)

feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
feature_importances.plot(kind='bar', color='skyblue')
plt.title('Feature Importance (Random Forest)')
plt.show()
```

## Technologies Used
- **Programming Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Machine Learning Models**: Random Forest, Logistic Regression, etc.
- **Visualization Tools**: Matplotlib, Seaborn

## Contributors
- [Your Name]

## License
This project is licensed under the MIT License.

## Acknowledgments
- Dataset Source: [Mention if applicable]
- Inspiration from various machine learning research articles and business case studies.
