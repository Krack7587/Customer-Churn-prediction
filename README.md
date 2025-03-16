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
![Churn Distribution](images/churn_distribution.png)

### Monthly Charges vs Churn
![Monthly Charges](images/monthly_charges_vs_churn.png)

### Feature Importance (Random Forest)
![Feature Importance](images/feature_importance.png)

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
