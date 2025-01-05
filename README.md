
# Machine Learning & XAI Demo Application

## Overview

This application demonstrates the integration of **Machine Learning** techniques and **Explainable AI (XAI)** methods using Streamlit. It utilizes Seaborn datasets (`mpg` and `titanic`) as examples for regression and classification tasks. Key functionalities include dataset exploration, statistical analysis, feature importance visualization, and partial dependence plots.

---

## Features

- **Dataset Selection**: Choose between preloaded datasets (`mpg` and `titanic`) or upload your custom dataset.
- **Dataset Summary**: View column descriptions, data types, and statistics.
- **Exploratory Data Analysis (EDA)**: Perform ANOVA, correlation analysis, and visualize data distributions.
- **Machine Learning Models**:
  - Regression with `LGBMRegressor` on the `mpg` dataset.
  - Classification with `RandomForestClassifier` on the `titanic` dataset.
- **Explainable AI**:
  - SHAP summary plots for feature importance.
  - Partial dependence plots for interaction effects.
  - SHAP waterfall plots for individual predictions.

---

## How to Run

1. Clone this repository:

    ```bash
    git clone https://github.com/your-repo/ml-xai-demo.git
    cd ml-xai-demo
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Start the Streamlit application:

    ```bash
    streamlit run app.py
    ```

4. Open the app in your browser at `http://localhost:8501`.

---

## File Structure

```plaintext
ml-xai-demo/
├── app.py                  # Main Streamlit app file
├── assets/                 # Contains models, SHAP values, and other assets
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

---

## Dependencies

This application uses the following Python libraries:

- **Basic**: `streamlit`, `pickle`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `lightgbm`
- **Explainable AI**: `shap`, `pdpbox`
- **Statistics**: `statsmodels`, `scipy`

Install all dependencies via `requirements.txt`.

---

## Screenshots

### Home Page
![Home Page](assets/home_page.png)

### SHAP Summary Plot
![SHAP Summary Plot](assets/shap_summary.png)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue.

---

## Author

**Jean**  
Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/in/your-profile/) | [GitHub](https://github.com/your-profile/)
