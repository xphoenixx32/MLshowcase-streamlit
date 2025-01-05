# Basic
import streamlit as st
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML, XAI
import shap
from pdpbox import pdp
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, classification_report, f1_score, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor

# UI
from streamlit_option_menu import option_menu
#------------------------------------------------------------------------------------------------------#

st.title("a Demo of ML Model & XAI")
st.caption("*using **Machine Learning** method to modeling seaborn dataset, leveraging **SHAP value** to explain the explanatory power of independent variables*")
st.logo("assets/button.png")

sns.set_theme(style = "whitegrid")
#------------------------------------------------------------------------------------------------------#

# Predefined dataset selection
dataset_options = ['mpg', 'titanic']

# Dataset summaries
dataset_summaries = {
    'mpg': "Dataset about fuel efficiency of cars, with attributes such as miles per gallon (MPG), number of cylinders, horsepower, weight, model year, and origin. Often used for regression and exploratory data analysis.",
    'titanic': "Famous dataset on Titanic passengers, including attributes such as age, sex, class, survival status, and ticket price. Widely used for machine learning classification tasks and survival analysis."
}

# Dataset column descriptions
dataset_columns = {
    'mpg': {
        'mpg': "Miles per gallon, a measure of fuel efficiency.",
        'cylinders': "Number of cylinders in the car engine.",
        'displacement': "Engine displacement in cubic inches.",
        'horsepower': "Horsepower of the car.",
        'weight': "Weight of the car in pounds.",
        'acceleration': "Time to accelerate from 0 to 60 mph in seconds.",
        'model_year': "Year of the car's model (e.g., 70 for 1970).",
        'origin': "Origin of the car (1: USA, 2: Europe, 3: Japan).",
        'name': "Name of the car model."
    },
    'titanic': {
        'survived': "Survival status (0: No, 1: Yes).",
        'pclass': "Passenger class (1: First, 2: Second, 3: Third).",
        'sex': "Sex of the passenger (male or female).",
        'age': "Age of the passenger in years.",
        'sibsp': "Number of siblings/spouses aboard.",
        'parch': "Number of parents/children aboard.",
        'fare': "Fare amount paid in USD.",
        'embarked': "Port of embarkation (C: Cherbourg, Q: Queenstown, S: Southampton).",
        'class': "Passenger class as a string (First, Second, Third).",
        'who': "Categorical description of who (man, woman, child).",
        'deck': "Deck of the ship the passenger was on.",
        'embark_town': "Town where the passenger embarked.",
        'alive': "Survival status as a string (yes or no).",
        'alone': "Whether the passenger was alone (True or False)."
    }
}

#------------------------------------------------------------------------------------------------------#


# Allow user to upload a file or choose a predefined dataset
with st.sidebar:
    st.title("👾 *Choose a Dataset*")
    selected_dataset = st.selectbox(
        ['None'] + dataset_options  # Add 'None' for default empty selection
    )
    #------------------------------------------------------------------------------------------------------#

# Load the selected dataset or uploaded file
if selected_dataset != 'None':
    df = sns.load_dataset(selected_dataset)
    st.success(f"✅ Have Loaded <`{selected_dataset}`> dataset from Seaborn!")
else:
    df = None
#------------------------------------------------------------------------------------------------------#
st.subheader("🕹️  *Switch Tab* ")

# Option Menu
with st.container():
    selected = option_menu(
        menu_title = None,
        options = ["Summary", "EDA Plot", "ML & XAI"],
        icons = ["blockquote-left", "bar-chart-line-fill", "diagram-3-fill"],
        orientation = 'horizontal'
    )

# Proceed only if a dataset is loaded
if df is not None:
    if selected == "Summary":
        tab00, tab01, tab02, tab03 = st.tabs(['⌈ ⁰ Dataset Intro ⌉', 
                                              '⌈ ¹ Columns Info ⌉',
                                              '⌈ ² Dtypes Info ⌉', 
                                              '⌈ ³ Filter & View ⌉'])
        with tab00:
            st.subheader("🪄 Brief Intro to this Data")
            st.info(dataset_summaries[selected_dataset], icon = "ℹ️")
        #------------------------------------------------------------------------------------------------------#
        with tab01:
            if selected_dataset in dataset_columns:
                st.subheader("🪄 Definitions of the Columns")
                for col, desc in dataset_columns[selected_dataset].items():
                    st.markdown(f"**{col}**: {desc}")
        #------------------------------------------------------------------------------------------------------#
        with tab02:
            st.warning(" Summary & Data types of the Dataset ", icon = "🕹️")
            st.info('Here is the Dataset', icon = "1️⃣")
            st.dataframe(df)
            
            st.divider()

            st.info('Data Type of Variables', icon = "2️⃣")
            
            # Data types overview
            data_types = df.dtypes.to_frame('Types')
            data_types['Types'] = data_types['Types'].astype(str)  # Convert to strings for sorting
            st.write(data_types.sort_values('Types'))
            
            st.divider()
        
            # Only describe numeric columns
            st.info('Statistic of `Numeric` Variables', icon = "3️⃣")
            numeric_df = df.select_dtypes(include = ['number'])
            if not numeric_df.empty:
                st.write(numeric_df.describe([.25, .75, .9, .95]))
            else:
                st.write("No numeric columns to describe.")
        #------------------------------------------------------------------------------------------------------#
        with tab03:
            st.warning(" Filter & View on Specific Column & Value ", icon = "🕹️")
            # Filter Data Section
            columns = df.columns.tolist()

            # Unique keys for selectbox
            selected_column = st.selectbox(
                'Select column to filter by',
                columns,
                key = 'column_selector_tab2',
            )
        
            if selected_column:
                # Show Filtered Data
                unique_values = df[selected_column].dropna().unique()  # Drop NaNs for filtering
                unique_values = [str(value) for value in unique_values]  # Ensure all values are string
                selected_value = st.selectbox(
                    'Select value',
                    unique_values,
                    key = 'value_selector_tab2',
                )

                st.divider()
                
                # Filter DataFrame
                st.info(f'Filtered Data of {selected_column} = {selected_value}', icon = "1️⃣")
                filtered_df = df[df[selected_column].astype(str) == selected_value]
                st.write("Filtered DataFrame:")
                st.write(filtered_df)
                
                st.divider()
                
                # Calculate Data Groupby Selected-Column
                st.info(f'Value Count Groupby {selected_column}', icon = "2️⃣")
                group_stats = df.groupby(selected_column).size().reset_index(name = 'counts')
                group_stats.set_index(selected_column, inplace = True)
                st.write(group_stats.sort_values('counts', ascending = False))
    #------------------------------------------------------------------------------------------------------#
    if selected == "EDA Plot":
        tab10, tab11, tab12, tab13, tab14 = st.tabs(['⌈ ⁰ ANOVA & Violin Plot ⌉', 
                                                     '⌈ ¹ Area Plot ⌉', 
                                                     '⌈ ² Density Plot ⌉', 
                                                     '⌈ ³ VIF & Corr Matrix ⌉',
                                                     '⌈ ⁴ Pair Plot ⌉'])
        #------------------------------------------------------------------------------------------------------#
        with tab10:
            st.warning(" Testing the Statistically Significant Differences ", icon = "🕹️")
            
            # Filter numeric and categorical columns
            numeric_columns = df.select_dtypes(include = ['number']).columns.tolist()
            if selected_dataset == "mpg":
                st.error(" 🚫  Avoid using 'name' in case of `over-loading` ")
                categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()
            else:
                categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()
            
            if numeric_columns and categorical_columns:
                # Allow user to select a categorical column and a numeric column
                selected_category_column = st.selectbox('Select `Categorical` Column',
                                                        categorical_columns,
                                                        key = 'category_selector_tab3',
                                                       )
                selected_numeric_column = st.selectbox('Select `Numeric` Column',
                                                       numeric_columns,
                                                       key = 'numeric_selector_tab3',
                                                      )

                st.divider()

                if selected_category_column and selected_numeric_column:
                    # #0 Check the Anova Test
                    # Remove rows with missing values in the selected columns
                    df = df.dropna(subset = [selected_numeric_column, selected_category_column])

                    # Ensure the data columns are of the correct type
                    df[selected_numeric_column] = pd.to_numeric(df[selected_numeric_column], errors = 'coerce')
                    df[selected_category_column] = df[selected_category_column].astype(str)

                    # Retrieve unique category values and group data by categories
                    unique_category_values = df[selected_category_column].unique().tolist()
                    category_groups = [df[df[selected_category_column] == category][selected_numeric_column] for category in unique_category_values]

                    # Check if each group has sufficient data
                    for i, group in enumerate(category_groups):
                        if len(group) < 2:
                            st.error(f"⛔ Group '{unique_category_values[i]}' does not have enough data for ANOVA analysis!")
                            st.stop()
                        if group.var() == 0:
                            st.error(f"⛔ Group '{unique_category_values[i]}' has constant values, making ANOVA analysis impossible!")
                            st.stop()

                    # Perform ANOVA
                    anova_result = f_oneway(*category_groups)

                    # Output the results
                    st.info(f'One-way ANOVA between {selected_category_column} on {selected_numeric_column}', icon = "ℹ️")
                    st.write(f"ANOVA F-statistic: {anova_result.statistic:.3f}")
                    st.write(f"ANOVA p-value: {anova_result.pvalue:.3f}")

                    if anova_result.pvalue < 0.05:
                        st.success("✅ The differences between groups are statistically significant (p < 0.05).")
                    else:
                        st.warning("⛔ The differences between groups are `NOT` statistically significant (p >= 0.05).")
                    
                    st.divider()
                    
                    # Violin plot
                    st.info(f'Violin plot of {selected_numeric_column} by {selected_category_column}', icon = "ℹ️")
                    fig, ax = plt.subplots(figsize = (12, 6))
                    sns.violinplot(
                        data = df,
                        x = selected_category_column,
                        y = selected_numeric_column,
                        palette = "muted",
                        ax = ax,
                    )
                    ax.set_xlabel(selected_category_column)
                    ax.set_ylabel(selected_numeric_column)
                    
                    st.pyplot(fig)

                    st.divider()
                    
                    # Calculate Statistics
                    st.info(f'Statistics of {selected_numeric_column} by {selected_category_column}', icon = "ℹ️")
                    grouped_stats = df.groupby(selected_category_column)[selected_numeric_column].agg(count = 'count',
                                                                                                      mean = 'mean',
                                                                                                      std = 'std',
                                                                                                      q1 = lambda x: x.quantile(0.25),
                                                                                                      median = 'median',
                                                                                                      q3 = lambda x: x.quantile(0.75),
                                                                                                      ).reset_index()

                    grouped_stats[['mean', 'std', 'q1', 'median', 'q3']] = grouped_stats[['mean', 'std', 'q1', 'median', 'q3']].round(3)
                
                    # Rename Columns of Statistics
                    grouped_stats.rename(columns = {'count': 'Count',
                                                    'mean': 'Mean',
                                                    'std': 'STD',
                                                    'q1': 'Q1','median': 'Q2',
                                                    'q3': 'Q3',
                                                    },
                                         inplace = True,
                                         )
                    grouped_stats.set_index(selected_category_column, inplace = True)
                    st.write(grouped_stats.T)
            else:
                st.write("Ensure your dataset contains both numeric and categorical columns.", icon = "❗")
        #------------------------------------------------------------------------------------------------------#
        with tab11:
            st.warning(" Realize the Concentration of Data points ", icon = "🕹️")
            
            # Filter numeric and categorical columns
            numeric_columns = df.select_dtypes(include = ['number']).columns.tolist()
            if selected_dataset == "mpg":
                st.error(" 🚫  Avoid using 'name' in case of `over-loading` ")
                categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()
            else:
                categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()

            if numeric_columns and categorical_columns:
                # Allow user to select a categorical column and a numeric column
                selected_category_column = st.selectbox('Select `Categorical` Column',
                                                        categorical_columns,
                                                        key = 'category_selector_tab4',
                                                        )
                selected_numeric_column = st.selectbox('Select `Numeric` Column',
                                                       numeric_columns,
                                                       key = 'numeric_selector_tab4',
                                                       )

                if selected_category_column and selected_numeric_column:
                    df = df.dropna(subset = [selected_numeric_column, selected_category_column])
                    # Displot
                    st.info(f'Area Distribution of {selected_numeric_column} by {selected_category_column}', icon = "ℹ️")
                    sns_displot = sns.displot(data = df,
                                              x = selected_numeric_column,
                                              hue = selected_category_column,
                                              kind = "kde",
                                              height = 6,
                                              aspect = 1.5, # ratio of width:height = aspect
                                              multiple = "fill",
                                              clip = (0, None),
                                              palette = "ch:rot = -.25, hue = 1, light = .75",
                                              )

                    st.pyplot(sns_displot.fig)
            else:
                st.write("Ensure your dataset contains both `Numeric` and `Categorical` columns.", icon = "❗")
        #------------------------------------------------------------------------------------------------------#
        with tab12:
            st.warning(" Brief Realization on Correlation by Categorical Var Between Numeric Var ", icon = "🕹️")
            
            # Filter numeric columns
            numeric_columns = df.select_dtypes(include = ['number']).columns.tolist()

            # Filter categorical columns
            if selected_dataset == "mpg":
                st.error(" 🚫  Avoid using 'name' in case of `over-loading` ")
                categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()
            else:
                categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()

            if numeric_columns and categorical_columns:
                # Allow user to select a categorical column
                selected_category_column = st.selectbox('Select `Categorical` Column',
                                                        categorical_columns,
                                                        key = 'category_selector_tab5',
                                                        )
                unique_category_values = df[selected_category_column].unique().tolist()

                # Allow user to select numeric columns for X and Y axes
                st.info(" X & Y Should be Different ", icon = "ℹ️")
                selected_x = st.selectbox('Select *X-axis* column',
                                          numeric_columns,
                                          key = 'x_axis_selector_tab5',
                                          )
                selected_y = st.selectbox('Select *Y-axis* column',
                                          numeric_columns,
                                          key = 'y_axis_selector_tab5',
                                          )
                if selected_x and selected_y:
                    # Create subplots based on the number of unique category values
                    num_categories = len(unique_category_values)
                    cols = 2  # Maximum 2 plots per row
                    rows = (num_categories + cols - 1) // cols  # Calculate rows needed

                    # Initialize the figure
                    fig, axes = plt.subplots(
                    rows, cols,
                    figsize = (12, 6 * rows),
                    constrained_layout = True,
                    )
                    axes = axes.flatten()  # Flatten axes for easy iteration

                    # Plot each category
                    for i, category in enumerate(unique_category_values):
                        ax = axes[i]
                        filtered_data = df[df[selected_category_column] == category]
                        sns.kdeplot(data = filtered_data,
                                    x = selected_x,
                                    y = selected_y,
                                    fill = True,
                                    cmap = "Greens",
                                    ax = ax,
                                    warn_singular = False,  # Suppress singular warnings
                                    )
                        ax.set_title(f'{selected_category_column}: {category}')
                        ax.set_xlabel(selected_x)
                        ax.set_ylabel(selected_y)

                    # Hide unused subplots
                    for i in range(num_categories, len(axes)):
                        axes[i].axis('off')
                    # Display the plot
                    st.pyplot(fig)
        #------------------------------------------------------------------------------------------------------#
        with tab13:
            st.warning("Check the Multi-collinearity between Numeric Variables", icon = "🕹️")
            
            # Filter numeric columns
            numeric_columns = df.select_dtypes(include = ['number']).columns.tolist()
            
            if numeric_columns:
                # Put Numeric Var into Multi-Select
                selected_columns = st.multiselect("Select `Numeric` columns:",
                                                  numeric_columns,
                                                  default = numeric_columns,  # default settings for select all numeric
                                                  )
                st.divider()
                
                if selected_columns:
                    # VIF: Variance Inflation Factors
                    X = df[selected_columns].dropna()

                    # Add an Intercept
                    X = sm.add_constant(X)
                    
                    vif_data = pd.DataFrame()
                    vif_data["feature"] = X.columns
                    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    
                    st.info(' Use Variance Inflation Factors(`VIF`) to check `Multi-collinearity` ', icon = "ℹ️")
                    st.write(vif_data)
                    st.markdown('''
                                - VIF = 1: No multicollinearity.
                                - 1 < VIF < 5: Acceptable range.
                                - VIF ≥ 5 or 10: Severe multicollinearity; consider removing or combining features.
                    ''')
                    st.divider()

                    # Compute correlation matrix
                    correlation_matrix = df[selected_columns].corr()
        
                    # Mask to hide the upper triangle
                    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
                    # Plot the heatmap
                    fig, ax = plt.subplots(figsize = (12, 12))
                    sns.heatmap(correlation_matrix,
                                mask = mask,  # Apply the mask to hide the upper triangle
                                annot = True,
                                cmap = "coolwarm",
                                fmt = ".3f",
                                ax = ax,
                                )
                    ax.set_title("Correlation Matrix Heatmap (Lower Triangle Only)")
                    
                    st.info(' Use `Correlation Matrix Heatmap` for further checking ', icon = "ℹ️")
                    st.pyplot(fig)
                else:
                    st.warning("No columns selected. Please select at least one numeric column.", icon = "⚠️")
            else:
                st.error("Your dataset does not contain any numeric columns.", icon = "❗")
        #------------------------------------------------------------------------------------------------------#
        with tab14:
            st.warning(" Comparison between Numeric Var GroupBy Categorical Var  ", icon = "🕹️")
            
            # Filter numeric and categorical columns
            numeric_columns = df.select_dtypes(include = ['number']).columns.tolist()
            if selected_dataset == "mpg":
                st.error(" 🚫  Avoid using 'name' in case of `over-loading` ")
                categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()
            else:
                categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()

            if numeric_columns and categorical_columns:
                selected_category_column = st.selectbox(
                'Select Categorical Column',
                categorical_columns,
                key = 'category_selector_tab7',
                )

                if selected_category_column:
                    st.write(f"Selected Category: {selected_category_column}")

                    # Check if selected columns exist in df
                    if selected_category_column not in df.columns:
                        st.error(f"Column {selected_category_column} not found in dataframe.")
                    else:
                        # Generate pairplot
                        pairplot_fig = sns.pairplot(
                        df,
                        hue = selected_category_column,
                        vars = numeric_columns,
                        corner = True,
                        plot_kws = {'alpha': 0.7},
                        )
                        
                        # Display the plot using Streamlit
                        st.pyplot(pairplot_fig)
            else:
                st.write("Ensure your dataset contains both numeric and categorical columns.", icon = "❗")
    #------------------------------------------------------------------------------------------------------#
    if selected == "ML & XAI":
        tab30, tab31, tab32, tab33 = st.tabs(['⌈ ⁰ Model Summary ⌉',
                                              '⌈ ¹ Feature Importance ⌉',
                                              '⌈ ² Interaction Effect ⌉',
                                              '⌈ ³ Prediction on Sample ⌉'])
        
        # ------------------------------------------- #
        # MPG (Regression)
        # ------------------------------------------- #
        if selected_dataset == "mpg":
            # ---------- (1) Import models & parameters ------------- #
            with open("assets/mpg_best_model.pkl", "rb") as f:
                best_model = pickle.load(f)
        
            with open("assets/mpg_explainer.pkl", "rb") as f:
                explainer = pickle.load(f)
        
            shap_values = np.load("assets/mpg_shap_values.npy", allow_pickle = True)
        
            with open("assets/mpg_best_params.pkl", "rb") as f:
                best_params = pickle.load(f)
        
            # ---------- (2) Loading Data & Pre-processing ---------- #
            df = sns.load_dataset('mpg')
            X = df.drop(columns = ["mpg", "name"])
            X = pd.get_dummies(X, drop_first = True)
            y = df["mpg"]
            
            # ---------- (3) Visualization ---------- #
            with tab30:
                st.caption("*Regression Showcase*")
                st.write("### *LightGBM Regressor*")
                st.warning(" 🎖️ Prediction on the Fuel Efficiency of cars `mpg` (*Miles per Gallon*) ")
        
                y_pred = best_model.predict(X)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                residuals = y - y_pred
        
                st.write("1️⃣", f"**R-squared**: *{r2:.2f}*")
                st.markdown(
                    """
                    - R-squared measures the proportion of variance in the target variable that is explained by the model.
                    - A score of 0.94 indicates that 94% of the variability in the target variable is explained by the model, which demonstrates a strong fit.
                    """)

                st.divider()
                
                st.write("2️⃣", f"**Residual Mean**: *{np.mean(residuals):.2f}*")
                st.markdown(
                    """
                    - This represents the mean of the residuals (difference between observed and predicted values).
                    - A value close to 0 implies that the model's predictions, on average, are unbiased.
                    """)

                st.divider()
                
                st.write("3️⃣", f"**Mean Absolute Error (MAE)**: *{mae:.2f}*")
                st.markdown(
                    """
                    - MAE measures the average magnitude of the errors in a set of predictions, without considering their direction.
                    - An MAE of 1.37 indicates that the model's predictions deviate from the actual values by an average of 1.37 units.
                    """)

                st.divider()
                
                st.write("4️⃣", "**Best Model Parameters** *(GridSearchCV)*", best_params)
                st.markdown(
                    """
                    - max_depth: This parameter controls the maximum depth of a tree. Limiting the depth helps reduce overfitting while maintaining model performance.
                    - n_estimators: This parameter specifies the number of boosting iterations (trees). A value of 100 strikes a balance between computational efficiency and prediction accuracy.
                    """)
        
            with tab31:
                st.caption("*Regression Showcase*")
                st.write("### *SHAP Summary Plot*")

                st.info("ℹ️ This summary plot visualizes SHAP values, showing the impact of each feature on the model's output.")
                st.success("The combination of position (impact on the SHAP value) and color (feature value) allows you to understand how the magnitude of a feature value influences the prediction.")
        
                fig_summary, ax_summary = plt.subplots()
                shap.summary_plot(shap_values, X, show=False)
                st.pyplot(fig_summary)

                st.divider()

                st.markdown('''
                Key Components of the SHAP Summary Plot:
                1. X-Axis (SHAP Values):
                    - The X-axis represents the magnitude and direction of each feature's impact on the model's output.
                    - Positive SHAP values indicate that the feature contributes positively to the prediction (e.g., leaning towards a specific class), while negative SHAP values indicate a negative contribution.
                
                2. Y-Axis (Feature Names):
                    - Features are listed on the Y-axis, ranked by their importance. The most impactful features appear at the top.
                
                3. Point Distribution (Horizontal Spread):
                    - The horizontal spread of points shows the range of the feature's impact across all samples. A wider spread suggests a feature has varying impacts on predictions.
                
                4. Color (Feature Values):
                    - Each point's color reflects the actual value of the feature for a given observation.
                    - Typically, blue represents low feature values, while red represents high feature values.
                ''')
        
            with tab32:
                st.caption("*Regression Showcase*")
                st.write("### *2D Partial Dependence Plot*")
                
                st.info("ℹ️ 2D Partial Dependence Plot (PDP) shows how two features influence the predicted outcome of a machine learning model, while keeping all other features constant. This plot Helps identify interactions between key features, providing valuable insights.")
                st.success("Color or Height represents the model's prediction value. A smooth surface suggests minimal interaction between the two features, while distinct peaks or valleys indicate significant interaction effects.")
                
                feature_1 = st.selectbox("Select Feature 1:", X.columns)
                feature_2 = st.selectbox("Select Feature 2:", X.columns)
        
                if feature_1 and feature_2:
                    fig_pdp, ax_pdp = plt.subplots(figsize=(6, 4))
                    PartialDependenceDisplay.from_estimator(
                        estimator=best_model,
                        X=X,
                        features = [(feature_1, feature_2)],
                        kind="average",
                        ax=ax_pdp
                    )
                    st.pyplot(fig_pdp)
        
            with tab33:
                st.caption("*Regression Showcase*")
                st.write("### *SHAP Waterfall Plot*")
                st.info("ℹ️ Waterfall plot illustrates how specific features contribute to the final prediction for a single instance in a machine learning model.")
        
                row_index = st.number_input("Select a Row Index of a Sample ⤵️", 
                                            min_value = 0, 
                                            max_value = len(X) - 1, 
                                            step = 1)
                if row_index is not None:
                    fig_waterfall, ax_waterfall = plt.subplots()
                    shap.waterfall_plot(
                        shap.Explanation(
                            base_values=explainer.expected_value,
                            values=shap_values[row_index],
                            data=X.iloc[row_index],
                            feature_names=X.columns.tolist()
                        ),
                        show=False
                    )
                    st.pyplot(fig_waterfall)
        
        # ------------------------------------------- #
        # Titanic (Classification)
        # ------------------------------------------- #
        elif selected_dataset == "titanic":
            # ---------- (1) Import models & parameters ------------- #
            with open("assets/titanic_best_model.pkl", "rb") as f:
                best_model = pickle.load(f)
        
            with open("assets/titanic_explainer.pkl", "rb") as f:
                explainer = pickle.load(f)
        
            shap_values = np.load("assets/titanic_shap_values.npy", allow_pickle = True)
        
            with open("assets/titanic_best_params.pkl", "rb") as f:
                best_params = pickle.load(f)
        
            # ---------- (2) Loading Data & Pre-processing ---------- #
            df = sns.load_dataset('titanic')
            
            valid_values = ["yes", "no"]
            df = df[df["alive"].isin(valid_values)]
            df = df.dropna(subset = ["alive"])
            df['alive'] = df['alive'].map({'yes': 1, 'no': 0})

            df['age'] = df['age'].fillna(df['age'].median())
            df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
            df['embark_town'] = df['embark_town'].fillna('Unknown')
            df['family_size'] = df['sibsp'] + df['parch'] + 1
            df = df.drop(columns = ['deck'])

            columns_to_drop = ['adult_male', 'who', 'survived', 'deck', 'embarked', 'pclass', 'alone', 'deck']
            df.drop(columns = [col for col in columns_to_drop if col in df.columns], inplace = True)
            df.dropna(axis = 0, how = "any")
            
            y = df["alive"]
            X = df.drop(columns = ["alive"])
            
            X = pd.get_dummies(X, drop_first = True)
        
            # ---------- (3) Visualization ---------- #
            with tab30:
                st.caption("*Classification Showcase*")
                st.write("### *RandomForest Classifier*")
                st.warning(" 🎖️ Prediction on `alived` of Titanic passengers (*is alived or not*) ")

        
                y_pred = best_model.predict(X)
                report_dict = classification_report(y, y_pred, output_dict = True)
                cm = pd.DataFrame(report_dict).transpose()
                f1 = f1_score(y, y_pred)
        
                st.write("1️⃣", "**Confusion Matrix**:")
                st.write(cm)
                st.markdown(
                    """
                    The confusion matrix provides a detailed breakdown of the model's performance for each class:
                    
                    - Precision: The proportion of correctly predicted positive observations to the total predicted positives.
                    - Recall: The proportion of correctly predicted positive observations to the all observations in the actual class.
                    - F1-Score: The harmonic mean of precision and recall, balancing both metrics.
                    - Support: The actual number of occurrences of each class in the dataset.
                    
                    Additional metrics:
                    
                    - Accuracy: The overall percentage of correctly predicted observations.
                    - Macro Average: Average performance across all classes, treating each class equally.
                    - Weighted Average: Average performance weighted by the support of each class.
                    """)

                st.divider()
                
                st.write("2️⃣", f"**F1-score**: *{f1:.3f}*")
                st.markdown(
                    """
                    The overall F1-score for the model is 0.916, indicating strong balance between precision and recall.
                    """)

                st.divider()
                
                st.write("3️⃣", "**Best Model Parameters** *(GridSearchCV)*:", best_params)
                st.markdown(
                    """
                    - max_depth: This parameter controls the maximum depth of a tree. Limiting the depth helps reduce overfitting while maintaining model performance.
                    - n_estimators: This parameter specifies the number of boosting iterations (trees). A value of 100 strikes a balance between computational efficiency and prediction accuracy.
                    """)
        
            with tab31:
                st.caption("*Classification Showcase*")
                st.write("### *SHAP Summary Plot*")

                st.info("ℹ️ This summary plot visualizes SHAP values, showing the impact of each feature on the model's output.")
                st.success("The combination of position (impact on the SHAP value) and color (feature value) allows you to understand how the magnitude of a feature value influences the prediction.")

                # st.success("✅ being a man (*who_man*) may negatively influence survival predictions (negative SHAP values), while being a woman (*who_woman*) has a positive influence.")
                # st.info("ℹ️ *age* plays an essential role in survival prediction, and higher ticket prices (*fare*) correlate with better survival odds.")

                fig_summary, ax_summary = plt.subplots()
                # with Two-classification: shap_values.shape = (n_samples, n_features, 2)
                shap.summary_plot(shap_values[:, :, 1], X, show = False)
                st.pyplot(fig_summary)

                st.divider()

                st.markdown('''
                Key Components of the SHAP Summary Plot:
                1. X-Axis (SHAP Values):
                    - The X-axis represents the magnitude and direction of each feature's impact on the model's output.
                    - Positive SHAP values indicate that the feature contributes positively to the prediction (e.g., leaning towards a specific class), while negative SHAP values indicate a negative contribution.
                
                2. Y-Axis (Feature Names):
                    - Features are listed on the Y-axis, ranked by their importance. The most impactful features appear at the top.
                
                3. Point Distribution (Horizontal Spread):
                    - The horizontal spread of points shows the range of the feature's impact across all samples. A wider spread suggests a feature has varying impacts on predictions.
                
                4. Color (Feature Values):
                    - Each point's color reflects the actual value of the feature for a given observation.
                    - Typically, blue represents low feature values, while red represents high feature values.
                ''')
        
            with tab32:
                st.caption("*Classification Showcase*")
                st.write("### *2D Partial Dependence Plot*")

                st.info("ℹ️ 2D Partial Dependence Plot (PDP) shows how two features influence the predicted outcome of a machine learning model, while keeping all other features constant. This plot Helps identify interactions between key features, providing valuable insights.")
                st.success("Color or Height represents the model's prediction value. A smooth surface suggests minimal interaction between the two features, while distinct peaks or valleys indicate significant interaction effects.")
                
                feature_1 = st.selectbox("Select Feature 1:", X.columns)
                feature_2 = st.selectbox("Select Feature 2:", X.columns)
        
                if feature_1 and feature_2:
                    fig_pdp, ax_pdp = plt.subplots(figsize=(6, 4))
                    PartialDependenceDisplay.from_estimator(
                        estimator=best_model,
                        X=X,
                        features = [(feature_1, feature_2)],
                        kind="average",
                        target=1,  
                        ax=ax_pdp
                    )
                    st.pyplot(fig_pdp)
        
            with tab33:
                st.caption("*Classification Showcase*")
                st.write("### *SHAP Waterfall Plot*")
                st.info("ℹ️ Waterfall plot illustrates how specific features contribute to the final prediction for a single instance in a machine learning model.")
        
                row_index = st.number_input("Select a Row Index of a Sample ⤵️", 
                                            min_value=0, 
                                            max_value=len(X) - 1, 
                                            step=1)
                if row_index is not None:
                    fig_waterfall, ax_waterfall = plt.subplots()
                    shap.waterfall_plot(
                        shap.Explanation(
                            base_values=explainer.expected_value[1],
                            values=shap_values[1][row_index, :],
                            data=X.iloc[row_index, :],
                            feature_names=X.columns.tolist()
                        ),
                        show=False
                    )
                    st.pyplot(fig_waterfall)
    #------------------------------------------------------------------------------------------------------#
else:
    st.error('Click TOP-LEFT Side Bar Navigation to GET STARTED', icon = "📎")
