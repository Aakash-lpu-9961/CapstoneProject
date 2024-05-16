import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(page_title="Predicting Heart Disease Holistically: A Machine Learning Approach")

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #4B0082;
        }
        .stButton button {
            background-color: #4B0082;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            width: 100%;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .stDataFrame table {
            font-size: 18px;
        }
        .stTable table {
            font-size: 18px;
        }
        .stPlotlyChart, .stPyplot {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Heading
st.markdown("<h1 style='text-align: center;'>Predicting Heart Disease Holistically: A Machine Learning Approach</h1>", unsafe_allow_html=True)

# Initialize session state for plot visibility
if 'show_accuracy_plot' not in st.session_state:
    st.session_state.show_accuracy_plot = False
if 'show_sensitivity_plot' not in st.session_state:
    st.session_state.show_sensitivity_plot = False
if 'show_roc_curve' not in st.session_state:
    st.session_state.show_roc_curve = False
if 'show_f1_score_plot' not in st.session_state:
    st.session_state.show_f1_score_plot = False
if 'show_precision_plot' not in st.session_state:
    st.session_state.show_precision_plot = False

# Import dataset
data_path = st.text_input("Enter the path to your dataset:", "C:/Users/HP/OneDrive/Documents/Capstone Project Documents/archive/heart.csv")
if not data_path:
    st.warning("Please enter a valid dataset path to proceed.")
else:
    data = pd.read_csv(data_path)

    # Display dataset
    st.markdown("<h2>Dataset Overview:</h2>", unsafe_allow_html=True)
    st.write(data.head())

    # Separate features and target variable
    X = data.drop('target', axis=1)
    y = data['target']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier()
    }

    # Train and evaluate models
    results = {}
    roc_curves = {}
    sensitivity = {}
    f1_scores = {}
    precisions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)  # For ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)  # For ROC curve
        roc_curves[name] = {'fpr': fpr, 'tpr': tpr}  # For ROC curve
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {'accuracy': accuracy, 'classification_report': report}
        
        # Sensitivity calculation
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity[name] = tp / (tp + fn)
        
        # Calculate F1-score and precision
        f1_scores[name] = f1_score(y_test, y_pred)
        precisions[name] = precision_score(y_test, y_pred)

    # Display results
    st.markdown("<h2>Model Performance Metrics</h2>", unsafe_allow_html=True)
    st.markdown("This table shows the performance metrics for each machine learning model. These metrics help evaluate the effectiveness of the models in predicting heart disease.")
    
    results_table = {
        "Model": [],
        "Accuracy": [],
        "Sensitivity": [],
        "F1-score": [],
        "Precision": []
    }

    for name, result in results.items():
        results_table["Model"].append(name)
        results_table["Accuracy"].append(result['accuracy'])
        results_table["Sensitivity"].append(sensitivity[name])
        results_table["F1-score"].append(f1_scores[name])
        results_table["Precision"].append(precisions[name])

    results_df = pd.DataFrame(results_table)
    st.dataframe(results_df)

    # Buttons to show plots
    if st.button("Show Accuracy Comparison"):
        st.session_state.show_accuracy_plot = True
    if st.button("Show Sensitivity Comparison"):
        st.session_state.show_sensitivity_plot = True
    if st.button("Show ROC Curves"):
        st.session_state.show_roc_curve = True
    if st.button("Show F1-score Comparison"):
        st.session_state.show_f1_score_plot = True
    if st.button("Show Precision Comparison"):
        st.session_state.show_precision_plot = True

    # Accuracy Comparison Plot
    if st.session_state.show_accuracy_plot:
        st.markdown("<h2>Accuracy Comparison</h2>", unsafe_allow_html=True)
        st.markdown("Accuracy is the proportion of true results (both true positives and true negatives) among the total number of cases examined. This plot compares the accuracy of each model.")
        fig, ax = plt.subplots()
        sns.barplot(x=results_df["Model"], y=results_df["Accuracy"], palette="viridis", ax=ax)
        ax.set_title('Accuracy Comparison')
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        st.pyplot(fig)

    # Sensitivity Comparison Plot
    if st.session_state.show_sensitivity_plot:
        st.markdown("<h2>Sensitivity (True Positive Rate) Comparison</h2>", unsafe_allow_html=True)
        st.markdown("Sensitivity, also known as the true positive rate, measures the proportion of actual positives that are correctly identified by the model. This plot compares the sensitivity of each model.")
        fig, ax = plt.subplots()
        sns.barplot(x=results_df["Model"], y=results_df["Sensitivity"], palette="viridis", ax=ax)
        ax.set_title('Sensitivity (True Positive Rate) Comparison')
        ax.set_xlabel('Models')
        ax.set_ylabel('Sensitivity')
        st.pyplot(fig)

    # Plot ROC curves
    if st.session_state.show_roc_curve:
        st.markdown("<h2>ROC Curves</h2>", unsafe_allow_html=True)
        st.markdown("The Receiver Operating Characteristic (ROC) curve illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. The area under the curve (AUC) provides an aggregate measure of performance across all classification thresholds.")
        fig, ax = plt.subplots()
        for name, roc_curve_data in roc_curves.items():
            ax.plot(roc_curve_data['fpr'], roc_curve_data['tpr'], label=f'{name} (AUC = {auc(roc_curve_data["fpr"], roc_curve_data["tpr"]):.2f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='black')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)

    # Plot F1-score comparison
    if st.session_state.show_f1_score_plot:
        st.markdown("<h2>F1-score Comparison</h2>", unsafe_allow_html=True)
        st.markdown("The F1-score is the harmonic mean of precision and recall. It considers both false positives and false negatives and is a good measure when you need a balance between precision and recall. This plot compares the F1-scores of each model.")
        fig, ax = plt.subplots()
        sns.barplot(x=results_df["Model"], y=results_df["F1-score"], palette="viridis", ax=ax)
        ax.set_xlabel('Model')
        ax.set_ylabel('F1-score')
        ax.set_title('F1-score Comparison')
        st.pyplot(fig)

    # Plot Precision comparison
    if st.session_state.show_precision_plot:
        st.markdown("<h2>Precision Comparison</h2>", unsafe_allow_html=True)
        st.markdown("Precision is the ratio of correctly predicted positive observations to the total predicted positives. High precision indicates a low false positive rate. This plot compares the precision of each model.")
        fig, ax = plt.subplots()
        sns.barplot(x=results_df["Model"], y=results_df["Precision"], palette="viridis", ax=ax)
        ax.set_xlabel('Model')
        ax.set_ylabel('Precision')
        ax.set_title('Precision Comparison')
        st.pyplot(fig)


# Group Members Data

import streamlit as st
import pandas as pd

# Group members details
members_details = {
    "Name": ["Aakash Kumar", "Prakhar Tripathi", "Shubham Tripathi", "Vishal Pramanik", "Prince Kumar", "Anuj"],
    "Registration No.": ["12015260", "12020632", "12020252", "12015479", "12020736", "12011729"],
    "Email": ["mananaakash527@gmail.com", "prakhar.tripathi.dibiyapur@gmail.com", "shubhtripathi.96274@gmail.com", "vishalpramanik480@gmail.com", "princejnv902@gmail.com", "anujthukran123@gmail.com"],
    "Contacts": ["7544935760", "7505250254", "8006370216", "7976418444", "7544935760", "9729845571"]
}

members_df = pd.DataFrame(members_details)

# Display group members details
st.markdown("<h2 style='text-align: center;'>Group Members Details</h2>", unsafe_allow_html=True)
st.dataframe(members_df.style.set_table_styles([
    {'selector': 'thead th', 'props': [('background-color', '#4B0082'), ('color', 'white'), ('font-size', '20px')]},
    {'selector': 'tbody td', 'props': [('text-align', 'center'), ('font-size', '18px')]},
    {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', '#f2f2f2')]},
    {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', '#ffffff')]},
    {'selector': 'tr:hover', 'props': [('background-color', '#ffff99')]}
]).set_properties(**{'font-size': '18px'}))

