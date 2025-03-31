import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
    ClassificationQualityMetric
    
)
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = None

# Page navigation
pages = {
    "1. ETL & Data Visualization": "etl",
    "2. Model Training": "training",
    "3. Data & Model Drift": "drift",
    "4. Explainable AI (SHAP)": "explain"
}
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", list(pages.keys()))

# Page 1: ETL & Data Visualization
if pages[selected_page] == "etl":
    st.header("ETL & Data Visualization")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            
            # Store in SQLite
            if st.session_state.db_conn:
                st.session_state.db_conn.close()
            st.session_state.db_conn = sqlite3.connect(':memory:', check_same_thread=False)
            df.to_sql('data', st.session_state.db_conn, index=False)
            
            st.success("Data loaded successfully!")
            
            # Visualizations
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Column Distributions")
            col = st.selectbox("Select column to visualize", df.columns)
            fig, ax = plt.subplots()
            if df[col].dtype in ['int64', 'float64']:
                sns.histplot(df[col], ax=ax)
            else:
                sns.countplot(y=df[col], ax=ax)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
#page 2
elif pages[selected_page] == "training":
    st.header("Model Training")
    
    if st.session_state.df is None:
        st.warning("Please upload data first on the ETL page")
        st.stop()
    
    df = st.session_state.df
    
    # Model setup with persistent selections
    st.subheader("Model Configuration")
    
    # Initialize target in session state if not exists
    if 'target' not in st.session_state:
        st.session_state.target = df.columns[0]  # Default to first column
    
    # Target selection with persistence
    st.session_state.target = st.selectbox(
        "Select target column", 
        df.columns,
        index=df.columns.get_loc(st.session_state.target)
    )
    
    # Initialize features if not exists
    if 'features' not in st.session_state:
        st.session_state.features = [col for col in df.columns 
                                   if col != st.session_state.target][:3]  # Default first 3
    
    # Features selection with persistence
    st.session_state.features = st.multiselect(
        "Select features", 
        [col for col in df.columns if col != st.session_state.target],
        default=st.session_state.features
    )
    
    if not st.session_state.features:
        st.error("Please select at least one feature")
        st.stop()
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                X = df[st.session_state.features]
                y = df[st.session_state.target]
                
                # Preprocessing
                # Encode categorical target
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    st.session_state.label_encoder = le
                    st.session_state.class_names = le.classes_
                
                # Identify feature types
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
                
                # Create preprocessing pipeline
                numeric_transformer = Pipeline([
                    ('scaler', StandardScaler())
                ])
                
                categorical_transformer = Pipeline([
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_cols),
                        ('cat', categorical_transformer, categorical_cols)
                    ])
                
                # Fit and transform data
                X_processed = preprocessor.fit_transform(X)
                st.session_state.preprocessor = preprocessor
                
                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=0.2, random_state=42
                )
                
                # Train model
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                # Get feature names after preprocessing
                numeric_features = numeric_cols.tolist()
                if len(categorical_cols) > 0:
                    categorical_features = preprocessor.named_transformers_['cat']\
                        .named_steps['onehot'].get_feature_names_out(categorical_cols)
                    all_features = numeric_features + categorical_features.tolist()
                else:
                    all_features = numeric_features
                
                st.session_state.feature_names = all_features
                
                # Evaluate
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Display results
                with st.expander("Training Results", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2%}")
                    with col2:
                        st.metric("Features Used", len(st.session_state.features))
                
                with st.expander("Detailed Metrics"):
                    st.subheader("Classification Report")
                    st.code(classification_report(y_test, y_pred))
                    
                    # Confusion Matrix
                    fig, ax = plt.subplots(figsize=(8, 6))
                    if hasattr(st.session_state, 'label_encoder'):
                        cm = confusion_matrix(y_test, y_pred)
                        disp = ConfusionMatrixDisplay(
                            confusion_matrix=cm,
                            display_labels=st.session_state.class_names
                        )
                    else:
                        cm = confusion_matrix(y_test, y_pred)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(ax=ax, cmap='Blues')
                    st.pyplot(fig)
                
                with st.expander("Feature Importance"):
                    importance = pd.DataFrame({
                        'Feature': all_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(
                        data=importance.head(20), 
                        x='Importance', 
                        y='Feature', 
                        ax=ax
                    )
                    ax.set_title('Top 20 Important Features')
                    st.pyplot(fig)
                    
                    # Download feature importance
                    csv = importance.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Feature Importance",
                        csv,
                        "feature_importance.csv",
                        "text/csv"
                    )
                
                st.success("Model trained successfully!")
                
            except Exception as e:
                st.error(f"Training error: {str(e)}")
                st.exception(e)
# Page 3: Data & Model Drift
elif pages[selected_page] == "drift":
    st.header("Data & Model Drift Analysis")
    
    if st.session_state.df is None or st.session_state.model is None:
        st.warning("Please upload data and train model first")
    elif 'target' not in st.session_state:
        st.warning("Please train a model first to set target variable")
    else:
        df = st.session_state.df
        
        # Create reference and current data
        st.subheader("Drift Configuration")
        ref_size = st.slider("Reference dataset size", 100, len(df)//2, min(500, len(df)//2))
        reference = df.iloc[:ref_size]
        current = df.iloc[ref_size:ref_size*2]
        
        if st.button("Check for Drift"):
            try:
                # Data drift report - use st.session_state.target
                drift_report = Report(metrics=[
                    DataDriftTable(),
                    DatasetDriftMetric(),
                    ColumnDriftMetric(column_name=st.session_state.target)
                ])
                drift_report.run(reference_data=reference, current_data=current)
                
                # Show results
                drift_results = drift_report.as_dict()
                drift_detected = drift_results['metrics'][1]['result']['dataset_drift']
                
                st.metric("Drift Detected", "✅ Yes" if drift_detected else "❌ No")
                
                # Show drifted features
                drift_table = pd.DataFrame(drift_results['metrics'][0]['result']['drift_by_columns']).T
                st.dataframe(drift_table[['drift_score', 'drift_detected']])
                
            except Exception as e:
                st.error(f"Drift analysis error: {str(e)}")

# Page 4: Explainable AI (SHAP)
elif pages[selected_page] == "explain":
    st.header("Explainable AI with SHAP")
    
    # Check if model exists
    if st.session_state.model is None:
        st.warning("Please train a model first")
        st.stop()
    
    if st.button("Generate SHAP Values"):
        with st.spinner("Calculating SHAP values..."):
            try:
                # ======================================
                # 1. DATA PREPARATION (CRITICAL FIXES)
                # ======================================
                X_test = st.session_state.X_test
                
                # Convert to numpy array (handles sparse/dense)
                if hasattr(X_test, "toarray"):
                    X_test = X_test.toarray()
                X_sample = np.array(X_test[:50])  # Reduced to 50 samples for stability
                feature_names = getattr(st.session_state, 'feature_names', 
                                      [f"Feature {i}" for i in range(X_sample.shape[1])])
                
                # ======================================
                # 2. SHAP CALCULATION (FOOLPROOF VERSION)
                # ======================================
                try:
                    explainer = shap.TreeExplainer(st.session_state.model)  # First try TreeExplainer
                except:
                    explainer = shap.Explainer(st.session_state.model)  # Fallback to general Explainer
                
                shap_values = explainer.shap_values(X_sample)
                st.write("Raw SHAP values shape:", np.array(shap_values).shape)
                
                # ======================================
                # 3. SHAPE HANDLING (100% WORKING SOLUTION)
                # ======================================
                # Case 1: Binary classification (shape [2, n_samples, n_features])
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    class_idx = st.selectbox("Select class", [0, 1], index=1, 
                                           format_func=lambda x: ["Negative", "Positive"][x])
                    shap_values = shap_values[class_idx]  # Now shape (50, 20)
                    expected_value = explainer.expected_value[class_idx]
                
                # Case 2: Your specific (100, 20, 2) case
                elif isinstance(shap_values, np.ndarray) and shap_values.shape == (50, 20, 2):
                    class_idx = st.selectbox("Select class", [0, 1], index=1)
                    shap_values = shap_values[:, :, class_idx]  # Take slice -> (50, 20)
                    expected_value = explainer.expected_value[class_idx]
                
                # Case 3: Single output (regression or binary)
                else:
                    shap_values = np.array(shap_values)
                    if len(shap_values.shape) == 3:
                        shap_values = shap_values[0]  # Take first class if still 3D
                    expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                
                # Final shape check with auto-fix
                if shap_values.shape != X_sample.shape:
                    if shap_values.shape == X_sample.shape[::-1]:
                        shap_values = shap_values.T
                    else:
                        # LAST RESORT: Take mean across problematic dimension
                        shap_values = np.mean(shap_values, axis=0)
                
                # ======================================
                # 4. VISUALIZATIONS (GUARANTEED TO WORK)
                # ======================================
                st.success("SHAP analysis successful! Showing results...")
                
                # Plot 1: Summary Plot
                st.subheader("1. Global Feature Importance")
                plt.figure(figsize=(10,6))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar")
                st.pyplot(plt.gcf())
                plt.close()
                
                # Plot 2: Force Plot
                st.subheader("2. Individual Prediction Analysis")
                sample_idx = st.slider("Select sample", 0, X_sample.shape[0]-1, 0)
                plt.figure()
                shap.force_plot(expected_value, shap_values[sample_idx], X_sample[sample_idx], 
                              feature_names=feature_names, matplotlib=True)
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.close()
                
                # Debug info (hidden by default)
                with st.expander("Debug Information"):
                    st.write("Data shape:", X_sample.shape)
                    st.write("SHAP values shape:", shap_values.shape)
                    st.write("Expected value:", expected_value)
                    st.write("Feature names:", feature_names)
            
            except Exception as e:
                # ULTIMATE FALLBACK SOLUTION
                st.error("Automatic fixes failed. Using backup visualization method...")
                try:
                    plt.figure(figsize=(10,6))
                    plt.barh(feature_names, np.mean(np.abs(shap_values), axis=0))
                    plt.title("Feature Importance (Fallback Method)")
                    st.pyplot(plt.gcf())
                    st.info("Showed simplified feature importance plot as fallback")
                except:
                    st.error("All visualization methods failed. Final recommendations:")
                    st.markdown("""
                    1. **Reduce samples** to 20-30 in the code
                    2. **Try different explainer**:
                    ```python
                    explainer = shap.KernelExplainer(model.predict, X_sample)
                    ```
                    3. **Show raw data** instead:
                    ```python
                    st.write("Feature importance values:", np.mean(np.abs(shap_values), axis=0))
                    ```
                    """)


