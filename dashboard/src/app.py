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
    
    # Check requirements
    if st.session_state.model is None or 'preprocessor' not in st.session_state:
        st.warning("Please train a model first")
        st.stop()
    
    st.subheader("SHAP Explanations")
    
    if st.button("Generate SHAP Values"):
        with st.spinner("Calculating SHAP values..."):
            try:
                # ======================
                # 1. Data Preparation
                # ======================
                X_test = st.session_state.X_test
                model = st.session_state.model
                
                # Convert to dense array if sparse
                if hasattr(X_test, "toarray"):
                    X_test = X_test.toarray()
                X_test = np.array(X_test).astype(np.float64)
                X_test = np.nan_to_num(X_test)
                
                # Sample the data (smaller sample for Render's free tier)
                sample_size = min(50, X_test.shape[0])
                X_sample = X_test[:sample_size]
                feature_names = getattr(st.session_state, 'feature_names', 
                                      [f"Feature {i}" for i in range(X_sample.shape[1])])
                
                # ======================
                # 2. SHAP Calculation
                # ======================
                # Determine model type
                model_type = ('classifier' if hasattr(model, 'predict_proba') 
                            else 'regressor')
                
                # Initialize appropriate explainer
                if model_type == 'classifier' and hasattr(model, 'tree_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, X_sample, algorithm='auto')
                
                # Calculate SHAP values
                shap_values = explainer(X_sample)
                
                # ======================
                # 3. Shape Handling
                # ======================
                if isinstance(shap_values, list):
                    shap_values = np.array(shap_values)
                
                st.write("Raw SHAP values shape:", shap_values.shape)
                
                # Handle different output types
                if len(shap_values.shape) == 3:
                    # Classification (n_classes, n_samples, n_features)
                    class_names = (["Negative", "Positive"] if shap_values.shape[0] == 2 
                                 else [f"Class {i}" for i in range(shap_values.shape[0])])
                    output_idx = st.selectbox(
                        "Select class to explain",
                        options=list(range(shap_values.shape[0])),
                        index=1 if shap_values.shape[0] > 1 else 0,
                        format_func=lambda x: class_names[x]
                    )
                    shap_values = shap_values[output_idx]
                    expected_value = explainer.expected_value[output_idx]
                else:
                    # Regression (n_samples, n_features)
                    expected_value = (explainer.expected_value[0] 
                                    if isinstance(explainer.expected_value, (list, np.ndarray)) 
                                    else explainer.expected_value)
                
                # Final shape validation
                if shap_values.shape != X_sample.shape:
                    if shap_values.shape == X_sample.shape[::-1]:
                        shap_values = shap_values.T
                    else:
                        raise ValueError(
                            f"Shape mismatch: SHAP {shap_values.shape} vs Data {X_sample.shape}\n"
                            f"Try using a different explainer or checking your model output."
                        )
                
                # ======================
                # 4. Visualizations
                # ======================
                tab1, tab2, tab3 = st.tabs(["Global Importance", "Individual Explanation", "Debug Info"])
                
                with tab1:
                    st.subheader("Global Feature Importance")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(
                        shap_values,
                        X_sample,
                        feature_names=feature_names,
                        plot_type="dot",
                        show=False
                    )
                    st.pyplot(fig, bbox_inches='tight')
                    plt.close()
                
                with tab2:
                    st.subheader("Individual Prediction Analysis")
                    sample_idx = st.select_slider(
                        "Select sample to explain",
                        options=list(range(sample_size)),
                        value=0
                    )
                    
                    # Force plot
                    st.markdown("#### Force Plot")
                    fig = shap.force_plot(
                        expected_value,
                        shap_values[sample_idx],
                        X_sample[sample_idx],
                        feature_names=feature_names,
                        matplotlib=True
                    )
                    st.pyplot(fig, bbox_inches='tight')
                    
                    # Decision plot
                    st.markdown("#### Decision Plot")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.decision_plot(
                        expected_value,
                        shap_values[sample_idx],
                        feature_names=feature_names,
                        show=False
                    )
                    st.pyplot(fig, bbox_inches='tight')
                    plt.close()
                
                with tab3:
                    st.subheader("Debug Information")
                    st.json({
                        "data_shape": str(X_sample.shape),
                        "shap_values_shape": str(shap_values.shape),
                        "expected_value": str(expected_value),
                        "model_type": model_type,
                        "explainer_type": str(type(explainer))
                    })
                    if st.checkbox("Show raw values"):
                        st.write("Sample data:", X_sample[sample_idx])
                        st.write("SHAP values:", shap_values[sample_idx])
            
            except Exception as e:
                st.error("SHAP analysis failed. Common fixes:")
                st.markdown("""
                1. **Reduce sample size** (try 20-30 samples)
                2. **Check model type**:
                   - Tree models need `TreeExplainer`
                   - Linear models work with `LinearExplainer`
                3. **Verify feature names** match data dimensions
                """)
                
                if isinstance(e, ValueError) and "Shape mismatch" in str(e):
                    st.warning("Try adding this before SHAP calculation:")
                    st.code("""
                    if hasattr(model, 'predict_proba'):
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.Explainer(model, X_sample)
                    """)
                
                if st.checkbox("Show technical details"):
                    st.exception(e)
           
