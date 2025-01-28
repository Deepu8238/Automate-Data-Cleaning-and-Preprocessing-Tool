import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
import featuretools as ft
import sqlite3
from sqlalchemy import create_engine
import pyodbc

# Title of the app
st.title("🚀 Automated Data Cleaning and Preprocessing Tool")

# Sidebar for data input selection
st.sidebar.header("Data Input Options")
input_type = st.sidebar.radio(
    "Select Input Source",
    ["File Upload", "Database Connection"]
)

if input_type == "File Upload":
    # File upload section
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        # Load the dataset based on file type
        file_extension = uploaded_file.name.split(".")[-1]
        
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
            
        st.write("### Original Dataset")
        st.write(df)

elif input_type == "Database Connection":
    # Database connection section
    db_type = st.selectbox(
        "Select Database Type",
        ["SQLite", "MySQL", "PostgreSQL", "SQL Server"]
    )
    
    if db_type == "SQLite":
        db_file = st.file_uploader("Upload SQLite Database", type=["db", "sqlite"])
        if db_file:
            query = st.text_area("Enter your SQL query")
            if st.button("Execute Query"):
                conn = sqlite3.connect(db_file.name)
                df = pd.read_sql_query(query, conn)
                conn.close()
                st.write("### Original Dataset")
                st.write(df)
    
    else:
        st.text_input("Host", key="host")
        st.text_input("Port", key="port")
        st.text_input("Database Name", key="database")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        
        query = st.text_area("Enter your SQL query")
        
        if st.button("Connect and Execute Query"):
            try:
                if db_type == "MySQL":
                    connection_string = f"mysql+pymysql://{st.session_state.username}:{st.session_state.password}@{st.session_state.host}:{st.session_state.port}/{st.session_state.database}"
                elif db_type == "PostgreSQL":
                    connection_string = f"postgresql://{st.session_state.username}:{st.session_state.password}@{st.session_state.host}:{st.session_state.port}/{st.session_state.database}"
                elif db_type == "SQL Server":
                    connection_string = f"mssql+pyodbc://{st.session_state.username}:{st.session_state.password}@{st.session_state.host}:{st.session_state.port}/{st.session_state.database}?driver=SQL+Server"
                
                engine = create_engine(connection_string)
                df = pd.read_sql_query(query, engine)
                st.write("### Original Dataset")
                st.write(df)
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")

if 'df' in locals():
    # Display basic info about the dataset
    st.write("### Dataset Info")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write("Columns and their data types:")
    st.write(df.dtypes)

    # Sidebar for user inputs
    st.sidebar.header("Data Cleaning and Preprocessing Options")

    # Handle missing values
    st.sidebar.subheader("Missing Values")
    missing_value_strategy = st.sidebar.selectbox(
        "Choose a strategy for handling missing values:",
        ["Drop rows with missing values", "Impute with mean", "Impute with median", "Impute with mode"]
    )

    # Outlier detection
    st.sidebar.subheader("Outlier Detection")
    outlier_detection = st.sidebar.checkbox("Detect and handle outliers")
    if outlier_detection:
        outlier_method = st.sidebar.selectbox(
            "Choose a method for outlier detection:",
            ["Z-score", "IQR"]
        )

    # Encode categorical variables
    st.sidebar.subheader("Categorical Variables")
    encode_categorical = st.sidebar.checkbox("Encode categorical variables (One-Hot Encoding)")

    # Scale numerical features
    st.sidebar.subheader("Feature Scaling")
    scale_numerical = st.sidebar.checkbox("Scale numerical features (Standardization)")

    # Feature selection
    st.sidebar.subheader("Feature Selection")
    feature_selection = st.sidebar.checkbox("Automatically select important features")
    if feature_selection:
        feature_selection_method = st.sidebar.selectbox(
            "Choose a feature selection method:",
            ["Correlation", "Feature Importance"]
        )

    # Automated feature engineering
    st.sidebar.subheader("Automated Feature Engineering")
    auto_feature_engineering = st.sidebar.checkbox("Generate new features using FeatureTools")

    # Apply preprocessing
    if st.button("Clean and Preprocess Data"):
        # Copy the original dataframe
        df_cleaned = df.copy()

        # Handle missing values
        if missing_value_strategy == "Drop rows with missing values":
            df_cleaned = df_cleaned.dropna()
        else:
            strategy = missing_value_strategy.split()[-1].lower()  # Extract "mean", "median", or "mode"
            imputer = SimpleImputer(strategy=strategy)
            df_cleaned = pd.DataFrame(imputer.fit_transform(df_cleaned), columns=df_cleaned.columns)

        # Outlier detection and handling
        if outlier_detection:
            numerical_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns
            if outlier_method == "Z-score":
                z_scores = np.abs(stats.zscore(df_cleaned[numerical_cols]))
                df_cleaned = df_cleaned[(z_scores < 3).all(axis=1)]
            elif outlier_method == "IQR":
                Q1 = df_cleaned[numerical_cols].quantile(0.25)
                Q3 = df_cleaned[numerical_cols].quantile(0.75)
                IQR = Q3 - Q1
                df_cleaned = df_cleaned[~((df_cleaned[numerical_cols] < (Q1 - 1.5 * IQR)) | (df_cleaned[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

        # Encode categorical variables
        if encode_categorical:
            categorical_cols = df_cleaned.select_dtypes(include=["object", "category"]).columns
            if len(categorical_cols) > 0:
                encoder = OneHotEncoder(sparse=False, drop="first")
                encoded_data = encoder.fit_transform(df_cleaned[categorical_cols])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
                df_cleaned = df_cleaned.drop(categorical_cols, axis=1)
                df_cleaned = pd.concat([df_cleaned, encoded_df], axis=1)

        # Scale numerical features
        if scale_numerical:
            numerical_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns
            if len(numerical_cols) > 0:
                scaler = StandardScaler()
                df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])

        # Feature selection
        if feature_selection:
            if feature_selection_method == "Correlation":
                corr_matrix = df_cleaned.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
                df_cleaned = df_cleaned.drop(to_drop, axis=1)
            elif feature_selection_method == "Feature Importance":
                # Placeholder for feature importance (requires a target variable)
                st.warning("Feature Importance requires a target variable. Please add a target column.")

        # Automated feature engineering
        if auto_feature_engineering:
            try:
                # Create an EntitySet and add the dataframe
                es = ft.EntitySet(id="data")
                # Use add_dataframe instead of entity_from_dataframe
                es.add_dataframe(
                    dataframe_name="df",
                    dataframe=df_cleaned,
                    index="id" if "id" in df_cleaned.columns else None,
                    make_index=True if "id" not in df_cleaned.columns else False
                )
                
                # Generate features
                feature_matrix, feature_defs = ft.dfs(
                    entityset=es,
                    target_entity="df",
                    max_depth=2,  # Limit depth to avoid explosion of features
                    features_only=False
                )
                df_cleaned = feature_matrix
            except Exception as e:
                st.warning(f"Feature engineering failed: {str(e)}. Proceeding with original features.")

        # Display cleaned dataset
        st.write("### Cleaned and Preprocessed Dataset")
        st.write(df_cleaned)

        # Download cleaned dataset
        st.write("### Download Cleaned Dataset")
        csv = df_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned Data as CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )