import requests
import zipfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import altair as alt
import streamlit as st

        
    
@st.cache_data  # Cache the function to enhance performance
def load_data():
    # Define the file path
    zip_url_1= "https://github.com/aaubs/ds-master/raw/main/data/assignments_datasets/KIVA/kiva_loans_part_0.csv.zip"

    
    # Download the ZIP file
    response = requests.get(zip_url_1)
    response.raise_for_status()  # Check if the request was successful

    # Open the ZIP file from the response content
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # List all files in the ZIP
        print(zf.namelist())
        
        # Read a specific CSV file from the ZIP
        df1 = pd.read_csv(zf.open('kiva_loans_part_0.csv'))
    return df1

# Load the data using the defined function
df1 = load_data()




@st.cache_data  # Cache the function to enhance performance
def load_data():
    # Define the file path
    zip_url_2= "https://github.com/aaubs/ds-master/raw/main/data/assignments_datasets/KIVA/kiva_loans_part_1.csv.zip"

    
    # Download the ZIP file
    response = requests.get(zip_url_2)
    response.raise_for_status()  # Check if the request was successful

    # Open the ZIP file from the response content
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # List all files in the ZIP
        print(zf.namelist())
        
        # Read a specific CSV file from the ZIP
        df2 = pd.read_csv(zf.open('kiva_loans_part_1.csv'))
    return df2

# Load the data using the defined function
df2 = load_data()

@st.cache_data  # Cache the function to enhance performance
def load_data():
    # Define the file path
    zip_url_3= "https://github.com/aaubs/ds-master/raw/main/data/assignments_datasets/KIVA/kiva_loans_part_2.csv.zip"

    
    # Download the ZIP file
    response = requests.get(zip_url_3)
    response.raise_for_status()  # Check if the request was successful

    # Open the ZIP file from the response content
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # List all files in the ZIP
        print(zf.namelist())
        
        # Read a specific CSV file from the ZIP
        df3 = pd.read_csv(zf.open('kiva_loans_part_2.csv'))
    return df3

# Load the data using the defined function
df3 = load_data()

data = pd.concat([df1, df2, df3])





