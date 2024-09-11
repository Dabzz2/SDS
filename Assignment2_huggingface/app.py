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
import statsmodels.formula.api as smf
from duckduckgo_search import DDGS

st.title("Assignment 2: Bla bla bla")        
st.text("""We have the following research question that we aim to 
        investigate and attempt to answer: 
        Do men borrow more money than women?""")    

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
data.drop(['tags'], axis = 'columns', inplace = True)
data.dropna(inplace = True)

valid_genders = ['male', 'female']
data = data[data['borrower_genders'].isin(valid_genders)]

st.subheader("""Cleaning data""")
st.text("""We have eliminated the column tags, as well as the associated tags, 
        since they merely consisted of quotations such as “User favorite,” 
        among others. Additionally, these columns contained a 
        significant amount of missing data (NAs).""")   

st.text(f'We just saved {(len(data) / 671205) * 100} % of the data!')
st.text(f'Number of remaining {len(data)} rows')



st.subheader("Basic statistics for key variables")
st.dataframe(data[['loan_amount','term_in_months','lender_count']].agg(['mean','var','min','median','max','sum']))

#Laver en forklaring med chatgpt herpå til


st.subheader("Statistics for key variables based on gender and sector")

st.text('Pick what to group by')
selected1 = st.selectbox("Select variable1",['loan_amount','term_in_months','lender_count'])

st.text('Pick what statistic to inspect')
selected2 = st.selectbox("Select variable2",['mean','var','min','median','max','sum','std'])


st.table(data.groupby(['borrower_genders','sector'])[selected1].agg(selected2))




st.subheader("Visualizations")
correlation_matrix = data[['loan_amount', 'term_in_months', 'lender_count']].corr(method='spearman')
# Dropdown to select the type of visualization
visualization_option = st.selectbox(
    "Select Visualization 🎨", 
    ["Number of loans in sectors Distribution",
     "Loan Amount Distribution by Gender", 
     "Loan Amount Distribution by Sector Type", 
     "KDE Plot: Loan amount based on sectors", 
     "Correlation Matrix of Loan amount, length of loan and amount of lenders"]
)

# Visualizations based on user selection
if visualization_option == "Number of loans in sectors Distribution":
    plt.figure(figsize=(12, 6))

    # Number of loans in sectors Distribution
    sns.histplot(data['sector'], kde=True)
    plt.title('Number of loans in sectors Distribution')

    plt.xlabel('Sector')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()
    st.pyplot(plt, use_container_width=True)

elif visualization_option == "KDE Plot: Loan amount based on sectors":
    # KDE plot for Distance from Home based on Attrition
    sns.kdeplot(data = data, x = 'loan_amount', hue = 'sector', clip = (0,4000))
    plt.title('KDE Plot: Loan amount based on sectors')
    st.pyplot(plt)

elif visualization_option == "Loan Amount Distribution by Gender":
    # Bar chart for attrition by job role
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='borrower_genders', y='loan_amount', data=data, order=data['borrower_genders'].value_counts().index)
    plt.title('Loan Amount Distribution by Gender')
    plt.xlabel('Borrower Gender')
    plt.ylabel('Loan amount')
    plt.xticks(rotation=45)
    plt.ylim(0, 3000)
    st.pyplot(plt, use_container_width=True)

elif visualization_option == "Loan Amount Distribution by Sector Type":
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='sector', y='loan_amount', data=data, order=data['sector'].value_counts().index)
    plt.title('Loan Amount Distribution by Sector Type')
    plt.xlabel('Sector')
    plt.ylabel('Loan amount')
    plt.xticks(rotation=45)
    plt.ylim(0, 12500)
    st.pyplot(plt, use_container_width=True)

elif visualization_option == "Correlation Matrix of Loan amount, length of loan and amount of lenders":
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm').set_title('Correlation Matrix of Loan amount, length of loan and amount of lenders')
    st.pyplot(plt)




st.subheader("Regression")
data['gender_binary'] = data['borrower_genders'].apply(lambda x: 1 if x == 'male' else 0)
model = smf.ols('loan_amount ~gender_binary+ lender_count+ term_in_months', data = data).fit()
st.write(model.summary())

st.subheader("""We can conclude with 73% significans that men borrow more money than women.""")


st.subheader("The world-known economist answering the OLS-regression")

results = DDGS().chat(
    "You are an extremely good economist with lots of knowledge about econometrics. "
    "Interpret the following OLS results: " + str(model.summary()) + 
    ". Specifically, answer if men borrow more money than women.",
    model='gpt-4o-mini')
st.write(results)
