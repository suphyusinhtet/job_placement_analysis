import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error

import streamlit as st

df = pd.read_csv(r'job_placement.csv')

print ("Dataset Length: ", len(df))
print ("Dataset Shape: ", df.shape)
df

# df.drop("id", axis=1, inplace=True)
df["salary"].fillna(0, inplace=True)
df["years_of_experience"].fillna(0, inplace=True)
print ("Dataset Shape: ", df.shape)
df["years_of_experience"].fillna(df["years_of_experience"].median(), inplace=True)
df["college_name"] = df["college_name"].apply(lambda x: x.split("--")[0])


# """#Total values counts of Placed and Not Palced"""
# st.subheader('Total values counts of Job Placement Status')
# val = df['placement_status'].value_counts()
# st.dataframe(val)

"""#Total values counts of Gender, Degree, Stream, College, Placement Status"""
st.subheader('Total values counts of Gender, Degree, Stream, College, Placement Status')
# titles = ["gender", "degree", "stream", "college_name","placement_status"]
titles = ["gender", "degree", "stream", "years_of_experience","college_name","placement_status"]

# Set up the layout
col1, col2 = st.columns(2)

# Iterate through titles and display value_counts
for i, title in enumerate(titles):
    counts = df[title].value_counts()
    if i < 4:
        col1.header(title)
        col1.write(counts)
    else:
        col2.header(title)
        col2.write(counts)

# #Filter colleges that have only one person
# college_counts = df['college_name'].value_counts()
# valid_colleges = college_counts[college_counts > 1].index
# filtered_df = df[df['college_name'].isin(valid_colleges)]

"""#Visualize gender of each Placement Status"""
st.subheader('Gender for Placement Status')
figure = px.parallel_categories(df[[titles[0], titles[-1]]])
st.write(figure)
crosstab_result = pd.crosstab(df[titles[0]], df[titles[-1]])
st.write(crosstab_result)

"""#Visualize stream of each Placement Status"""
st.subheader('Stream for Placement Status')
# fig = px.parallel_categories(df[[titles[2], titles[-1]]])
# st.write(fig)
# crosstab_result_1 = pd.crosstab(df[titles[2]], df[titles[-1]])
# st.write(crosstab_result_1)
res = df.groupby(['stream','placement_status'])['placement_status'].size().reset_index(name='placement_size')
agegp = px.bar(data_frame=res, x='stream', y='placement_size', color='placement_status', title='Placement Status by Stream').update_layout(xaxis_title='stream', yaxis_title='placement size')
st.plotly_chart(agegp)

"""#Visualize college of each Placement Status"""
st.subheader('College for Placement Status')
res1 = df.groupby(['college_name','placement_status'])['placement_status'].size().reset_index(name='placement_size')
# st.write(res1)
agegp = px.bar(data_frame=res1, x='college_name', y='placement_size', color='placement_status', title='Placement Status by College').update_layout(xaxis_title='college', yaxis_title='placement size')
st.plotly_chart(agegp)

res2 = df.groupby(['age','placement_status'])['placement_status'].size().reset_index(name='placement_size')
agegp = px.bar(data_frame=res2, x='age', y='placement_size', color='placement_status', title='Placement Status by Age').update_layout(xaxis_title='age', yaxis_title='placement size')
st.plotly_chart(agegp)

res3 = df.groupby(['years_of_experience','placement_status'])['placement_status'].size().reset_index(name='placement_size')
agegp = px.bar(data_frame=res3, x='years_of_experience', y='placement_size', color='placement_status', title='Placement Status by Years of experience').update_layout(xaxis_title='years of experience', yaxis_title='placement size')
st.plotly_chart(agegp)

res4 = df.groupby(['gpa','placement_status'])['placement_status'].size().reset_index(name='placement_size')
agegp = px.bar(data_frame=res4, x='gpa', y='placement_size', color='placement_status', title='Placement Status by GPA').update_layout(xaxis_title='GPA', yaxis_title='placement size')
st.plotly_chart(agegp)

def plots(df, name, nums, axes):
    grouped = df.groupby(name)
    means = grouped[nums].mean()

    # Plotting barplot using Streamlit
    axes[0].bar(means.index, means)
    for i, val in enumerate(means):
        axes[0].text(i, val, str(round(val, 2)), ha='center', va='bottom', fontsize=10, rotation=90)
    axes[0].set_xticks(range(len(means)))
    axes[0].set_xticklabels(means.index, rotation=90, fontsize=10)
    axes[0].set_ylabel(nums)

    # Plotting KDE plots for each group using Streamlit
    for group_name, group_data in grouped:
        sns.kdeplot(group_data[nums], ax=axes[1], label=group_name)

    # Plotting boxplot using Streamlit
    sns.boxplot(data=df, x=name, y=nums, ax=axes[2])
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=90, fontsize=10)
    axes[2].set_ylabel(nums)

# Filter the dataframe for placed students
temp_df = df[df["placement_status"] == "Placed"].copy()

# Set up the layout
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 7))

# Call the plots function
plots(temp_df, "placement_status", "salary", axes)

# Display the plots in Streamlit
st.pyplot(fig)

le = LabelEncoder()

for i in ["gender", "placement_status"]:
    df[i] = le.fit_transform(df[i])

st.write(df)
st.subheader('Correlation Matrix')
#Create a correlation matrix to show relationship between select variables
corr_matrix = df[["gender","age","gpa","salary","years_of_experience","placement_status"]].corr()
corr_matrix

st.subheader('Correlation between Selected Variables')
fig, ax = plt.subplots(figsize=[8, 8])
sns.heatmap(corr_matrix, annot=True, cmap='Reds', ax=ax)
# ax.set_title("Correlation between Selected Variables")
st.pyplot(fig)
