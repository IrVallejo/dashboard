import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder,
                                   LabelEncoder,
                                   MinMaxScaler)
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             classification_report,
                             ConfusionMatrixDisplay,
                             RocCurveDisplay,
                             DetCurveDisplay)
from sklearn.tree import plot_tree



def load_data():
    url = 'https://raw.githubusercontent.com/rfordatascience/' + \
    'tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv'
    df = pd.read_csv(url)
    df_interim = df.copy()
    df_interim = df_interim[['total_cup_points',
                                'species',
                                'country_of_origin',
                                'variety',
                                'aroma',
                                'aftertaste',
                                'acidity',
                                'body',
                                'balance',
                                'altitude_mean_meters', 'moisture']]
    df_interim = df_interim.dropna()
    df_interim["species"] = pd.Categorical(df_interim["species"])
    df_interim["country_of_origin"] = pd.Categorical(df_interim["country_of_origin"])
    df_interim["variety"] = pd.Categorical(df_interim["variety"])
    df_interim["specialty"] = df_interim["total_cup_points"].apply(lambda x: "yes" if x>82.43 else "no")
    df_interim["altitude"] = df_interim["altitude_mean_meters"].apply(lambda x:1300 if x > 1000 else x)
    df_interim["specialty"] = pd.Categorical(df_interim["specialty"])
    df_interim = df_interim[df_interim["acidity"]!=0].copy()
    df_interim = df_interim[df_interim["altitude_mean_meters"]<=10000].copy()
    df_interim = df_interim.drop('total_cup_points', axis=1)
    df = df_interim.copy()
    return df
df_ch = load_data()
df_ch

df_ch.drop(["species","country_of_origin","variety"], axis=1).describe().T

fig1 = sns.pairplot(data = df_ch[["aroma","aftertaste"]])
st.pyplot(fig1)




#Analisis univariado
#fig2 = df_train.hist(figsize=(15,15))
#st.pyplot(fig2)

# Step 4 Univaried Analysis
fig3 = sns.countplot(data= df_ch, y= 'species')
st.pyplot(fig3)

fig4 = sns.countplot(data= df_ch, y= 'country_of_origin')
st.pyplot(fig4)

fig5 = sns.countplot(data= df_ch, y= 'variety')
st.pyplot(fig5)

fig6 = sns.countplot(data= df_ch, y= 'specialty')
st.pyplot(fig6)











df_ch = load_data()
st.write(df_ch.shape[0])

st.title('coffe dashboard')
st.dataframe(df_ch)
fig1 = px.histogram(df_ch, x='aroma')
st.plotly_chart(fig1)
fig2 = sns.pairplot(data=df_ch, vmin=1, vmax=1, annot=True)

st.pyplot(fig2 = sns)

#st.write('Hello word')
#x=st.slider('Select a value: ', min_value=-5, max_value=5, value =0)
#st.write(x, 'Squared is', x**2)

#y=st.slider('Select another value: ', min_value=-5, max_value=5, value =0)
#st.write(x, 'Cubic', y**3)