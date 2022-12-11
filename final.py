import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
import pandas as pd
import numpy as np
#from sklearn import preprocessing
#import matplotlib.pyplot as plt 
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import xgboost as xgb
from collections import Counter
from imblearn.over_sampling import RandomOverSampler  
#from sklearn.model_selection import RandomizedSearchCV
#import calendar
import time
#from datetime import datetime 





#######################################################################

# Questions for Monday:
# - If I add a cache function, will it still scrape the web everytime the page is run
# - Fix the map

# Build the app interface

# Sidebar
st.sidebar.image("ncaa.png")

page = st.sidebar.selectbox('Select Page:', 
    ['Head-to-Head', 'Top Teams', 'Map'])

with st.sidebar:

    # st.write(
    #     "Future work: Enable time stamps to only pull data/run models once a day"
    # )

    st.write(
        "Disclaimer: I do not own the rights to any of the data used to create this dashboard or any of the models. " 
        "All data has been pulled from https://www.sports-reference.com/cbb/"
    )
    st.write(
        "If there are any concerns, please reach out to ctpage95@gmail.com for removal."
    )



st.header("NCAA Champion Prediction")



########################################################################

data = pd.read_csv("bbdatafull.csv")
prob_data = pd.read_csv("probabilities.csv")
model_probs = pd.read_csv("model_probs.csv")


if page == 'Head-to-Head':
    st.write(
    "In this module, you will be able to pull real time data and run an analysis to determine which teams are most likely to win in a Head-to-Head match-up"
)

    # Create a button to run the most recent data

    st.subheader(
        "Pull the most updated data:"
    )
    st.text(
        "This will scrape the data from the source website (avoid rerunning too many times)."
    )
    rerun_data =  st.button("Reload Most Recent Data")

    if rerun_data:

        # Set up the loading bar
        my_bar = st.progress(0)

        for percent_complete in range(100):
            time.sleep(.1)
            my_bar.progress(percent_complete + 1)

        # Scrape the data from the web and load it into
        years = list(range(2023, 2024))
        all_data = pd.DataFrame()
        for i in range(0,len(years)):
            table = pd.read_html(f"https://www.sports-reference.com/cbb/seasons/{years[i]}-school-stats.html",
                            header = 1)
            df = pd.DataFrame(table[0])
            df = df[df['Rk'].notna()]
            df = df[df['Rk'] != 'Rk']
            df['School'] = df['School'].str.replace("NCAA","")
            df['Year'] = years[i]
            #df = df.loc[:,df.notna().any(axis=0)]
            all_data = all_data.append(df, ignore_index=True)
            all_data.to_csv("AllData2023.csv", header = True)

    # Read in the data
    df = pd.read_csv("AllData2023.csv")
    df_train = pd.read_csv("bbdatafull.csv")

    # Clean up the data
    df['IsWinner'] = 0
    df['HasNaismithPlayer'] = 0

    df = df.rename(columns={'W.1': 'Conference Wins',
                            'L.1': 'Conference Loses',
                            'W.2': 'Home Wins',
                            'L.2': 'Home Loses',
                            'W.3': 'Road Wins',
                            'L.3': 'Road Loses',
                            'Tm.': 'Total Points',
                            'Opp.': 'Total Opponent Points'})


    df = df.drop(['Unnamed: 8','Unnamed: 11','Unnamed: 14','Unnamed: 17', 'Unnamed: 20', 
                    'MP'], 1)


    df_train = df_train.rename(columns={'W.1': 'Conference Wins',
                            'L.1': 'Conference Loses',
                            'W.2': 'Home Wins',
                            'L.2': 'Home Loses',
                            'W.3': 'Road Wins',
                            'L.3': 'Road Loses',
                            'Tm.': 'Total Points',
                            'Opp.': 'Total Opponent Points'})


    df_train = df_train.drop(['Unnamed: 0.1', 'Unnamed: 8','Unnamed: 11','Unnamed: 14','Unnamed: 17', 'Unnamed: 20', 
                    'MP', 'Winning team', 'NaismithCollege', 'Player', 'Conference'], 1)


    df = df.append(df_train, ignore_index=True)
    df['Year'] = df['Year'].astype(str)

    df_test = df.loc[(df['Year'] == '2023')]
    df_train = df.loc[(df['Year'] != '2023')]

    df_train = df_train.drop(['Year'],1)
    df_test = df_test.drop(['Year'],1)

    df_train = df_train.drop(['W', 'L', 'W-L%', 'FG', '3P', 'FT'], 1)
    df_test = df_test.drop(['W', 'L', 'W-L%', 'FG', '3P', 'FT'], 1)

    df_train = df_train.dropna()
    df_test = df_test.dropna()

    y_train = df_train['IsWinner']
    X_train = df_train.drop(['IsWinner','School'],1)

    y_test = df_test['IsWinner']
    X_test = df_test.drop(['IsWinner','School'],1)

    X_train_scaled = (X_train-X_train.mean())/X_train.std()
    X_test_scaled = (X_test-X_test.mean())/X_test.std()


    # Create the button to have option to rerun the model
    st.subheader(
        "If the data has been recently updated, rerun the model for updated results:"
    )
    model_results = st.button("Rerun Model")

    if model_results:
        # with st.spinner('Loading...'):
        #     time.sleep()
        #     st.success('Done!')

        my_bar = st.progress(0)

        for percent_complete in range(100):
            time.sleep(.1)
            my_bar.progress(percent_complete + 1)

        # # Randomly over sample the minority class
        ros = RandomOverSampler(random_state=42)
        X_ros, y_ros= ros.fit_resample(X_train_scaled, y_train)

        # Create the model
        xg_param_dict = {'objective': ['reg:linear'],
                        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
                            'colsample_bytree': [0.01,0.05,0.1,0.5],
                            'max_depth': [5,10,15,25],
                            'n_estimators': [1,5,10,25,50]}

        xg = xgb.XGBRegressor()

        # Instantiate the grid search model
        xg_rand_search = RandomizedSearchCV(estimator = xg , param_distributions = xg_param_dict, cv = 3, n_jobs = -1)

        xg_rand_search.fit(X_ros, y_ros.values.ravel())


        xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.1, learning_rate = 0.005,
                        max_depth = 5, n_estimators = 10)

        xg_reg.fit(X_ros,y_ros)

        xg_y_pred = xg_reg.predict(X_test_scaled)
        xg_predictions = [round(value) for value in xg_y_pred]

        df_test_name = pd.DataFrame(df_test["School"])
        xgb_probabilities = pd.DataFrame(xg_y_pred/(xg_y_pred.sum()))
        df_test_name["xgb_probabilities"] = xgb_probabilities.values


        df_test_name.to_csv("model_probs.csv")

    st.write(
        "\n",
        "\n",
        "\n"
    )

    st.subheader(
        "Choose two teams to compare:"
    )

    col1, col2 = st.columns(2)

    all_schools = data['School'].unique()

    with col1:
        team1 = st.selectbox(
            "Select Team One", all_schools
        )

    with col2:
        team2 = st.selectbox(
            "Select Team Two", all_schools
        )

    prob_data_filtered = model_probs[model_probs["School"].isin([team1, team2])]
    prob_data_filtered.to_csv("prob_data_filtered.csv")


    team1_prob = float(prob_data_filtered[prob_data_filtered["School"].isin([team1])]["xgb_probabilities"])
    team2_prob = float(prob_data_filtered[prob_data_filtered["School"].isin([team2])]["xgb_probabilities"])

    if team1 == team2:
        st.subheader("Please select two different teams")
    elif team1_prob > team2_prob:
        st.subheader(team1 + " " + "Wins!")
    else:
        st.header(team2 + " " + "Wins!")

    fig_target = go.Figure(data=[go.Pie(labels=prob_data_filtered['School'],
                                        values=prob_data_filtered["xgb_probabilities"],
                                        hole=.3)])

    st.plotly_chart(fig_target, use_container_width=True)



if page == 'Top Teams':
    schooldata = pd.read_csv("schooldata.csv")
    confs_table = pd.read_csv("SchoolsWithConfs.csv")
    confs_dict = dict(zip(confs_table['School'], confs_table['Conf']))
    states_dict = dict(zip(schooldata['School'], schooldata['State']))
    model_probs = pd.read_csv("model_probs.csv")
    model_probs = model_probs.sort_values(['xgb_probabilities'], ascending=[False])
    #model_probs = model_probs.drop('Unnamed: 0')
    idx = 0
    model_probs.insert(idx, 'index', value=range(len(model_probs)))
    model_probs['index'] = model_probs['index']+1
    model_probs['Conference'] = model_probs['School'].map(confs_dict)
    model_probs['State'] = model_probs['School'].map(states_dict)
    model_probs = model_probs[['index', 'School', 'Conference', 'State']]
    model_probs = model_probs.rename(columns={'index': 'Rank'})

    all_schools = model_probs['School'].unique()
    all_confs = model_probs['Conference'].unique()
    all_states = model_probs['State'].unique()

    col1, col2 = st.columns(2)

    with col1:
        counter = st.slider('Select number of teams', 5, len(model_probs), 5)

        for i in range(0, counter):
            st.write(
                f"{i+1}. ", model_probs['School'].tolist()[i], "\n"
                # With probs - f"{i+1}. ", model_probs['School'].tolist()[i],model_probs['xgb_probabilities'][i], "\n"
            )
    
    with col2:
        comp_method = st.selectbox("Choose Comparison Method:", ["All", "School", "Conference", "State"])
        if comp_method == "All":
            filtered_df = model_probs
        elif comp_method == "School":
            schools_selected = st.multiselect("Choose Individual Teams for Comparison:", all_schools)
            filtered_df = model_probs[model_probs["School"].isin(schools_selected)]
        elif comp_method == "Conference":
            schools_selected = st.multiselect("Choose Conferences for Comparison:", all_confs)
            filtered_df = model_probs[model_probs["Conference"].isin(schools_selected)]
        else:
            schools_selected = st.multiselect("Choose State for Comparison:", all_states)
            filtered_df = model_probs[model_probs["State"].isin(schools_selected)]            

        st.write(
            filtered_df
        )

schooldata = pd.read_csv("schooldata.csv")

if page == 'Map':

    metric = st.selectbox("Choose the Metric to Visualize:", 
                            ["Avg. Win-Loss Percentage", "Total Championship Wins", "Final Four Appearances",
                            "Total Tournament Appearances", "Regular Season Conference Champions"])

    #schooldata_gb = schooldata.groupby('State')
    if metric == "Avg. Win-Loss Percentage":
        schooldata_gb = schooldata.groupby(['State_abrv'])['W-L%'].mean().reset_index(name='Measure')
    elif metric == "Total Championship Wins":
        schooldata_gb = schooldata.groupby(['State_abrv'])['NC'].sum().reset_index(name='Measure')
    elif metric == "Final Four Appearances":
        schooldata_gb = schooldata.groupby(['State_abrv'])['FF'].sum().reset_index(name='Measure')
    elif metric == "Total Tournament Appearances":
        schooldata_gb = schooldata.groupby(['State_abrv'])['NCAA'].sum().reset_index(name='Measure')
    elif metric == "Regular Season Conference Champions":
        schooldata_gb = schooldata.groupby(['State_abrv'])['CREG'].sum().reset_index(name='Measure')    

    fig = go.Figure(data=go.Choropleth(
        locations=schooldata_gb['State_abrv'], # Spatial coordinates
        z = schooldata_gb['Measure'], # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Blues',
        colorbar_title = "NCAA Tournament Wins",
    ))


    fig.update_layout(
        title_text = f"{metric} by State",
        geo_scope='usa', # limite map scope to USA
    )

    st.plotly_chart(fig)
    
    
