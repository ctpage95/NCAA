import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
import pandas as pd
import numpy as np
#from sklearn import preprocessing
#import matplotlib.pyplot as plt 
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go








#######################################################################

# Questions for Monday:
# - If I add a cache function, will it still scrape the web everytime the page is run
# - Fix the map

# Add a button to refresh the data


# # Get the data and create a logistic regression model

# # Scrape data off of the web - exported to CSV

# # years = list(range(2023, 2024))
# # all_data = pd.DataFrame()
# # for i in range(0,len(years)):
# #     table = pd.read_html(f"https://www.sports-reference.com/cbb/seasons/{years[i]}-school-stats.html",
# #                      header = 1)
# #     df = pd.DataFrame(table[0])
# #     df = df[df['Rk'].notna()]
# #     df = df[df['Rk'] != 'Rk']
# #     df['School'] = df['School'].str.replace("NCAA","")
# #     df['Year'] = years[i]
# #     #df = df.loc[:,df.notna().any(axis=0)]
# #     all_data = all_data.append(df, ignore_index=True)
    
# # # Export data to CSV to avoid rerunning code every time
# # all_data.to_csv("AllData2023.csv", header = True)

# # Read in the full data set to train the model on - remove missing values
# df = pd.read_csv("bbdatafull.csv")
# df = df.loc[:, df.columns.isin(['W-L%', 'FG%', 'FT%', 'BLK','AST', 'STL', 'TRB', 'IsWinner'])]
# df = df.dropna()

# # Initiate the model
# model = LogisticRegression(solver='liblinear', random_state=0)

# # Create the X and y variables to run the model on
# X = df.loc[:, df.columns.isin(['W-L%', 'HasNaismithPlayer', 'FG%', 'FT%', 'BLK', 'AST', 'STL', 'TRB'])]
# X = X.dropna()
# y = df.loc[:, df.columns=='IsWinner']

# # Fit the model
# model.fit(X,y)

# # Create a new dataframe with the newly gathered 2023 season data
# # We will also create the X_test variable to feed into the model
# new_df = pd.read_csv("AllData2023.csv")
# X_test = new_df.loc[:, new_df.columns.isin(['W-L%', 'HasNaismithPlayer', 'FG%', 'FT%', 'BLK', 'AST', 'STL', 'TRB'])]

# # Calculate the probabilities on the 2023 season data.
# np.set_printoptions(suppress=True)
# probs = model.predict_proba(X_test)

# # Create a dataframe that attaches the probabilities to the dataset
# probs_df = pd.DataFrame(data = probs[:,1:], columns = ['prob'])
# new_df['probs'] = pd.Series(probs_df['prob'])

# # Export to a csv file
# #new_df.to_csv("2023probs.csv", header = True)

#######################################################################


page = st.sidebar.selectbox('Select Page:', 
    ['Head-to-Head', 'Top 10', 'Map'])

with st.sidebar:
    st.write("Source: https://www.sports-reference.com/cbb/")


st.header("NCAA Champion Prediction")

data = pd.read_csv("bbdatafull.csv")
prob_data = pd.read_csv("probabilities.csv")

# models = {
#     "Logistic Regression":"logit_probabilities",
#     "XGBoost":"xgb_probabilities"
#     }

st.write("Model Used: XGBoost")

# st.write(
#     data.head(20)
# )

if page == 'Head-to-Head':
    col1, col2 = st.columns(2)

    all_schools = data['School'].unique()

    with col1:
        team1 = st.selectbox(
            "Select Team One", all_schools
        )
        # with st.form('Form1'):
        #     st.selectbox('Select Team 1', all_schools, key=1)
        #     team1 = st.selectbox('Submit Team 1')
        #     submitted = st.form_submit_button('Submit')

    with col2:
        team2 = st.selectbox(
            "Select Team Two", all_schools
        )


#     with st.form('Form2'):
#         st.selectbox('Select Team 2', all_schools, key=2)
#         team2 = st.selectbox('Submit Team 2')


# duke = data[data["School"] == "Duke"]
# unc = data[data["School"] == "North Carolina"]
# #combined = pd.concat([team1, team2], axis=0)
# data2 = data[(data["School"] == str("Duke")) | (data["School"] == str("North Carolina"))]

# scatter = alt.Chart(data2).mark_line().encode(
#     x="Year:O", y="W-L%", color="School"
# )

# st.altair_chart(scatter, use_container_width=True)

#prob_data_filtered = prob_data[(prob_data["School"] == team1) | (prob_data["School"] == team2)]


#prob_data_filtered = prob_data[prob_data["School"].isin(["Duke", "North Carolina"])]
    prob_data_filtered = prob_data[prob_data["School"].isin([team1, team2])]

    fig_target = go.Figure(data=[go.Pie(labels=prob_data_filtered['School'],
                                        values=prob_data_filtered["xgb_probabilities"],
                                        hole=.3)])

    st.plotly_chart(fig_target, use_container_width=True)

    team1_prob = float(prob_data[prob_data["School"].isin([team1])]["xgb_probabilities"])
    team2_prob = float(prob_data[prob_data["School"].isin([team2])]["xgb_probabilities"])

    # st.write(
    #     team1_prob,
    #     team2_prob
    # )

    if team1 == team2:
        st.header("Please select two different teams")
    elif team1_prob > team2_prob:
        st.header("<h1 style='text-align: center;", team1 + " " + "Wins!")
    else:
        st.header(team2 + " " + "Wins!")


# st.write(prob_data_filtered.head())




if page == 'Top 10':

    st.write(
        "1. Choose to be able to toggle between model results?", "\n",
        "2. Clean it up to look more professional", "\n",
    )

    st.header("Top 10 Teams To Win:")
    st.write(
        "1. ", prob_data['School'].tolist()[0],prob_data['probs_norm'][0], "\n",
        "2. ", prob_data['School'].tolist()[1],prob_data['probs_norm'][1], "\n",
        "3. ", prob_data['School'].tolist()[2],prob_data['probs_norm'][2], "\n",
        "4. ", prob_data['School'].tolist()[3],prob_data['probs_norm'][3], "\n",
        "5. ", prob_data['School'].tolist()[4],prob_data['probs_norm'][4], "\n",
        "6. ", prob_data['School'].tolist()[5],prob_data['probs_norm'][5], "\n",
        "7. ", prob_data['School'].tolist()[6],prob_data['probs_norm'][6], "\n",
        "8. ", prob_data['School'].tolist()[7],prob_data['probs_norm'][7], "\n",
        "9. ", prob_data['School'].tolist()[8],prob_data['probs_norm'][8], "\n",
        "10. ", prob_data['School'].tolist()[9],prob_data['probs_norm'][9], "\n",
    )

schooldata = pd.read_csv("schooldata.csv")

# st.write(
#     schooldata.head()
# )


if page == 'Map':

    st.write(
        "1. Be able to toggle between different variables. Use dictionary to change variable names", "\n",
        "2. Update titles using an fstring to update according to which variable is measured", "\n",
        "3. Fix the map to get color working", "\n"
    )

    st.write(schooldata.head())

    fig = go.Figure(data=go.Choropleth(
        locations=schooldata['State_abrv'], # Spatial coordinates
        z = schooldata['NC'], # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Blues',
        colorbar_title = "NCAA Tournament Wins",
    ))


    fig.update_layout(
        title_text = 'Total (fstring variable) by State',
        geo_scope='usa', # limite map scope to USA
    )

    st.plotly_chart(fig)

    # st.write(
    #     gb_df = schooldata.groupby('State_abrv')
    #     gb_df.first()
    # )

    st.code(
        '''
            fig = go.Figure(data=go.Choropleth(
            locations=schooldata['State_abrv'], # Spatial coordinates
            z = schooldata['NC'], # Data to be color-coded
            locationmode = 'USA-states', # set of locations match entries in `locations`
            colorscale = 'Blues',
            colorbar_title = "NCAA Tournament Wins",
        ))
        ''', language='python')







