# Predicting NCAA Outcomes

Streamlit App: https://ctpage95-ncaa-final.streamlit.app/

## Research Questions:
1. Are we able to accurately predict the outcome of the NCAA tournament?
2. Are we able to predict the outcome of any given head to head matchup within all teams in the NCAA regular season?

*Note: Modeling was complete to determine the probability that each team has to win the National Champiosnhip in March 2023.*

## In this app:
- In this app you will be able to navigate across three pages:
  1. Head-to-Head
  2. Top Teams
  3. Map (A map of statistics by state)

### Head-to-Head
- This is where you are able to compare data across all teams and determine who is more likely to win the match-up.

- The first option you have is to rerun the most current data. By clicking this button, it will scrape whatever the current data is from the data source (https://www.sports-reference.com/cbb/). This data is updated on a regular basis and will modify outcomes as more data comes in for the season. We want to avoid running this too much, as we want to avoid making excessive calls to the site. Once a day is sufficient.

- The next option we have is to rerun the model. Once we have scraped the most recent data, we will want to rerun the model to calculate new probabilities given the most recent games and data. We don't need to worry about excessively rerunning this as it only calls on the code, but if there is no updated data, we will not need to rerun the model. Only rerun if there is new data.

- Once we have the new data and the model has been retuned, we can now move on to our head-to-head. Here, we can select two different teams to determine who is mroe likely to win. Again, the model is built upon likeliness to win the championship and does not take into consideration opponent match-ups. However, overall probability to win the tournament is a good indicator into winning individual match-ups.


### Top Teams
In the Top Teams page, you will be able to visualize a list of all of the top teams given the most recent iteration of the model. This page is split into two sections:
  1. Top Teams list
  2. Team Comparisons

#### Top Teams List
The first section comes with a slider where you can display the top teams. The slider determines how many teams you would like to see. It is by default, sorted so that the teams that are most likely to win are at the top and then populate in descending order.

#### Team Comparison
This section gives you the ability to choose a comparison method. You can choose to compare teams based on individual team names, conference, and state. Once you select the comparison method in the drop down, another drop down will appear with the options. The ranks are in the table so you can see how each team in the subset that you select compare to one another.


