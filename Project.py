import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pandas.api.types import is_numeric_dtype
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

plt.savefig("Fig")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.precision', 4)

st.title("Alvin's Amazing Project")
st.markdown("Created by: Alvin Zou")
st.markdown("[Github Repo](https://github.com/happinyz/Math-10-Final-Project)")

st.image("https://cdn.vox-cdn.com/thumbor/E9lgLaSUj4IC2q_t34U38apDCjI=/0x151:1920x1156/fit-in/1200x630/cdn.vox-cdn.com/uploads/chorus_asset/file/22182760/patch_10_25_banner.jpg",
    use_column_width = True)

st.markdown("#### This project explores the data behind competitive _League of Legends_, a team-based strategy game where two teams of five powerful champions face off to destroy the other's base.")
st.markdown("Over the course of 10+ years, League of Legends has developed into the largest and most successful esport in the world. Thousands of professional matches are played each year across the globe. Our data will be focusing on the professional matches from the 2021 season.")

st.markdown("##### I used data and information from [Oracles Elixir](https://oracleselixir.com/), a League of Legends pro stats site. The **_2021 Match Data_** dataset can be found [here](https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com/2021_LoL_esports_match_data_from_OraclesElixir_20211205.csv).")
st.markdown("##### A definition of some of the columns can be found [here](https://oracleselixir.com/definitions).")

st.markdown("# Preliminary Data Cleaning")
st.write("Let's take a peek at what our dataset looks like currently.")

@st.cache
def getData():
    url = "https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com/2021_LoL_esports_match_data_from_OraclesElixir_20211204.csv"
    return pd.read_csv(url)

df = getData()
st.write(df.head(10))
st.write("")
st.write("There's a lot of columns we won't need for this project, so let's clean up our dataset a bit, removing the unnecessary/broken columns and changing certain values to strings so it's easier to work with.")
st.write("")

relevant_col = ["patch", "side", "position", "playername", "teamname", "champion", "result", "kills",
               "deaths", "assists", "dpm", "damageshare", "wardsplaced", "wpm", "visionscore", "vspm", "totalgold", 
                "earned gpm", "earnedgoldshare", "total cs", "cspm", "goldat10", "xpat10", "csat10", 
                "golddiffat10", "xpdiffat10", "csdiffat10", "killsat10", "assistsat10", "deathsat10", "goldat15", 
                "xpat15", "csat15", "golddiffat15", "xpdiffat15", "csdiffat15", "killsat15", "assistsat15", "deathsat15"]

df = df[relevant_col]
df = df[~df.isnull().any(axis=1)].copy()
df["patch"] = df["patch"].apply(lambda x : f'{x:.2f}')

players = df["playername"].unique()

st.write(df.head(10))
st.markdown("###### _Some columns are absolute indicators of a player's performance, and some are indicators of a player's performance relative to his opponent. For example, the 'diff' columns represent how a player performed against his opponent. The 'pm' columns represent that stat per minute. The 'at10/15' columns represent that stat at 10/15 minutes respectively. We use 10/15 minutes as thresholds as those times are highly influential parts of the game, and not all games go on longer than 20 minutes._")
st.write("Looks good to go!")

st.markdown("# Some Interesting Data ")
st.write("Patches are biweekly cycles where changes are made within the game to keep the game in a healthy state. Because of the impactful nature of patches, they can drastically affect how a champion or a player performs.")
st.header("Champion Winrate Across Patches")
st.write("Patches bring changes to the game that can impact a champion's performance. A champion can be made weaker or stronger in any given patch. Let's explore a champion's performance across the patches.")

champ_option = st.selectbox("Choose a champion:", np.sort(df["champion"].unique()))
champ_df = df[df["champion"] == champ_option]

def generateChampWinDataFrame():
    winrates = []
    games = []
    patch_list = np.sort(champ_df["patch"].unique())
    
    for patch in patch_list:
        patch_df = champ_df.loc[champ_df["patch"] == patch, ["result"]]
        winrates.append(patch_df["result"].sum()/patch_df.shape[0])
        games.append(patch_df.shape[0])
    
    temp_df = {'patch': patch_list, "winrate": winrates, "# games": games}
    return pd.DataFrame(data=temp_df)

champ_winrate_df = generateChampWinDataFrame()

if champ_option:
    champ_col1, champ_col2 = st.columns([3, 2])

    champ_chart = alt.Chart(champ_winrate_df).mark_line(
        point = alt.OverlayMarkDef(color = "red")
    ).encode(
        x = "patch:O",
        y = alt.Y("winrate:Q", scale = alt.Scale(domain=[0,1]))
    )

    with champ_col1:
        st.subheader(f"{champ_option} Winrate Across Patches")
        st.altair_chart(champ_chart, use_container_width = True)
    
    with champ_col2:
        st.subheader("Stats Per Patch")
        st.write(champ_winrate_df)

st.header("Player Winrate Across Patches")
st.write("Patches bring changes to the game that can significantly impact a team or an individual's performance. Let's explore a player's winrate across patches.")
st.write("Here are some example players you can input: Faker, Spica, huhi, Rekkles")

player_option = st.text_input("Choose a player (Case Sensitive)")

player_df = df[df["playername"] == player_option]

def generatePlayerWinDataFrame():
    winrates = []
    games = []
    patch_list = np.sort(player_df["patch"].unique())
    
    for patch in patch_list:
        patch_df = player_df.loc[player_df["patch"] == patch, ["result"]]
        winrates.append(patch_df["result"].sum()/patch_df.shape[0])
        games.append(patch_df.shape[0])
    
    temp_df = {'patch': patch_list, "winrate": winrates, "# games": games}
    return pd.DataFrame(data=temp_df)

player_winrate_df = generatePlayerWinDataFrame()

if player_option in players:
    player_col1, player_col2 = st.columns([3, 2])

    player_chart = alt.Chart(player_winrate_df).mark_line(
        point = alt.OverlayMarkDef(color = "red")
    ).encode(
        x = "patch:O",
        y = alt.Y("winrate:Q", scale = alt.Scale(domain=[0,1])),
    )

    with player_col1:
        st.subheader(f"{player_option}'s Winrate Across Patches")
        st.altair_chart(player_chart, use_container_width = True)
    
    with player_col2:
        st.subheader("Stats Per Patch")
        st.write(player_winrate_df)
elif player_option == "":
    pass
else:
    st.error(f"{player_option} is not a valid player.")

st.markdown("# Investigating with Machine Learning")
st.header("Logistic Regression")
st.write("Here we investigate the correlation of in-game performance with game results using logistic regression. I will be using train_test_split to train 5 different models with 5 different sets of predictors. Each set of predictors contains some information about a player's performance in the game.")
st.write("Model 1 uses every performance stat as predictors.")
st.write("Model 2 uses 'per minute' stats.")
st.write("Model 3 uses 10/15 minute differential stats.")
st.write("Model 4 uses absolute stats.")
st.write("Model 5 uses 10/15 minute personal performance stats.")
st.write("You can choose the size of our test data for the training splits using the slider.")

# Everything
ml_cols = ["kills","deaths", "assists", "dpm", "wardsplaced", "wpm", "visionscore", "vspm", 
                "earned gpm", "total cs", "cspm", "goldat10", "xpat10", "csat10", 
                "golddiffat10", "xpdiffat10", "csdiffat10", "killsat10", "assistsat10", "deathsat10", "goldat15", 
                "xpat15", "csat15", "golddiffat15", "xpdiffat15", "csdiffat15", "killsat15", "assistsat15", "deathsat15"]

# Per minute stats
ml_cols2 = ["wpm", "dpm", "vspm", "earned gpm", "cspm"]

# Differentials at 10/15 minutes
ml_cols3 = ["golddiffat10", "xpdiffat10", "csdiffat10", "golddiffat15", "xpdiffat15", "csdiffat15"]

# Absolutes
ml_cols4 = ["kills", "deaths", "assists", "wardsplaced", "visionscore", "total cs"]

# Personal performance at 10/15 minutes
ml_cols5 = ["goldat10", "xpat10", "csat10", "killsat10", "assistsat10", "deathsat10", "goldat15", 
                "xpat15", "csat15", "killsat15", "assistsat15", "deathsat15"]

st.header("Model Results and Analysis")

test_size_slider = st.slider("Choose proportion of dataset to include in test split.", min_value = 0.1, max_value = 0.9, step = 0.1)

def logistic_regression(predictors):
    X_train, X_test, y_train, y_test = train_test_split(df[predictors], df["result"], test_size = test_size_slider, random_state = 0)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

log_col1, log_col2, log_col3 = st.columns(3)
log_col4, log_col5 = st.columns(2)
log_col1.metric(label = "Model 1 Score (Every predictor)", value = "{:.2%}".format(logistic_regression(ml_cols)))
log_col2.metric(label = "Model 2 Score (Per minute stats)", value = "{:.2%}".format(logistic_regression(ml_cols2)))
log_col3.metric(label = "Model 3 Score (10/15 minutes diff)", value = "{:.2%}".format(logistic_regression(ml_cols3)))
log_col4.metric(label = "Model 4 Score (Absolute stats)", value = "{:.2%}".format(logistic_regression(ml_cols4)))
log_col5.metric(label = "Model 5 Score (Personal at 10/15 minutes)", value = "{:.2%}".format(logistic_regression(ml_cols5)))

st.write("It seems that Model 4, using the absolute stats as predictors, had the best performance. Model 1 and Model 2 also had strong scores." +
    "This makes sense, as absolute stats tell the outcome of the game. Per minute stats are also a strong indicator of the outcome of the game." +
    "The models containing stats at 10/15 minutes did not predict as well, likely because games are not entirely decided at 10/15 minutes, and even with an advantage at those times, teams can still struggle to close it out.")

def linear_regression(a, b):
    x = df[a].values
    y = df[b].values
    
    # x = np.reshape(dfl[0], (-1,2))
    # y = np.reshape(dfl[1], (-1,2))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    X_train= X_train.reshape(-1, 1)
    y_train= y_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    return (reg.intercept_, reg.coef_)

st.header("Linear Regression")
st.write("Here we investigate the correlation within a player's stats. Because of how League of Legends is structured, there is generally high correlation between any two of a player's stats. For example, if a player's kills are high, their damage share (compared to teammates) is generally high as well.")
st.write("Use the dropdown boxes below to select two stats to compare.")

lr_cols = ml_cols + ["damageshare", "earnedgoldshare"]

lr_option1 = st.selectbox("Choose the x variable.", sorted(lr_cols))
lr_option2 = st.selectbox("Choose the y variable.", sorted(lr_cols))

if lr_option1 == lr_option2:
    st.error("You cannot have the same x and y variable.")
elif lr_option1 and lr_option2:
    lr_col1, lr_col2 = st.columns([3,1.5])
    lr = linear_regression(lr_option1, lr_option2)

    fig, ax = plt.subplots()
    ax.plot(lr[0][0], lr[1][0][0])
    ax.scatter(df[lr_option1], df[lr_option2], s= 2)

    with lr_col1:
        st.subheader(f"Graph of {lr_option1} vs. {lr_option2}")
        st.pyplot(fig)
    with lr_col2:
        lr_cont1 = st.container()
        lr_cont2 = st.container()
        lr_cont1.metric(label = "Intercept", value = round(lr[0][0], 2))
        lr_cont2.metric(label = "Coefficient", value = round(lr[1][0][0], 2))    
