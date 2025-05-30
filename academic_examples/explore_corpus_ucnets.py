"""
Academic research example using private UCNets dataset.
This example demonstrates the methodology used in [An Empirical Investigation into the Utility of Large Language Models in Open-Ended Survey Data Categorization].
    
Note: This requires access to the private UCNets dataset.
Contact fischer1@berkeley.edu for data access requests.
"""

import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv
import os
import catllm as cat

_ = load_dotenv(find_dotenv())

os.chdir('/Users/chrissoria/Documents/Research/Categorization_AI_experiments')
current_directory = os.getcwd()

column1 = "a19i"
column2 = "a19f"
column3 = "e1b"

question1 = "Why did you move?"
question2 = "After this last move, what steps, if any, did you take in order to make new friends?"
question3 = "If you had a serious problem, like a life-threatening illness or possibly losing your home, do you feel that you have some relatives that you can rely on to help?"

user_categories1 = ["to start living with or to stay with partner/spouse",
                   "relationship change (divorce, breakup, etc)",
                   "the person had a job or school or career change, including transferred and retired",
                   "the person's partner's job or school or career change, including transferred and retired",
                   "financial reasons (rent is too expensive, pay raise, etc)",
                   "related specifically features of the home, such as a bigger or smaller yard"]

user_categories2 = ["Engaged with local religious institutions such as churches, synagogues, mosques, or other forms of religious communities.",
                   "Frequented local establishments like bars, cafes, shops, or malls to interact with individuals present in the vicinity.",
                   "Direct involvement in secular volunteering efforts, contributing through action and service rather than mere membership in volunteer groups.",
                   "Utilizing digital platforms such as online chats, internet networking websites, or dating apps to establish connections and friendships.",
                   "Engaged in informal, non-professional interactions and outings with colleagues to foster friendships.",
                   "Involvement in sports, exercise, or outdoor recreational activities through gyms, teams, or athletic clubs."]

user_categories3 = ["indicating a deep-seated faith or inherent knowledge in their family's willingness and ability to provide support during crises, often expressed as 'because I know they would help.''",
                   "indicating an unspoken duty or obligation that all family members have to support each other in times of crisis, usually expressed in very simple terms like 'because they're family.'",
                   "explicitly referring to the principle of family obligations, the cultural or traditional expectations of family support, or values that relatives inherently help each other, again applying to families in general.",
                   "referring to their particular family being particularly close, as well as emotional support and love within their own family (rather than references to families in general).",
                   "referencing past instances of receiving help or support from family members",
                   "They mention their relatives' financial situation or resources as a factor either enabling or preventing them from providing assistance."]

# read in the Excel files and sheets with survey data
file_configs = [
    {
        'path': "/Users/chrissoria/Documents/Research/UCNets_Classification/data/Raw_Cond_for_Coding_all_waves.xlsx",
        'sheet': "JOINT_DATA",
        'usecols': [column1],
        'rename_col': None,
        'final_col': column1,
        'var_name': 'UCNets_a19i'
    },
    {
        'path': "../UCNets_Classification/Hand_Coding_Surveys/a19fg/a19fg_Master.xlsx", 
        'sheet': "master_a19f",
        'usecols': ['Response'],
        'rename_col': {'Response': 'a19f'},
        'final_col': column2,
        'var_name': 'UCNets_a19f'
    },
    {
        'path': "../UCNets_Classification/Hand_Coding_Surveys/e1b/e1ab_Master.xlsx",
        'sheet': "Master_coded", 
        'usecols': ['Response'],
        'rename_col': {'Response': 'e1b'},
        'final_col': column3,
        'var_name': 'UCNets_e1b'
    }
]

dataframes = {}
for config in file_configs:
    df = pd.read_excel(
        config['path'], 
        engine='openpyxl', 
        sheet_name=config['sheet'], 
        usecols=config['usecols']
    )
    
    if config['rename_col']:
        df.rename(columns=config['rename_col'], inplace=True)
    
    dataframes[config['var_name']] = df

# Extract individual DataFrames for easier access
UCNets_a19i = dataframes['UCNets_a19i']
UCNets_a19f = dataframes['UCNets_a19f'] 
UCNets_e1b = dataframes['UCNets_e1b']

move_reason_categories = cat.explore_corpus(
    survey_question=question1,
    survey_input=UCNets_a19i[column1],
    api_key=os.environ.get("OPENAI_API_KEY"),
    to_csv=True,
    filename="data/phase5/UCNets_a19i_exploration.csv",
    cat_num=15,
    divisions=20
)

new_friends_categories = cat.explore_corpus(
    survey_question=question2,
    survey_input=UCNets_a19f[column2],
    api_key=os.environ.get("OPENAI_API_KEY"),
    to_csv=True,
    filename="data/phase5/UCNets_a19f_exploration.csv",
    cat_num=15,
    divisions=20
)

social_support_categories = cat.explore_corpus(
    survey_question=question3,
    survey_input=UCNets_e1b[column3],
    api_key=os.environ.get("OPENAI_API_KEY"),
    to_csv=True,
    filename="data/phase5/UCNets_e1b_exploration.csv",
    cat_num=15,
    divisions=20
)