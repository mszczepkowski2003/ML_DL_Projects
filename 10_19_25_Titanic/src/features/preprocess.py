import pandas as pd
import numpy as np
import re
import sys
import os
from src.utils.helpers import one_ticket, gender_map, get_personal_title, add_ticket_repetition

def deterministic_fe(df):
    df = add_ticket_repetition(df)
    # df = get_personal_title(df)
    # df = gender_map(df)
    df['Family_size'] = df['SibSp'] + df['Parch']
    drop_cols = ['PassengerId', 'Cabin']
    return df.drop(drop_cols, axis=1)
