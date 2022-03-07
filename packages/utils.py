from typing import Union
import pandas as pd

def most_recent_businesses_df(
    df1: pd.DataFrame,
    df2: Union[pd.DataFrame, None] = None,
    on: Union[str, None] = None,
    remove_na: bool = False
) -> pd.DataFrame:
    """
    A utility function to quickly get the most recent business holdings with add. options,
    also business rating is automatically converted to a float
    Params:
        :Param df1: pd.DataFrame, the left dataframe to be joined
        :Param df2: pd.DataFrame, the right dataframe to be joined
        :Param on: str, the column on which to join the dataframes
        :Param remove_na: Boolean, option of whether or not  to remove the nulls
    Returns: pd.DataFrame
    >>> most_recent_bh_df = most_recent_businesses_df(df1=bh_df, remove_na=True)
    >>> most_recent_bh_df
    """
    most_recent_bh_df = df1.drop_duplicates(subset='BusinessName', keep='last')      
    if remove_na==True:
        most_recent_bh_df = most_recent_bh_df[~ most_recent_bh_df['total_review_cnt_delta'].isna()]

    if on:
        most_recent_bh_df = pd.merge(left=most_recent_bh_df, right=df2, on=on, how='inner')
        most_recent_bh_df
    most_recent_bh_df.loc[:, 'BusinessRating'] = most_recent_bh_df.loc[:, 'BusinessRating'].astype(float)
    most_recent_bh_df = most_recent_bh_df.reset_index(drop=True)
    return most_recent_bh_df

def row_lvl_comparison(row, target_col:str,  target_list: list,  col_name_edit: Union[str, None] = None):
    """
    To be used with the apply function on a Pandas DataFrame
    """
    the_dict = {}
    the_dict[f'{col_name_edit}_match'] =  0 
    the_dict[f'{col_name_edit}_matches'] =  0
    for item in row[target_col]:
        if item in target_list:
            the_dict[f'{col_name_edit}_match'] = 1
            the_dict[f'{col_name_edit}_matches'] += 1
        else:
            continue
    
    return pd.Series(the_dict)