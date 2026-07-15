import pandas as pd
def Missing_Summary(df : pd.DataFrame) -> pd.DataFrame :
    missing = []
    missing_value = [] 
    missing_percentage = []
    for cols in  df.columns.tolist():
        if df[cols].isnull().sum() != 0 :
            value = int((df[cols].isnull().sum()))
            total = int(df.shape[0])
            percentage = (value /total)*100
            missing.append(cols)
            missing_value.append(value)
            missing_percentage.append(percentage)
    missing_columns = pd.DataFrame(list(zip(missing_value,missing_percentage)) , index = missing  , columns = ["Missing_Count","Missing_Percentage"]).sort_values("Missing_Percentage",ascending=False)
    return missing_columns