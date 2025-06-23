def print_dataframe_info(df, df_name):
    print(f"\n{df_name} DataFrame info:")
    print(df.info())
    print(f"\nFirst few rows of the {df_name} DataFrame:")
    print(df.head())