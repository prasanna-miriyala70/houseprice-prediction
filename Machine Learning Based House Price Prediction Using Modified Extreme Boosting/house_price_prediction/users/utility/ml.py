def regression(userinput):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from django.conf import settings
    import os
    path = os.path.join(settings.MEDIA_ROOT + "\\" + "Bengaluru_House_Data.csv")
    df = pd.read_csv(path)
    df.drop(['society'], axis=1, inplace=True)
    df2 = df.dropna()
    df2['bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))
    print('---------------')
    print(df2)
    # We create a function for finding inconsistancies in total_sqft
    def sqft_value(x):
        try:
            float(x)
        except:
            return False
        return True
    df2[~df2.total_sqft.apply(sqft_value)].head(10)
    def convert_sqft_to_num(x):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0])+float(tokens[1]))/2
        try:
            return float(x)
        except:
            return None
    df3 = df2.copy()
    df3.total_sqft = df3.total_sqft.apply(convert_sqft_to_num)
    # df3.head(10)
    df4 = df3.copy()
    df4['price_per_sqft'] = (df4['price']*100000) / df4['total_sqft']
    # df4.head(3)
    df4.location = df4.location.apply(lambda x: x.strip())
    location_stats = df4['location'].value_counts(ascending=False)
    # location_stats
    location_stats_less_than_10 = location_stats[location_stats<=10]
    # location_stats_less_than_10
    df4.location = df4.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
    # len(df4.location.unique())
    df4[df4.total_sqft/df4.bhk<300].head()
    df5 = df4[~(df4.total_sqft/df4.bhk<300)]
    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
            df_out = pd.concat([df_out,reduced_df],ignore_index=True)
            return df_out
    df6 = remove_pps_outliers(df5)
    df9 = df6.drop(['area_type', 'availability', 'balcony', 'size','price_per_sqft'],axis='columns')
    # df9.head(3)
    dummies = pd.get_dummies(df9.location)
    df10 = pd.concat([df9,dummies.drop('other',axis='columns')],axis='columns')
    # df10.head()
    df11 = df10.drop('location',axis='columns')
    # df11.head(2)
    X = df11.drop(['price'],axis='columns')
    # X.head(3)
    Y = df11.price
    # Y.head(3)
    # Train test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
    from sklearn.model_selection import GridSearchCV

    # We are going to compare 3 regression models
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Lasso
    from sklearn.tree import DecisionTreeRegressor


    # Create object for the classifier
    lr_clf = LinearRegression()
    lr_clf.fit(X_train,Y_train)
    #lr_clf.score(X_test,y_test)
    model = LinearRegression()
    model.fit(X_train,Y_train)
    y_pred = model.predict(userinput)
    print(y_pred)

    return y_pred


# The above function will return rows with a single numeric value in total_sqft