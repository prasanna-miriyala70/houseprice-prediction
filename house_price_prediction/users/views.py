from django.shortcuts import render, HttpResponse
from django.contrib import messages
from sklearn.tree import DecisionTreeClassifier
from .forms import UserRegistrationForm
from .models import UserRegistrationModel


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})



def Viewdata(request):
    import os
    import pandas as pd
    from django.conf import settings
    path =os.path.join(settings.MEDIA_ROOT,'Bengaluru_House_Data.csv')
    #path = os.path.join(settings.MEDIA_ROOT, "World_Happiness_2015_2017_.csv")
    data = pd.read_csv(path, nrows=500)
    print(data)
    data = data.to_html()
    return render (request, "users/Viewdata.html", {"data":data})


def processdata(request):
    import os 
    import pandas as pd 
    from django.conf import settings 
    path = os.path.join(settings.MEDIA_ROOT,'danaiahprocessdataset.csv')
    data = pd.read_csv(path,nrows=100)
    print(data)
    data = data.to_html 
    return render(request,'users/process.html',{'data':data})




def prediction(request):
    if request.method == 'POST':
        # location,sqft,bath,bhk
        location = request.POST.get('location')
        sqft = request.POST.get('sqft')
        bath= request.POST.get('bath')
        bhk = request.POST.get('bhk')
        userinput = [[location,sqft,bath,bhk]]
        print('****************************')
        print(userinput)
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        from django.conf import settings
       
        path = os.path.join(settings.MEDIA_ROOT + "\\" + "Bengaluru_House_Data.csv")
        df = pd.read_csv(path)

        # print(df.head())
        df.drop(['society'], axis=1, inplace=True)
        df2 = df.dropna()
        df2['bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))

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
        print('---------convert_sqft_to_num---------------')

        df4 = df3.copy()
        df4['price_per_sqft'] = (df4['price']*100000) / df4['total_sqft']

        df4.location = df4.location.apply(lambda x: x.strip())
        location_stats = df4['location'].value_counts(ascending=False)
        # location_stats
        location_stats_less_than_10 = location_stats[location_stats<=10]
        # location_stats_less_than_10
        print('-----------location_stats_less_than_10------------')

        df4.location = df4.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
        # len(df4.location.unique())
        df4[df4.total_sqft/df4.bhk<300].head()
        df5 = df4[~(df4.total_sqft/df4.bhk<300)]
        print(df5['location'].value_counts())
        def remove_pps_outliers(df):
            df_out = pd.DataFrame()
            for key, subdf in df.groupby('location'):
                m = np.mean(subdf.price_per_sqft)
                st = np.std(subdf.price_per_sqft)
                reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
                df_out = pd.concat([df_out,reduced_df],ignore_index=True)
            return df_out
        df6 = remove_pps_outliers(df5)
        print('------outlier romved------')
        df9 = df6.drop(['area_type', 'availability', 'balcony', 'size','price_per_sqft'],axis='columns')
        dummies = pd.get_dummies(df9.location)
        df10 = pd.concat([df9,dummies.drop('other',axis='columns')],axis='columns')
        df11 = df10.drop('location',axis='columns')
        X = df11.drop(['price'],axis='columns')
        Y = df11.price
        #split
        print('-----splitting------')
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)            
        from sklearn.model_selection import GridSearchCV

        # We are going to compare 3 regression models
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Lasso
        from sklearn.tree import DecisionTreeRegressor

        lr_clf = LinearRegression()
        lr_clf.fit(X_train,Y_train)
        #lr_clf.score(X_test,y_test)

        # Predict test set
        y_pred = lr_clf.predict(X_test)

        # Check the accuracy of the model
        score = lr_clf.score(X_test, Y_test)
        print('Score ====> ',score)


        def predict_price(location,sqft,bath,bhk):    
            loc_index = np.where(X.columns==location)[0][0]
            x = np.zeros(len(X.columns))
            x[0] = sqft
            x[1] = bath
            x[2] = bhk
            if loc_index >= 0:
                x[loc_index] = 1
            return lr_clf.predict([x])[0]        
        pred_res = predict_price(location,int(sqft), int(bath), int(bhk))
        pred_res1 = round(pred_res,2)
        print('===== > ',pred_res1)
        return render(request, "users/predictions_form.html", {"test_pred":pred_res1})    
    else:
        return render(request, 'users/predictions_form.html', {})
    return render(request, 'users/predictions_form.html')


def ML(request):
     import pandas as pd
     import numpy as np
     import matplotlib.pyplot as plt
     import seaborn as sns
     import os
     from django.conf import settings
    #  df = pd.read_csv('C:\\Users\\MMC\\Downloads\\Bengaluru_House_Data.csv')
     path = os.path.join(settings.MEDIA_ROOT + "\\" + "Bengaluru_House_Data.csv")
     df = pd.read_csv(path)
     df2 = df.dropna()
     df2['bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))

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
     print('---------convert_sqft_to_num---------------')

     df4 = df3.copy()
     df4['price_per_sqft'] = (df4['price']*100000) / df4['total_sqft']

     df4.location = df4.location.apply(lambda x: x.strip())
     location_stats = df4['location'].value_counts(ascending=False)
     # location_stats
     location_stats_less_than_10 = location_stats[location_stats<=10]
     # location_stats_less_than_10
    #  print('-----------location_stats_less_than_10------------')

     df4.location = df4.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
    #  # len(df4.location.unique())
     df4[df4.total_sqft/df4.bhk<300].head()
     df5 = df4[~(df4.total_sqft/df4.bhk<300)]
    #  print(df5['location'].value_counts())
     def remove_pps_outliers(df):
           df_out = pd.DataFrame()
           for key, subdf in df.groupby('location'):
               m = np.mean(subdf.price_per_sqft)
               st = np.std(subdf.price_per_sqft)
               reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
               df_out = pd.concat([df_out,reduced_df],ignore_index=True)
           return df_out
     df6 = remove_pps_outliers(df5)
    #  print('------outlier romved------')
     df9 = df6.drop(['area_type', 'availability', 'balcony', 'size','price_per_sqft','society'],axis='columns')
     dummies = pd.get_dummies(df9.location)
     df10 = pd.concat([df9,dummies.drop('other',axis='columns')],axis='columns')
     df11 = df10.drop('location',axis='columns')
     X = df11.drop(['price'],axis='columns')
     Y = df11.price
     #split
     print('-----splitting------')
     from sklearn.model_selection import train_test_split
     X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)            
     from sklearn.model_selection import GridSearchCV

     # We are going to compare 3 regression models
    
     #algrothem-1
     from sklearn.linear_model import LinearRegression    
     lr_clf = LinearRegression()
     lr_clf.fit(X_train,Y_train)
    #  accuracy = lr_clf.score(X_test,Y_test)*100

     # Predict test set
     y_pred = lr_clf.predict(X_test)
     from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    #  accuracy = accuracy_score(Y_test, y_pred) * 100
     accuracy = lr_clf.score(X_test,Y_test)*100
    #  clf.score(X_test, y_test)
    #  accuracy = accuracy_score(Y_test, Y_test) 

     print('Accuracy:', accuracy)
    #  from sklearn.metrics import precision_score
    #  precision = precision_score(Y_test, y_pred) * 100
    #  print('Precision Score:', precision)
    # #  from sklearn.metrics import recall_score
    #  recall = recall_score(Y_test, y_pred) * 100
    #  print('recall_score:',recall)
    # #  from sklearn.metrics import f1_score
    #  f1score = f1_score(Y_test, y_pred) * 100
    #  print('f1score:',f1score)


#second algrothem
     from sklearn.linear_model import Lasso
     lasso = Lasso()
     lasso.fit(X_train,Y_train)

     y_pred1 = lasso.predict(X_test)
     from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    #  accuracy1 = accuracy_score(Y_test, y_pred1) * 100
     accuracy1 = lasso.score(X_test,Y_test)*100

     print('Accuracy:', accuracy1)
    # #  from sklearn.metrics import precision_score
    #  precision1 = precision_score(Y_test, y_pred1) * 100
    #  print('Precision Score:', precision1)
    # #  from sklearn.metrics import recall_score
    #  recall1 = recall_score(Y_test, y_pred1) * 100
    #  print('recall_score:',recall1)
    # #  from sklearn.metrics import f1_score
    #  f1score1 = f1_score(Y_test, y_pred1) * 100
    #  print('f1score:',f1score1)


#third algrothem
     from sklearn.tree import DecisionTreeRegressor
     DTR = DecisionTreeRegressor()
     DTR.fit(X_train,Y_train)

     y_pred2 = DTR.predict(X_test)
     from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    #  accuracy2 = accuracy_score(Y_test, y_pred2) * 100
     accuracy2 = lasso.score(X_test,Y_test)*100

     print('Accuracy:', accuracy2)
    # #  from sklearn.metrics import precision_score
    #  precision2 = precision_score(Y_test, y_pred2) * 100
    #  print('Precision Score:', precision2)
    # #  from sklearn.metrics import recall_score
    #  recall2 = recall_score(Y_test, y_pred2) * 100
    #  print('recall_score:',recall2)
    # #  from sklearn.metrics import f1_score
    #  f1score2 = f1_score(Y_test, y_pred2) * 100
    #  print('f1score:',f1score2)



#fourth algrothem
     from sklearn.ensemble import  RandomForestRegressor
     RFR  = RandomForestRegressor()
     RFR.fit(X_train,Y_train)
     y_pred3 = RFR.predict(X_test)
     from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    #  accuracy3 = accuracy_score(Y_test, y_pred3) * 100
     accuracy3 = RFR.score(X_test,Y_test)*100

     print('Accuracy:', accuracy3)
    # #  from sklearn.metrics import precision_score
    #  precision3 = precision_score(Y_test, y_pred3) * 100
    #  print('Precision Score:', precision3)
    # #  from sklearn.metrics import recall_score
    #  recall3 = recall_score(Y_test, y_pred3) * 100
    #  print('recall_score:',recall3)
    # #  from sklearn.metrics import f1_score
    #  f1score3 = f1_score(Y_test, y_pred3) * 100
    #  print('f1score:',f1score3)

#fifith algrothem

     from sklearn.ensemble import AdaBoostRegressor
     ABR = AdaBoostRegressor()
     ABR.fit(X_train,Y_train)
     y_pred4 = ABR.predict(X_test)
     from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    #  accuracy4 = accuracy_score(Y_test, y_pred4) * 100
     accuracy4 = ABR.score(X_test,Y_test)*100

     print('Accuracy:', accuracy4)
    # #  from sklearn.metrics import precision_score
    #  precision4 = precision_score(Y_test, y_pred) * 100
    #  print('Precision Score:', precision4)
    # #  from sklearn.metrics import recall_score
    #  recall4 = recall_score(Y_test, y_pred4) * 100
    #  print('recall_score:',recall4)
    # #  from sklearn.metrics import f1_score
    #  f1score4 = f1_score(Y_test, y_pred4) * 100
    #  print('f1score:',f1score4)

#sixth algrothem
     from sklearn.ensemble import GradientBoostingRegressor
     GBR = GradientBoostingRegressor()
     GBR.fit(X_train,Y_train)
     y_pred5 = GBR.predict(X_test)
     from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    #  accuracy5 = accuracy_score(Y_test, y_pred5) * 100
     accuracy5 = GBR.score(X_test,Y_test)*100

     print('Accuracy:', accuracy5)
    # #  from sklearn.metrics import precision_score
    #  precision5 = precision_score(Y_test, y_pred5) * 100
    #  print('Precision Score:', precision5)
    # #  from sklearn.metrics import recall_score
    #  recall5 = recall_score(Y_test, y_pred5) * 100
    #  print('recall_score:',recall5)
    # #  from sklearn.metrics import f1_score
    #  f1score5 = f1_score(Y_test, y_pred5) * 100
    #  print('f1score:',f1score5)

#seven algrothem

     from xgboost import XGBRegressor
     XGB = XGBRegressor()
     XGB.fit(X_train,Y_train)
     y_pred6 = XGB.predict(X_test)
     from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    #  accuracy6 = accuracy_score(Y_test, y_pred6) * 100
     accuracy6 = lasso.score(X_test,Y_test)*100

     print('Accuracy:', accuracy6)
    # #  from sklearn.metrics import precision_score
    #  precision6 = precision_score(Y_test, y_pred6) * 100
    #  print('Precision Score:', precision6)
    # #  from sklearn.metrics import recall_score
    #  recall6 = recall_score(Y_test, y_pred6) * 100
    #  print('recall_score:',recall6)
    # #  from sklearn.metrics import f1_score
    #  f1score6 = f1_score(Y_test, y_pred6) * 100
    #  print('f1score:',f1score6)
    
     accuracy = {'LR':accuracy,  'LASSO': accuracy1,  'DTR': accuracy2,   'RFR':accuracy3,   'ABR':accuracy4,    'GBR':accuracy5,   'XBR':accuracy6}
     


    #  accuracy = {'LR':accuracy,  'LASSO': accuracy1,  'DTR': accuracy2,   'RFR':accuracy3,   'ABR':accuracy4,    'GBR':accuracy5,   'XBR':accuracy6}
    #  precision ={'LR':precision, 'LASSO': precision1, 'DTR': precision2,   'RFR':precision3, 'ABR':precision4,   'GBR':precision5,   'XBR':precision6}
    #  recall = {'LR': recall,     'LASSO': recall1,     'DTR': recall2,   'RFR':recall3,      'ABR':recall4,     'GBR':recall5,       'XBR':recall6}
    #  f1score = {'LR':f1score,    'LASSO':f1score1,    'DTR':f1score2,   'RFR':f1score3,      'ABR':f1score4,     'GBR':f1score5,      'XBR':f1score6}
    # # confusionmatrix = {'DT':confusionmatrix, 'KNN':confusionmatrix2, 'RF':confusionmatrix3}
    #'confusionmatrix:',confusionmatrix
    # roc = {'RF': roc1, 'SVM': roc2, 'LogisticRegression': roc,  'MLP': roc5}
     return render(request, 'users/ML.html',
                  {"accuracy": accuracy})
                   
                #   {"accuracy": accuracy, "precision": precision, "recall":recall, 'f1score':f1score})






