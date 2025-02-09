#Importing libraries
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Saved model file name from phase 2
model_pkl_file = '../phase2/credit_risk_classifier.pkl'

class Classifier():
    
    def __init__(self) -> None:
            #Initializing best model and columns for pre-processing   
            with open(model_pkl_file, 'rb') as file:  
                self.model = pickle.load(file)
            with open('../phase2/col_names.pickle', 'rb') as f:
                self.col_names = pickle.load(f)

    def pre_process_data(self,df) -> np.array:
        
        # Pre-processing data same as done in phase 1
        df[['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']] = df[['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']].astype('category') #converting the datatype of the columns which are incorrect in the dataframe to int
        df[['person_emp_length']] = df[['person_emp_length']].astype('int64') 
            
        bins = [20,26,31,36,41,46,51,56,60] #Creating bins for categories
        labels = ['Age:21-25','Age:26-30','Age:31-35','Age:36-40','Age:41-45','Age:46-50','Age:51-55','Age:56-60'] #Age groups
        df['Person_Age_Group'] = pd.cut(df['person_age'], bins = bins,labels = labels) #Mapping bins to categories using cut

        df.drop(['person_age'], inplace = True, axis = 1)
        df.drop('cb_person_cred_hist_length', axis = 1, inplace =  True)
        
        categorical_cols = df.select_dtypes(include=['category']).columns
        df1 = pd.get_dummies(df, columns=categorical_cols, drop_first=False,dtype=int)     
        
        df_new = pd.DataFrame(columns=self.col_names)
        common_cols = df1.columns.intersection(df_new.columns)
        for col in common_cols:
            df_new[col] = df1[col]
        
        df_new.drop('loan_status',axis = 1, inplace = True)

        df_new.fillna(0, inplace=True)
        
        X = df_new.to_numpy()
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        
        return scaled_X #Returns pre-processed X

    def predict(self, X): 
        
        return self.model.predict(X)
    
    def generate_prediction_table(self,df):
        #Pre-process data
        X = self.pre_process_data(df)
        #Predict data
        pred = self.predict(X)
        #Map 0 and 1 from pred to meaningful labels
        risk_labels = {0: 'Non-Default', 1: 'Default'}
        risk_pred = []
        for i in range(len(pred)):
            risk_pred.append(risk_labels.get(pred[i]))
        
        df['Predicted Risk'] = risk_pred
        
        return df
    
    
    def projection_df(self,df):
        #Displaying only projection of few important columns including prediction output 
        df = df[['person_income','loan_amnt','loan_int_rate','loan_percent_income','loan_intent','person_home_ownership','loan_grade','Predicted Risk']]
        return df

    