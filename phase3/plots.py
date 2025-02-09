#Importing libraries
import plotly.express as px
#Class to plot different plots for analysis
class EDA():
    
    def __init__(self) -> None:
        #Initialising colors for labels, red for default , non-default 
        self.color_map = {'Default': '#dc3545', 'Non-Default': '#28a745'}
    
    def risk_prediction_plot(self,df):
        #Pie chart
        fig = px.pie(df,names = 'Predicted Risk', color='Predicted Risk', color_discrete_map= self.color_map, labels= 'Predicted Risk')
        #Displaying data labels
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title_text='Loan Risk Prediction')
        return fig
    
    def home_ownership_plot(self,df):
        #Histogram distribution for person_home_ownership
        fig = px.histogram(df,x = 'person_home_ownership', color='Predicted Risk', color_discrete_map= self.color_map, labels= 'Predicted Risk',text_auto=True, barmode = 'group')
        fig.update_layout(title_text='Home Ownership Distribution')
        fig.update_xaxes(categoryorder='total descending')
        return fig
    
    def loan_intent_plot(self,df):
        #Histogram distribution for loan_intent
        fig = px.histogram(df,x = 'loan_intent', color='Predicted Risk', color_discrete_map= self.color_map, labels= 'Predicted Risk',text_auto=True, barmode = 'group')
        fig.update_layout(title_text='Loan Intent Distribution')
        fig.update_xaxes(categoryorder='total descending')
        return fig
    
    
    def age_group_plot(self,df):
        #Histogram distribution for Person_Age_Group
        fig = px.histogram(df,x = 'Person_Age_Group', color='Predicted Risk', color_discrete_map= self.color_map, labels= 'Predicted Risk',text_auto=True, barmode = 'group')
        fig.update_layout(title_text='Age Group Distribution')
        fig.update_xaxes(categoryorder='total descending')
        return fig
    
    def loan_grade_plot(self,df):
        #Histogram distribution for loan_grade
        fig = px.histogram(df,x = 'loan_grade', color='Predicted Risk', color_discrete_map= self.color_map, labels= 'Predicted Risk',text_auto=True, barmode = 'group')
        fig.update_layout(title_text='Loan Grade Distribution')
        fig.update_xaxes(categoryorder='total descending')
        return fig
        