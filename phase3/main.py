#Importing libraries
import streamlit as st
import pandas as pd
from model import Classifier
from plots import EDA

# A function which sets the streamlit layout mode to 'wide' when run
def wide_space_default():
    st.set_page_config(layout='wide')

wide_space_default()


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index= False).encode("utf-8")


st.title("LOAN CREDIT RISK PREDICTION")

# Uploading the data(.csv) file which contains the data which is going to be predicted
file = st.file_uploader('Upload your data:')
if file is not None:
    try:

        df = pd.read_csv(file)
    
    except:
        st.error("Please upload a valid .csv file with full data only")
    
    
    #Creating two tabs: One for recommendation, one for prediction results
    tab1, tab2 = st.tabs(['Analysis','Risk Predictor'])
    #Creating objects for classifier and plots
    eda = EDA()
    cl = Classifier()
    #Run model with the csv data uploaded
    df_full = cl.generate_prediction_table(df)
    df_predicted = cl.projection_df(df_full)
    
    
    #Under Analysis Tab
    with tab1:
        
        #Plotting various plots to get recommendations from analysis
        fig1= eda.age_group_plot(df)
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = eda.home_ownership_plot(df)
        st.plotly_chart(fig2, use_container_width=True)
        
        fig3 = eda.loan_intent_plot(df)
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = eda.loan_grade_plot(df)
        st.plotly_chart(fig4, use_container_width=True)
        
    #Under Risk prediction tabs  
    with tab2:
        csv = convert_df(df_full)
        
        
        df_predicted.index += 1 #Displaying data showing first row as 1 instead of 0.
        
        fig = eda.risk_prediction_plot(df) #Plotting risk prediction distribution (default or non-default)
        st.plotly_chart(fig, use_container_width=True) #Display above plot in UI
        
        
        
        #Initialising categorical columns 
        categorical_columns = ["All","person_home_ownership",'loan_intent', 'loan_grade','Predicted Risk']
        #Introducing dropdown box to filter table for categorical columns
        selected_column = st.selectbox("Filter By: ", categorical_columns)

        #Filtering table results based on column and category type selected in that column
        if selected_column == 'All': 
            st.markdown(df_predicted.astype('object').to_html(), unsafe_allow_html=True)
        else:
            column_categories = ['All'] + list(df_predicted[selected_column].unique())
            selected_group = st.selectbox("Select Category: ", column_categories)
            if selected_group != 'All':
                
                df_filtered = df_predicted.groupby(selected_column).get_group(selected_group)
                st.markdown(df_filtered.astype('object').to_html(), unsafe_allow_html=True)
            else:
                st.markdown(df_predicted.astype('object').to_html(), unsafe_allow_html=True)
        
        
        #Creating columns for placement of download button
        st.text("")
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        with col4:
            #Download button which converts table data to csv file
            st.download_button(
                            label=":red[Download CSV]",
                            data=csv,
                            file_name="output.csv",
                            mime="text/csv"
                        )    
                

                
        
    
    
    

    

    
    
    
      