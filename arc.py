# In[ ]:

# streamlit run /Users/Saily/Technocolabs internship/Deployment files/arc.py
# streamlit run /Users/Saily/Desktop/Technocolabs internship/arc.py
# streamlit run /Users/Saily/Downloads/arc.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
from sklearn.linear_model import SGDRegressor
from PIL import Image
# price_rf.pkl


data_o = pd.read_csv("origin_city.csv") 
gg = sorted(list(set(list(data_o.origin_city))))

data_d = pd.read_csv("dest_city.csv")  
hh = sorted(list(set(list(data_d.dest_city))))


from optimal_time_deploy_func import data_cleaner
from optimal_time_deploy_func import time_pred

from price_deploy_func import price_cleaner
from price_deploy_func import price_pred

# 

X_train = pd.read_csv("X_train.csv") 
X_train.drop(['Unnamed: 0'],axis=1,inplace=True)
price_X_train = pd.read_csv("price_X_train.csv") 
price_X_train.drop(['Unnamed: 0'],axis=1,inplace=True)

pickle_time = open("rf_random.sav","rb")
model_opt_time = pickle.load(pickle_time)

pickle_price = open("price_rf.pkl","rb")
model_price = pickle.load(pickle_price)

# @st.cache(suppress_st_warning=True)  # 👈 Changed this
def main():

    st.sidebar.title("Let's Travel")
    st.sidebar.markdown("Enter the flight details:")
    o_city = st.sidebar.selectbox('Origin City', gg)
    d_city = st.sidebar.selectbox('Destination City',hh)
    
    date = st.sidebar.date_input('Inbound Date',)
    f_type = st.sidebar.select_slider('Slide to select the flight type', options=["cheap","best","fast"])
    stops= st.sidebar.radio('No. of Stops', ["direct","1 stop","2 stops"])
    airline = st.sidebar.selectbox('Preferred Airline',["IndiGo","GoAir","Air India","SpiceJet","AirAsia India","Vistara","Multiple Airlines"])
    #st.sidebar.selectbox("Flight Type",["Cheap","Best","Fast"])
    
    
    #### Input from users to pass in the test set
    Origin = list(data_o.loc[data_o['origin_city'] == o_city, "origin_code"])[0]
    Destination =  list(data_d.loc[data_d['dest_city'] == d_city, "dest_code"])[0]
    Date = datetime.datetime.strptime(str(date), '%Y-%m-%d').strftime('%d/%m/%Y') 
    Airline = airline
    sort = f_type
    Stops = stops
    
        ## PREDICTING OPTIMAL TIME
    #2
    test_df = data_cleaner(Date,Origin,Destination,Airline,sort,Stops)
    
    #3
    test_df_pca = time_pred(X_train,test_df)
    
    ##
    
    test_df_pred = model_opt_time.predict(test_df_pca)
    Optimal_time = test_df_pred
    
#         ## PREDICTING PRICE FOR THE OPTIMAL TIME
#     #2
    price_test_df = price_cleaner(Date,Origin,Destination,Airline,sort,Stops)
    
    #3
    price_test_df_pca = price_pred(price_X_train,price_test_df)
    ###
    
    price_test_df_pred = model_price.predict(price_test_df_pca)
    
    
    x, y, z = st.beta_columns([1,3,2])
    with x:
#         image = Image.open('/Users/Saily/Downloads/gg.png')
#         st.image(image, caption='Travel Safe')
        st.image('gg.png')
    with y:
        st.title("Your Travel Partner")
    
    opt_time = ""
    app_price = ""
    if st.sidebar.button('Predict'):
#         st.write(Date)
#         st.write(Origin)
#         st.write(Destination)
#         st.write(Airline)
#         st.write(sort)
#         st.write(Stops)
#         st.write(test_df_pca.shape)
#        
        opt_time = int(Optimal_time)
#         opt_time = 2500
#         app_price = 15432
        app_price = int(price_test_df_pred)
    
        
        st.success('The Optimal Time for the flight is {} hrs i.e {} days'.format(round(opt_time/60),round(opt_time/(24*60))))
       
       # st.success('The Optimal Time for the flight is {} days'.format(round(opt_time/(24*60))))
        st.success("The price for the chosen flight is approximately Rs {}".format(app_price))
        
        
        b,c = st.beta_columns([1,2])
        with c:
#             image1 = Image.open('/Users/Saily/Downloads/img2.png')
#             st.image(image1,width = 300)
             st.image('img2.png', width = 300)
    else:
        with y:
#             image2 = Image.open('/Users/Saily/Downloads/img1.jpg')
#             st.image(image2,width = 400)
             st.image('img1.jpg', width =400)
        
if __name__ == '__main__':
    main()




