#!/usr/bin/env python
# coding: utf-8

# ## Business Problem Overview
# In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.
# 
# For many incumbent operators, retaining high profitable customers is the number one business goal.
# 
# 
# To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.
# 
#  
# In this project, you will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.
# 
#  
# 
# ## Understanding and Defining Churn
# There are two main models of payment in the telecom industry - postpaid (customers pay a monthly/annual bill after using the services) and prepaid (customers pay/recharge with a certain amount in advance and then use the services).
# 
#  
# 
# In the postpaid model, when customers want to switch to another operator, they usually inform the existing operator to terminate the services, and you directly know that this is an instance of churn.
# 
#  
# 
# However, in the prepaid model, customers who want to switch to another network can simply stop using the services without any notice, and it is hard to know whether someone has actually churned or is simply not using the services temporarily (e.g. someone may be on a trip abroad for a month or two and then intend to resume using the services again).
# 
#  
# 
# Thus, churn prediction is usually more critical (and non-trivial) for prepaid customers, and the term ‘churn’ should be defined carefully.  Also, prepaid is the most common model in India and southeast Asia, while postpaid is more common in Europe in North America.
# 
#  
# 
# This project is based on the Indian and Southeast Asian market.
# 
#  
# 
# ## Definitions of Churn
# There are various ways to define churn, such as:
# 
# Revenue-based churn: Customers who have not utilised any revenue-generating facilities such as mobile internet, outgoing calls, SMS etc. over a given period of time. One could also use aggregate metrics such as ‘customers who have generated less than INR 4 per month in total/average/median revenue’.
# 
#  
# 
# The main shortcoming of this definition is that there are customers who only receive calls/SMSes from their wage-earning counterparts, i.e. they don’t generate revenue but use the services. For example, many users in rural areas only receive calls from their wage-earning siblings in urban areas.
# 
#  
# 
# ### Usage-based churn: 
# Customers who have not done any usage, either incoming or outgoing - in terms of calls, internet etc. over a period of time.
# 
#  
# 
# A potential shortcoming of this definition is that when the customer has stopped using the services for a while, it may be too late to take any corrective actions to retain them. For e.g., if you define churn based on a ‘two-months zero usage’ period, predicting churn could be useless since by that time the customer would have already switched to another operator.
# 
#  
# 
# In this project, you will use the usage-based definition to define churn.
# 
#  
# 
# ### High-value Churn
# In the Indian and the southeast Asian market, approximately 80% of revenue comes from the top 20% customers (called high-value customers). Thus, if we can reduce churn of the high-value customers, we will be able to reduce significant revenue leakage.
# 
#  
# 
# In this project, you will define high-value customers based on a certain metric (mentioned later below) and predict churn only on high-value customers.
# 
#  
# 
# ### Understanding the Business Objective and the Data
# The dataset contains customer-level information for a span of four consecutive months - June, July, August and September. The months are encoded as 6, 7, 8 and 9, respectively. 
# 
# 
# The business objective is to predict the churn in the last (i.e. the ninth) month using the data (features) from the first three months. To do this task well, understanding the typical customer behaviour during churn will be helpful.
# 
#  
# 
# ### Understanding Customer Behaviour During Churn
# Customers usually do not decide to switch to another competitor instantly, but rather over a period of time (this is especially applicable to high-value customers). In churn prediction, we assume that there are three phases of customer lifecycle :
# 
# The __‘good’ phase:__ In this phase, the customer is happy with the service and behaves as usual.
# 
# The __‘action’ phase:__ The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a  competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. Also, it is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)
# 
# The ‘churn’ phase: In this phase, the customer is said to have churned. You define churn based on this phase. Also, it is important to note that at the time of prediction (i.e. the action months), this data is not available to you for prediction. Thus, after tagging churn as 1/0 based on this phase, you discard all data corresponding to this phase.
# 
#  
# 
# In this case, since you are working over a four-month window, the first two months are the ‘good’ phase, the third month is the ‘action’ phase, while the fourth month is the ‘churn’ phase.
# 
# 
# ### The attributes containing 6, 7, 8, 9 as suffixes imply that those correspond to the months 6, 7, 8, 9 respectively.
# ---------------
# ### Data Preparation
# The following data preparation steps are crucial for this problem:
# 
#  
# 
# ### 1. Derive new features
# 
# This is one of the most important parts of data preparation since good features are often the differentiators between good and bad models. Use your business understanding to derive features you think could be important indicators of churn.
# 
#  
# 
# ### 2. Filter high-value customers
# 
# As mentioned above, you need to predict churn only for the high-value customers. Define high-value customers as follows: Those who have recharged with an amount more than or equal to X, where X is the 70th percentile of the average recharge amount in the first two months (the good phase).
# 
#  
# 
# After filtering the high-value customers, you should get about 29.9k rows.
# 
#  
# 
# ### 3. Tag churners and remove attributes of the churn phase
# 
# Now tag the churned customers (churn=1, else 0) based on the fourth month as follows: Those who have not made any calls (either incoming or outgoing) AND have not used mobile internet even once in the churn phase. The attributes you need to use to tag churners are:
# 
# total_ic_mou_9
# 
# total_og_mou_9
# 
# vol_2g_mb_9
# 
# vol_3g_mb_9
# 
# 
# After tagging churners, remove all the attributes corresponding to the churn phase (all attributes having ‘ _9’, etc. in their names).
# 
#  
# 
# ### Modelling
# Build models to predict churn. The predictive model that you’re going to build will serve two purposes:
# 
# It will be used to predict whether a high-value customer will churn or not, in near future (i.e. churn phase). By knowing this, the company can take action steps such as providing special plans, discounts on recharge etc.
# 
# It will be used to identify important variables that are strong predictors of churn. These variables may also indicate why customers choose to switch to other networks.
# 
#  
# 
# In some cases, both of the above-stated goals can be achieved by a single machine learning model. But here, you have a large number of attributes, and thus you should try using a dimensionality reduction technique such as PCA and then build a predictive model. After PCA, you can use any classification model.
# 
#  
# 
# Also, since the rate of churn is typically low (about 5-10%, this is called class-imbalance) - try using techniques to handle class imbalance. 
# 
#  
# 
# You can take the following suggestive steps to build the model:
# 
# ### Preprocess data (convert columns to appropriate formats, handle missing values, etc.)
# 
# Conduct appropriate exploratory analysis to extract useful insights (whether directly useful for business or for eventual modelling/feature engineering).
# 
# ### Derive new features.
# 
# Reduce the number of variables using PCA.
# 
# Train a variety of models, tune model hyperparameters, etc. (handle class imbalance using appropriate techniques).
# 
# Evaluate the models using appropriate evaluation metrics. Note that is is more important to identify churners than the non-churners accurately - choose an appropriate evaluation metric which reflects this business goal.
# 
# Finally, choose a model based on some evaluation metric.
# 
#  
# 
# The above model will only be able to achieve one of the two goals - to predict customers who will churn. You can’t use the above model to identify the important features for churn. That’s because PCA usually creates components which are not easy to interpret.
# 
#  
# 
# Therefore, build another model with the main objective of identifying important predictor attributes which help the business understand indicators of churn. A good choice to identify important variables is a logistic regression model or a model from the tree family. In case of logistic regression, make sure to handle multi-collinearity.
# 
#  
# 
# After identifying important predictors, display them visually - you can use plots, summary tables etc. - whatever you think best conveys the importance of features.
# 
#  
# 
# Finally, recommend strategies to manage customer churn based on your observations.

# In[4]:


#Load Required Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import regex

import warnings
warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",300)
pd.set_option("display.max_columns",300)


# In[5]:


# Get the dataset
telecom = pd.read_csv(r"C:\Users\Ravi Verma\Documents\AIML\domain assignment\telecom\telecom_churn_data.csv")


# In[6]:


#check the data
telecom.head(10)


# In[11]:


#Check the metadata with verbose( Show all the column names)
telecom.info(verbose=1)


# - __Dataset contains 99999 no of rows.__
# - __226 no of columns.__
# - __Number of Float data type - 179__
# - __Number of int datatype - 35__
# - __Number of object datatype- 12__
# 

# In[14]:


# Segregate Categorcial, ID and Numeric columns for ease of analysis

#Categorcial columns separation : categorical columns are only date here
date_columns = [col for col in telecom.columns if telecom[col].dtype =="object"]    
print(f"Total Categorical columns:{len(date_columns)}")

#ID columns separation
id_columns = ["mobile_number","circle_id"]  # total ID columns are 2 
print(f"Total numeric columns:{len(id_columns)}")

#Numeric columns separation
numeric_columns = [ col for col in telecom.columns if col not in date_columns + id_columns]    
print(f"Total numeric columns:{len(numeric_columns)}")  


# In[15]:


#check the date columns
telecom[date_columns].head()


# - last_date_of_month_6
# - last_date_of_month_7 ,
# - last_date_of_month_8,
# - last_date_of_month_9
# 
# The above columns have only one value - That is the last day of correspnding month
# - Hence we can drop these columns

# - date_of_last_rech_6,
# - date_of_last_rech_7,
# - date_of_last_rech_8,
# - date_of_last_rech_9
# 
# Date of Last recharge of corresponding month also can be dropped, 
# We are not going to derive anything from these dates
# 

# - date_of_last_rech_data_6,
# - date_of_last_rech_data_7
# - date_of_last_rech_data_8
# - date_of_last_rech_data_9
# 
# These columns also can be dropped.

# ## Missing value Treatment and Initial data analysis

# In[16]:


#check the Null values column wise
(telecom.isnull().sum()/len(telecom)).sort_values(ascending = False)


# - Many false null value columns are available. if customer did not recharge, the value assigned as NaN
# - Hence we __can not__ drop these values blindly.
# - We can impute these columns as __zero__.
# 

# - When customer did not recharge,the total_rech_data_*  and date_of_last_rech_data_* are null
#     - Total Recharge data in month 6,7,8,9 would be null
#     - Maximum Recharge Data in Month 6,7,8 ,9 would be null
#     - Average Amount recharge Data in Month 6,7,8,9, would be null
#     
# - Hence this NULL __can not__ be dropped out.
# - We will impute with Zero.
# 

# In[17]:


# when total_rech_data and date_of_last_rech_data is null, Check date_of_last_rech_data,total_rech_data,max_rech_data etc. 
telecom[telecom["total_rech_data_6"].isna() & telecom["date_of_last_rech_data_6"].isna()][ \
    ["date_of_last_rech_data_6","total_rech_data_6","max_rech_data_6","max_rech_data_6","av_rech_amt_data_6"]]


# In[18]:


# Columns which we have to impute as Zero.
zero_impute = ['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8', 'total_rech_data_9',
        'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8', 'av_rech_amt_data_9',
        'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 'max_rech_data_9'
       ]
# Put zero in these columns
telecom[zero_impute] = telecom[zero_impute].apply(lambda x: x.fillna(0))


# Below columns are imputed with zeros.<br>
# - Total Recharge data in month 6,7,8,9 
# - Maximum Recharge Data in Month 6,7,8 ,9
# - Average Amount recharge Data in Month 6,7,8,9,
# 

# In[19]:


# We will drop date columns and ID columns as these will not contribute further to our analysis.

telecom.drop(columns=id_columns,inplace=True)
telecom.drop(columns=date_columns,inplace=True)


# In[20]:


# Check the columns associated with month 6 . From this, we can get an overview of columns/features in 7,8,9 months
month_6_cols = [col for col in telecom.columns if "_6" in col]
print(len(month_6_cols))
month_6_cols


# In[21]:


# check how the data looks for month 6
telecom[month_6_cols].head(10)


# In[22]:


# Check again the null values percentages
(telecom.isnull().sum()/len(telecom)).sort_values(ascending = False).head(50)


# - __Night pack user columns and FB User columns are categorical column.__
# - "night_pck_user_6"    
# - "night_pck_user_7"    
# - "night_pck_user_8"    
# - "night_pck_user_9"    
# - "fb_user_6"           
# - "fb_user_7"           
# - "fb_user_8"           
# - "fb_user_9"   

# In[23]:


# Check night_pck_user unique values in month 6
telecom["night_pck_user_6"].unique()


# In[24]:


#Check the percetages null values of these columns
categorical_columns = ["night_pck_user_6","night_pck_user_7","night_pck_user_8","night_pck_user_9","fb_user_6",          
"fb_user_7",          
"fb_user_8",           
"fb_user_9"]

telecom[categorical_columns].isna().sum()/len(telecom)


# -  __In the above columns,We can impute the NaN as  -1, as a part to mark as missing value.__

# In[25]:


#Fill NaN value as -1 to mark missing value
telecom[categorical_columns] = telecom[categorical_columns].fillna(-1)


# In[26]:


# Check if the null value is filled with -1
telecom[categorical_columns].isna().sum()


# - __Hence there are no null values in night_pck_user and fb_user columns in month 6,7,8,9.__

# In[27]:


#Check the null value pecentage
(telecom.isna().sum()/len(telecom)).sort_values(ascending=False)


# In[28]:


# Many columns have more than 70% null values
#Function to drop columns where there are more than 50% null values
def columns_tobe_dropped(cols):
    '''cols: list of columns in dataframe
      '''
    for col in cols:
        if (telecom[col].isna().sum()/len(telecom)) > .50:   # Check the condition if null values GT .50
            telecom.drop(columns=[col],inplace=True)



# In[29]:


# drop colums 
columns_tobe_dropped(telecom.columns)


# In[30]:


telecom.info()


# - We have removed 30 columns from the dataframe 

# In[31]:


# check the null value row wise.
telecom.isna().sum(axis=1).sort_values(ascending = False).head(30)


# - We have many rows having multiple null values. We are not dropping these and will fill these gradually

# In[32]:


# check the null value again
(telecom.isna().sum()/len(telecom)).sort_values(ascending = False)


# In[33]:


#check columns which have only 1 value.

# Create a DataFrame of no of unique  values and filter where only one value is available.
zero_variance_columns = pd.DataFrame(telecom.nunique()).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
print(zero_variance_columns[zero_variance_columns['nunique'] == 1])


# - The above columns have just one Unique value.
# - Hence they have zero variance and can be dropped.
# 

# In[34]:


# create columns list whihc have zero variance i:e 1 unique value.
columns_tobe_dropped = list(zero_variance_columns[zero_variance_columns['nunique'] == 1]["feature"])
columns_tobe_dropped


# In[35]:


# drop columns whish are having 1 unique values
telecom.drop(columns=columns_tobe_dropped,inplace=True)


# In[36]:


#check the shape
telecom.shape


# In[37]:


# Check the null values again
(telecom.isna().sum()/len(telecom)).sort_values(ascending=False).reset_index()


# - Still we have null values in 107 columns.
# - Majority of the null values are in Minitue of Usage columns 
# - As these values are not available, so we are imputing those values as 0 instead of iteratively imputing.
# 

# In[38]:


# Fill hr NaN as zero.
telecom = telecom.fillna(0)


# In[39]:


telecom.isna().sum()


# In[40]:


telecom.info()


# - __Now thre is no null values in the data.__
# - __still we have 99999 rows of data.__
# - __No of columns reduced to 185 from 226.__

# ## Filter High Value Customer
# - We need to predict churn only for the high-value customers. 
# - Those who have recharged with an amount more than or equal to X, where X is the 70th percentile of the average recharge
#   amount in the first two months (the good phase).
# 
# 

# ### Create derive columns to filter high value customer

# In[41]:


#Calculate total Data recharge amount--> Total Data Recharge * Average Amount of Data recharge
telecom["total_data_recharge_amnt_6"] = telecom.total_rech_data_6 * telecom.av_rech_amt_data_6
telecom["total_data_recharge_amnt_7"] = telecom.total_rech_data_7 * telecom.av_rech_amt_data_7


# In[42]:


#Calculate Total Amount recharge --> total talktime recharge + total data recharge
telecom["total_recharge_amnt_6"] = telecom.total_rech_amt_6 + telecom.total_data_recharge_amnt_6
telecom["total_recharge_amnt_7"] = telecom.total_rech_amt_7 + telecom.total_data_recharge_amnt_7


# In[43]:


#Calculate Average amount of recharge of 6th and 7th month
telecom['average_amnt_6_7'] = (telecom["total_recharge_amnt_6"] + telecom["total_recharge_amnt_7"])/2


# In[44]:


# Check the 70th percentile of "average_amnt_6_7"
telecom['average_amnt_6_7'].quantile(.70)


# - __70th percentile of average amount recharge in 6th and 7th month comes as 478.0.__
# - Now we need to filter the data based on this value.
# 

# In[45]:


#filter based on 70th percentile .
telecom_highvalue = telecom[telecom["average_amnt_6_7"]>= telecom["average_amnt_6_7"].quantile(.70)]


# In[46]:


#Delete the derived columns created in above step
telecom_highvalue.drop(columns=["total_data_recharge_amnt_6","total_data_recharge_amnt_7","total_recharge_amnt_6",\
                                "total_recharge_amnt_7","average_amnt_6_7"],inplace=True)


# In[47]:


telecom_highvalue.shape


# - __Finally we have 30001 rows of high value customer data with 185 columns.__
# 

# In[48]:


# check the data
telecom_highvalue.head()


# ## Tag churners and remove attributes of the churn phase
# 
# - Now we need to tag the churned customers (churn=1, else 0) based on the fourth month as follows: 
# 
# - __Those who have not made any calls (either incoming or outgoing) and  have not used mobile internet even once in the churn
#   phase.__
#   
# - Based on these below attributes we need to decide churners
# 
#     - total_ic_mou_9
# 
#     - total_og_mou_9
# 
#     - vol_2g_mb_9
# 
#     - vol_3g_mb_9

# In[49]:


#Calculate total call in mins by adding Incoming and Outgoing calls
telecom_highvalue['total_calls_9'] = telecom_highvalue.total_ic_mou_9 + telecom_highvalue.total_og_mou_9


# In[50]:


# Calculate total 2G and 3G consumption of data
telecom_highvalue["total_data_consumptions"] = telecom_highvalue.vol_2g_mb_9 + telecom_highvalue.vol_3g_mb_9


# - Now we need to create Churn variable.
# - __Customer who have not used any calls or have not consumed any data on month of 9 are tagged as Churn customer.__
# - Churn customer is marked as 1
# - non-churn custoner is marked as 0

# In[51]:


#Tag 1 as churner  where total_calls_9=0 and total_data_consumptions=0
# else 0 as non-churner
telecom_highvalue["churn"]=telecom_highvalue.apply(lambda row:1 if (row.total_calls_9==0 and row.total_data_consumptions==0) else 0,axis=1)


# In[52]:


#check the percentages of churn and non churn data
telecom_highvalue["churn"].value_counts(normalize=True)


# - __The data is imbalance.__ 
# - __Churn percentage is close 8 and non-churn percentage is close to 92.__

# In[53]:


#Drop the derived columns
telecom_highvalue.drop(columns=["total_calls_9","total_data_consumptions"],inplace=True)


# ## Delete columns belong to the 9th month :Churn Month
# - After tagging churners, remove all the attributes corresponding to the churn phase 
# (all attributes having ‘ _9’, etc. in their names.
# 
# - These columns  contain data for users, where these users are already churned.
# - Hence those will not contribute anything to churn prediction.

# In[54]:


# drop all 9th month columns
telecom_highvalue = telecom_highvalue.filter(regex='[^9]$',axis=1)


# In[55]:


# check the baisc info about high value customer
telecom_highvalue.info(verbose=1)


# - __Finally we have 30,001 rows of records and 141 columns are available to explore.__

# ## Exploratory Data Analysis

# In[56]:


# Check the percenatges of churn and non-churn customers
telecom_highvalue["churn"].value_counts(normalize=True)


# In[57]:


# plot to Check percetanges of churn and non churn data
plt.figure(figsize=(8,6))
telecom_highvalue["churn"].value_counts(normalize=True).plot.bar()
plt.tick_params(size=5,labelsize = 15) 
plt.title("Churn and Non-Churn distributions in percentage",fontsize=15)
plt.ylabel("Percentages",fontsize=15)
plt.xlabel("0-NonChurn        1- Churn",fontsize=15)
plt.grid(0.3)
plt.show()


#  - __We have 92% customers belong non-churn and 8% customers belong to Churn type.__

# In[58]:


# check basic statistics
telecom_highvalue.describe()


# In[59]:


#check columns associated with month 6, From month 6 we can figure out how the columns and data are in other months
cols = [col for col in telecom_highvalue.columns if "_6" in col]
cols


# ### Derive new faetures by comparing month 8 features  vs month 6 and month 7 features.

# In[60]:


#compare average revenue and calculate the difference
telecom_highvalue['arpu_diff'] = telecom_highvalue.arpu_8 - ((telecom_highvalue.arpu_6 + telecom_highvalue.arpu_7)/2)

# Check various columns related to Minutes of Usage and calculate difference
telecom_highvalue['onnet_mou_diff'] = telecom_highvalue.onnet_mou_8 - ((telecom_highvalue.onnet_mou_6 + telecom_highvalue.onnet_mou_7)/2)
telecom_highvalue['offnet_mou_diff'] = telecom_highvalue.offnet_mou_8 - ((telecom_highvalue.offnet_mou_6 + telecom_highvalue.offnet_mou_7)/2)
telecom_highvalue['roam_ic_mou_diff'] = telecom_highvalue.roam_ic_mou_8 - ((telecom_highvalue.roam_ic_mou_6 + telecom_highvalue.roam_ic_mou_7)/2)
telecom_highvalue['roam_og_mou_diff'] = telecom_highvalue.roam_og_mou_8 - ((telecom_highvalue.roam_og_mou_6 + telecom_highvalue.roam_og_mou_7)/2)
telecom_highvalue['loc_og_mou_diff'] = telecom_highvalue.loc_og_mou_8 - ((telecom_highvalue.loc_og_mou_6 + telecom_highvalue.loc_og_mou_7)/2)
telecom_highvalue['std_og_mou_diff'] = telecom_highvalue.std_og_mou_8 - ((telecom_highvalue.std_og_mou_6 + telecom_highvalue.std_og_mou_7)/2)
telecom_highvalue['isd_og_mou_diff'] = telecom_highvalue.isd_og_mou_8 - ((telecom_highvalue.isd_og_mou_6 + telecom_highvalue.isd_og_mou_7)/2)
telecom_highvalue['spl_og_mou_diff'] = telecom_highvalue.spl_og_mou_8 - ((telecom_highvalue.spl_og_mou_6 + telecom_highvalue.spl_og_mou_7)/2)
telecom_highvalue['total_og_mou_diff'] = telecom_highvalue.total_og_mou_8 - ((telecom_highvalue.total_og_mou_6 + telecom_highvalue.total_og_mou_7)/2)
telecom_highvalue['loc_ic_mou_diff'] = telecom_highvalue.loc_ic_mou_8 - ((telecom_highvalue.loc_ic_mou_6 + telecom_highvalue.loc_ic_mou_7)/2)
telecom_highvalue['std_ic_mou_diff'] = telecom_highvalue.std_ic_mou_8 - ((telecom_highvalue.std_ic_mou_6 + telecom_highvalue.std_ic_mou_7)/2)
telecom_highvalue['isd_ic_mou_diff'] = telecom_highvalue.isd_ic_mou_8 - ((telecom_highvalue.isd_ic_mou_6 + telecom_highvalue.isd_ic_mou_7)/2)
telecom_highvalue['spl_ic_mou_diff'] = telecom_highvalue.spl_ic_mou_8 - ((telecom_highvalue.spl_ic_mou_6 + telecom_highvalue.spl_ic_mou_7)/2)
telecom_highvalue['total_ic_mou_diff'] = telecom_highvalue.total_ic_mou_8 - ((telecom_highvalue.total_ic_mou_6 + telecom_highvalue.total_ic_mou_7)/2)

# Check total Recharge number
telecom_highvalue['total_rech_num_diff'] = telecom_highvalue.total_rech_num_8 - ((telecom_highvalue.total_rech_num_6 + telecom_highvalue.total_rech_num_7)/2)
#check total recharge amount
telecom_highvalue['total_rech_amt_diff'] = telecom_highvalue.total_rech_amt_8 - ((telecom_highvalue.total_rech_amt_6 + telecom_highvalue.total_rech_amt_7)/2)
#Check maximum recharge amount
telecom_highvalue['max_rech_amt_diff'] = telecom_highvalue.max_rech_amt_8 - ((telecom_highvalue.max_rech_amt_6 + telecom_highvalue.max_rech_amt_7)/2)
#check total recharge data
telecom_highvalue['total_rech_data_diff'] = telecom_highvalue.total_rech_data_8 - ((telecom_highvalue.total_rech_data_6 + telecom_highvalue.total_rech_data_7)/2)
#check maximum recharge data
telecom_highvalue['max_rech_data_diff'] = telecom_highvalue.max_rech_data_8 - ((telecom_highvalue.max_rech_data_6 + telecom_highvalue.max_rech_data_7)/2)
#Check average recharge amount in Data
telecom_highvalue['av_rech_amt_data_diff'] = telecom_highvalue.av_rech_amt_data_8 - ((telecom_highvalue.av_rech_amt_data_6 + telecom_highvalue.av_rech_amt_data_7)/2)
#check 2G data consumption difference in MB
telecom_highvalue['vol_2g_mb_diff'] = telecom_highvalue.vol_2g_mb_8 - ((telecom_highvalue.vol_2g_mb_6 + telecom_highvalue.vol_2g_mb_7)/2)
#Check 3G data consumption in MB
telecom_highvalue['vol_3g_mb_diff'] = telecom_highvalue.vol_3g_mb_8 - ((telecom_highvalue.vol_3g_mb_6 + telecom_highvalue.vol_3g_mb_7)/2)


# In[61]:


# Plot to visualize average revenue per user(ARPU)
telecom_highvalue.groupby("churn")["arpu_6","arpu_7","arpu_8"].median().plot.bar(figsize=[8,6])
plt.title("Average revenue per user in month 6,7,8",fontsize=15)
plt.tick_params(size=5,labelsize = 15) 
plt.ylabel("Median revenue",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.grid(0.3)
plt.show()


# - Average revenue per user more in month 6 means, if they are unsatisfied, those useres are more likely to churn

# In[62]:


## Plot to visualize onnet_mou
telecom_highvalue.groupby("churn")["onnet_mou_6","onnet_mou_7","onnet_mou_8" ].median().plot.bar(figsize=[8,6])
plt.tick_params(size=5,labelsize = 15) 
plt.title("Minutes of usage inside network in month 6,7,8",fontsize=15)
plt.ylabel("median",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.grid(0.3)
plt.show()


#  - __Users whose minutes of usage are more in month 6, they are more likely to churn.__

# In[63]:


# Plot to visualize  offnet_mou
telecom_highvalue.groupby("churn")["offnet_mou_6","offnet_mou_7","offnet_mou_8" ].median().plot.bar(figsize=[8,6])
plt.tick_params(size=5,labelsize = 15) 
plt.title("Minutes of usage outside network in month 6,7,8",fontsize=15)
plt.ylabel("median",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.grid(0.3)
plt.show()


# - __The users who have big difference of minutes of call duration  to other network between month 6 and month 7,are likely to churn.__ 

# In[64]:


# Plot to visualize total_rech_amt
telecom_highvalue.groupby("churn")["total_rech_amt_6","total_rech_amt_7","total_rech_amt_8" ].median().plot.bar(figsize=[8,6])
plt.tick_params(size=5,labelsize = 15) 
plt.title("Total Recharge amount in month 6,7,8",fontsize=15)
plt.ylabel("median",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.grid(0.3)
plt.show()


# -  __when the difference of total recharge amount is more, those users are more likely to churn.__

# In[65]:


# Plot to visualize total_rech_data_
telecom_highvalue.groupby("churn")["total_rech_data_6","total_rech_data_7","total_rech_data_8" ].mean().plot.bar()
plt.tick_params(size=5,labelsize = 15) 
plt.title("Total Recharge data in month 6,7,8",fontsize=15)
plt.ylabel("median",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.grid(0.3)
plt.show()


#  -  __Users who have not recharge in month 6, 7, 8 may or may not churn, we do not have much evidence from data.__

# In[66]:


## Plot to visualize vol_2g_mb_6
telecom_highvalue.groupby("churn")["vol_2g_mb_6","vol_2g_mb_7","vol_2g_mb_8" ].median().plot.bar(figsize=[7,5])
plt.tick_params(size=5,labelsize = 15) 
plt.title("2G recharge in month 6,7,8",fontsize=15)
plt.ylabel("median",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.grid(0.3)
plt.show()


#  - 2g recharge who have not done may or may not churn,There is no concrete evidence from data

# In[67]:


#Check the percenatges of churn in each category of Night Pack Users in month 8
pd.crosstab(telecom_highvalue.churn, telecom_highvalue.night_pck_user_8, normalize='columns')*100


# In[68]:


#Check the percenatges of churn in each category of Facebook Users in month 6
(pd.crosstab(telecom_highvalue.churn, telecom_highvalue.fb_user_8, normalize='columns')*100)


# - Night pack users(which we do not know whether using or not) in month 8 , high churn rate: close to 14%
# - Among Facebook users in month 8,  close to 2%  churns
# - Customers who are not using facebook, close to 7% churns in month 8
#  
#  

# In[69]:


# plot to visualize av_rech_amt_data
telecom_highvalue.groupby("churn")["av_rech_amt_data_6","av_rech_amt_data_7","av_rech_amt_data_8" ].median().plot.\
bar(figsize=[7,5])

plt.tick_params(size=5,labelsize = 15) 
plt.title("Average recharge amount in  month 6,7,8",fontsize=15)
plt.ylabel("median",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.grid(0.3)
plt.show()


# - Average recharge amount in  month 6,7,8 is none, from dataset, they are more likely to churn

# In[70]:


#Plot to visualize total_ic_mou
telecom_highvalue.groupby("churn")["total_ic_mou_6","total_ic_mou_7","total_ic_mou_8"].median().plot.bar(figsize=[6,5])
plt.tick_params(size=5,labelsize = 15) 
plt.title("Total incoming minute in  month 6,7,8",fontsize=15)
plt.ylabel("median",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.grid(0.3)
plt.show()


#  - Users who have more difference in Total incoming minutes in month 6,7,8 are more likely to churn

# In[71]:


#plot to visualize loc_og_mou
telecom_highvalue.groupby("churn")["loc_og_mou_6","loc_og_mou_7","loc_og_mou_8"].median().plot.bar(figsize=[6,5])
plt.tick_params(size=5,labelsize = 15) 
plt.title("local outgoing  minute in  month 6,7,8",fontsize=15)
plt.ylabel("median",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.grid(0.3)
plt.show()


#  - local outgoing minute are less, users are more likely to churn

# In[72]:


# total_og_mou_6
telecom_highvalue.groupby("churn")["total_og_mou_6","total_og_mou_7","total_og_mou_8"].median().plot.bar(figsize=[6,5])
plt.tick_params(size=5,labelsize = 15) 
plt.title("local outgoing  minute in  month 6,7,8",fontsize=15)
plt.ylabel("median",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.grid(0.3)
plt.show()


#  -  Total outgoing minute usage difference is more between month 6 and 7, users are mor likely to chrun 

# In[73]:


# loc_og_t2t_mou_6
telecom_highvalue.groupby("churn")["loc_og_t2t_mou_6","loc_og_t2t_mou_7","loc_og_t2t_mou_8"].median().plot.bar(figsize=[6,5])
plt.tick_params(size=5,labelsize = 15) 
plt.title("local outgoing  minute in same operator in  month 6,7,8",fontsize=15)
plt.ylabel("median",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.grid(0.3)
plt.show()


# - Local outgoing  minute in same operator in  month 6,7,8 are less, users are more likely to churn.

# In[74]:


telecom_highvalue.groupby("churn")["loc_og_t2m_mou_6","loc_og_t2m_mou_7","loc_og_t2m_mou_8"].median().plot.bar(figsize=[6,5])
plt.tick_params(size=5,labelsize = 15) 
plt.title("Local outgoing  minute to other operator in  month 6,7,8",fontsize=15)
plt.ylabel("median",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.grid(0.3)
plt.show()


# - Local outgoing  minute to other operator is less, more likely to churn

# In[75]:


plt.figure(figsize=[8,6])
sns.boxplot(data=telecom_highvalue,x="churn",y="aon")
plt.tick_params(size=5,labelsize = 15) 
plt.title("Age of Network",fontsize=15)
plt.xlabel("Churn type",fontsize=15)
plt.ylabel("Age of Network",fontsize=15)
plt.grid(0.3)
plt.show()


#  -  Median Age of network less,more likely to churn

# In[76]:


telecom_highvalue.groupby("churn")["std_ic_t2t_mou_6","std_ic_t2t_mou_7","std_ic_t2m_mou_8"].median().plot.bar(figsize=[6,5])
plt.show()


#  -  Users who are using more STD calls are more likely to churn.

# In[77]:


telecom_highvalue.groupby("churn")["roam_ic_mou_6","roam_ic_mou_7","roam_ic_mou_8"].mean().plot.bar()
plt.show()


#  - Roaming in incoming minutes more, they are likely to churn more.

# In[78]:


telecom_highvalue.groupby("churn")["roam_og_mou_6","roam_og_mou_7","roam_og_mou_8"].mean().plot.bar()
plt.show()


#  -  roaming in outgoing minutes more, Users are more likely to churn.

# In[79]:


telecom_highvalue.head()


# ## Model Building

# In[80]:


#Load required library
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier


# ## Train test split of data

# In[81]:


#Peform the train test split 
train,test = train_test_split(telecom_highvalue,test_size=0.2,random_state=48)


# In[82]:


# check the training and testing data shape
print(f"train data shape:{train.shape}")
print(f"Test data shape:{test.shape}")


# In[83]:


#Convert categorical data to numeric columns by aggregation.
categorical_columns = ["night_pck_user_6","night_pck_user_7",
                       "night_pck_user_8","fb_user_6",          
                       "fb_user_8","fb_user_7"]  


# In[84]:


train[categorical_columns].head()


# In[85]:


#Calculate categorical features mean and replace those with categorical value
print(train.groupby('night_pck_user_6')["churn"].mean())
print(train.groupby('night_pck_user_7')["churn"].mean())
print(train.groupby('night_pck_user_8')["churn"].mean())
print(train.groupby('fb_user_6')["churn"].mean())
print(train.groupby('fb_user_7')["churn"].mean())
print(train.groupby('fb_user_8')["churn"].mean())


# In[86]:


#Map each categorical value with mean value
mapping = {'night_pck_user_6' : {-1: 0.099621, 0: 0.066717, 1: 0.098462},
           'night_pck_user_7' : {-1: 0.116741, 0: 0.054784, 1: 0.058020},
           'night_pck_user_8' : {-1: 0.141980, 0: 0.028647, 1: 0.019084},
           'fb_user_6'        : {-1: 0.099621, 0: 0.083333, 1: 0.066233},
           'fb_user_7'        : {-1: 0.116741, 0: 0.065279, 1: 0.053977},
           'fb_user_8'        : {-1: 0.141980, 0: 0.067373, 1: 0.023955}}

#convert categorical to Numeric features by aggregation and replace in train data
train.replace(mapping, inplace = True)
#replace the same in test data
test.replace(mapping, inplace = True)


# In[87]:


# segregate  X_train and y_train 
y_train = train.pop("churn")
X_train = train


# In[88]:


# Segregate X_test and y_test
y_test = test.pop("churn")
X_test = test


# ## Perform Oversampling with  SMOTE

# - As we have imbalance data set, we will oversample only the training set data

# In[89]:


# Perform oversampling with traing data and pass both X_train and y_train to SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=48)
X_train_resample,y_train_resample = smote.fit_resample(X_train,y_train)


# In[90]:


# Check the shape after Oversampling
print(f"Shape of train data after oversampling: {X_train_resample.shape}")
print(f"Value count of training target variable:\n{y_train_resample.value_counts()}")


# - __Now the non-churn and churn data is balanced.__

# ### Scaling
# - We need to perform the scaling to feed the scaled data to PCA
# - We are using minmax scaling 

# In[91]:


# Import library and perform scaling
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scale = MinMaxScaler()
temp_x_train = scale.fit_transform(X_train_resample)

#Form the dataframe after scaling
X_train_scale = pd.DataFrame(temp_x_train,columns=X_train.columns)
# Check the shape of scaled data
X_train_scale.shape


# In[92]:


# check the scaled train data head 
X_train_scale.head()


# In[93]:


# Perform the scaling on test set
temp_x_test = scale.transform(X_test)
# form the test set dataframe after scaling
X_test_scale = pd.DataFrame(temp_x_test,columns=X_test.columns)


# In[94]:


# check the scaled test data head 
X_test_scale.head()


# - Use X_train_scale and X_test_scale in PCA

# ## PCA
# 
# - we have almost 140 features to train the model
# - to remove collinearity and faster training we can perform dimensionality reduction technique PCA.

# In[95]:


# Load the library
from sklearn.decomposition import PCA
pc_class = PCA(random_state=60)
X_train_pca = pc_class.fit(X_train_scale)


# In[96]:


# Check the explained_variance_ratio_ whihc tells us individual principal component variance.
X_train_pca.explained_variance_ratio_


# In[97]:


# perform the cumulaltive sum of explained variance
var_cumu = np.cumsum(X_train_pca.explained_variance_ratio_)
#Convert explained variance to DataFrame
var_cumu_df = pd.DataFrame({"variance":var_cumu}) 
var_cumu_df.head(30) 


# In[98]:


# Plot the cumulative explained variance : SCREE Plot
plt.figure(figsize=[6,4])
plt.plot(range(1,len(var_cumu)+1), var_cumu)
plt.title("Cumulative variance of principal components",size=15)
plt.ylabel("Explained variance",size=15)
plt.xlabel("No of Features",size=15)
plt.tick_params(size=5,labelsize = 15) # Tick size in both X and Y axes
plt.grid(0.3)


# In[99]:


# By providing variance value we can also get the suitable principal components.
pca_demo = PCA(0.96,random_state=40)
X_train_pca1 = pca_demo.fit_transform(X_train_scale)
print(f"suitable principal components for 96% of variance:{X_train_pca1.shape[1]}")


# - __Now we got suitable no of principal components as 17__
# - __Hence we will do PCA again with 18 components for train and test set.__

# In[100]:


# Instantiate PCA with 17 components 
pca_object = PCA(n_components=17,random_state=48)
# get the PCs for train data
X_train_pca_final = pca_object.fit_transform(X_train_scale)
# get the PCs for test data
X_test_pca_final = pca_object.fit_transform(X_test_scale)

#check the shape of train and test data after PCA
print(X_train_pca_final.shape)
print(X_test_pca_final.shape)


# In[101]:


# Check the correlations after PCA
np.corrcoef(X_train_pca_final.transpose())


#  - __The correlation values are almost close to 0( power raised to -17,-18,-19) except the diagonal.__

# ## Model Building:
# 
# - We will explore below models.
#     - Logistic regression
#     - Decision tree
#     - Randomforest
#     - Gradientboosting
#     - XGboost
# 

# In[102]:


#Function definition to check the performance of model on test data
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
# Check the performance on test set
#Precision
#recall
#f1_score
#ROC_AUC
def calculate_peformance_testdata(model_name,y_test,y_pred,pred_prob):
        
    '''y_test:Test Labels,
       y_pred: Prediction Labels ,
       pred_prob:Predicted Probability  '''
    
    print(f"{model_name}:")
    precision = metrics.precision_score(y_test,y_pred)
    print(f"precision: {precision}")
    recall = metrics.recall_score(y_test,y_pred)
    print(f"recall: {recall}")
    f1_score = metrics.f1_score(y_test,y_pred)
    print(f"f1_score: {f1_score}")
    roc_auc = metrics.roc_auc_score(y_test,pred_prob)
    print(f"roc_auc: {roc_auc}")
#     return a DataFrame with all the score
    return pd.DataFrame({"Model":[model_name],"precision":[precision],"recall":[recall],"f1_score":[f1_score],
                         "roc_auc":[roc_auc]})   


# In[103]:


# Create a DataFrame which stores all test score for each model
score_df = pd.DataFrame({"Model":[None],"precision":[None],"recall":[None],"f1_score":[None],"roc_auc":[None]})


# ### Logistic regression

# In[104]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#Instantiate logistic regression
lr_obj = LogisticRegression(random_state=40)
#pass PCA data as input
lr_obj.fit(X_train_pca_final, y_train_resample)
cv_score = cross_val_score(lr_obj, X_train_pca_final, y_train_resample, cv=5, scoring='f1_micro')
print(f"Cross validation score: {cv_score}")


# In[105]:


#Prediction on  pca testdata
y_pred_lr = lr_obj.predict(X_test_pca_final)
#check predict probability on pca data
pred_prob = lr_obj.predict_proba(X_test_pca_final)


# In[106]:


#check various scores on test data
df1 = calculate_peformance_testdata("LogisticRegression",y_test,y_pred_lr,pred_prob[:,1])


# In[107]:


#Add the score to dataframe for comparision with other model performance
score_df= score_df.dropna()
score_df = score_df.append(df1)
score_df


# In[172]:


#Plot confusion matrix for Logistic Regression 
#metrics.plot_confusion_matrix(lr_obj, X_test_pca_final, y_test)
#plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming lr_obj is your trained logistic regression classifier
# Assuming X_test_pca_final and y_test are your test data

# Compute confusion matrix
conf_mat = confusion_matrix(y_test, lr_obj.predict(X_test_pca_final))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[173]:


#Plot ROC_AUC Curve for Logistic Regression
#metrics.plot_roc_curve(lr_obj, X_test_pca_final, y_test)
#plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, lr_obj.predict_proba(X_test_pca_final)[:,1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# ## DecisionTree
# 
# X_train_resample,y_train_resample

# In[110]:


from sklearn.tree import DecisionTreeClassifier
#Instantiate Decision tree with defautl parameter
dt_obj=  DecisionTreeClassifier(random_state=40)

# here we have used data generated by SMOTE. 
dt_obj.fit(X_train_scale, y_train_resample)
cv_score = cross_val_score(dt_obj, X_train_scale, y_train_resample, cv=5, scoring='f1_micro')
print(cv_score)


# In[111]:


#check the default paramters 
dt_obj.get_params()


# In[113]:


#Perform hyperparamter tuning with randomizedsearchcv
param_grid = dict({"max_leaf_nodes":[4,5,6],"min_samples_leaf":[3,4,5],'min_samples_split':[3,4,5]})
dt_clf = DecisionTreeClassifier(random_state=40)
dt_clf_rcv = RandomizedSearchCV(dt_clf,param_grid,cv=5,scoring="f1_micro")# n_jobs=-1
dt_clf_rcv.fit(X_train_scale, y_train_resample)


# In[114]:


#check the beat score and best estimator paramters
print(dt_clf_rcv.best_score_)
print(dt_clf_rcv.best_estimator_)


# In[115]:


# dt_clf_rcv.cv_results_


# In[116]:


#Train the decision tree with best paramters obtained from above step
# Commented out the hyperparamter tuning as it takes sometime to execute
dt_clf = DecisionTreeClassifier(max_leaf_nodes=6,min_samples_leaf=4,min_samples_split=5,random_state=40)
dt_clf.fit(X_train_scale,y_train_resample)


# In[117]:


#perform the prediction 
y_pred_dt = dt_clf.predict(X_test_scale)
#Perform the prediction probability
pred_prob = dt_clf.predict_proba(X_test_scale)


# In[118]:


##check the scores.
df2 = calculate_peformance_testdata("DecisionTree",y_test,y_pred_dt,pred_prob[:,1])


# In[119]:


#Add the score to Dataframe  for comparision 
score_df = score_df.append(df2)
score_df.dropna(inplace=True)
score_df.drop_duplicates(inplace=True)
score_df


# In[171]:


#visualize the confusion matrix
#metrics.plot_confusion_matrix(dt_clf, X_test_scale, y_test)
#plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Compute confusion matrix
conf_mat = confusion_matrix(y_test, dt_clf.predict(X_test_scale))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[123]:


#plot the ROC_AUC curve


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, dt_clf.predict_proba(X_test_scale)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# ## Randomforest

# In[124]:


#Instantiate RandomForest, train with default parameters
rf_class = RandomForestClassifier(n_jobs=-1) #class_weight={0:1,1:2}
rf_class.fit(X_train_scale,y_train_resample)
y_pred_rf = rf_class.predict(X_test_scale)
pred_prob = rf_class.predict_proba(X_test_scale)


# In[125]:


#check the default parameters
rf_class.get_params()


# In[126]:


#Randomizedsearch cross validation is commented out for faster execution.


#perform hyperparameter tuning
# param_grid = dict({"n_estimators":[90,110],"min_samples_split":[2,3],"min_samples_leaf":[2,3]})
# rf_class = RandomForestClassifier(random_state=40,n_jobs=-1)
# rf_clf_rcv = RandomizedSearchCV(rf_class,param_grid,cv=5,scoring="f1_micro")
# rf_clf_rcv.fit(X_train_scale,y_train_resample)


# In[127]:


#check the best parameters and score in cross validation 
# print(rf_clf_rcv.best_score_)
# print(rf_clf_rcv.best_estimator_)


# In[128]:


#Use best paramters to train the model
rf_class = RandomForestClassifier(min_samples_leaf=3,n_estimators=120,n_jobs=-1,random_state=40)
rf_class.fit(X_train_scale,y_train_resample)
y_pred_rf = rf_class.predict(X_test_scale)
pred_prob = rf_class.predict_proba(X_test_scale)


# In[129]:


#check the scores 
df3 = calculate_peformance_testdata("RandomForest",y_test,y_pred_rf,pred_prob[:,1])


# In[130]:


#Add score to the dataframe for comparision 
score_df = score_df.append(df3)
score_df


# In[132]:


#visualize confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get predictions
y_pred = rf_class.predict(X_test_scale)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# In[135]:


#plot roc auc cureve

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, rf_class.predict_proba(X_test_scale)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# ## GradientBoosting

# In[136]:


#Train gradient boosting with default parameters
from sklearn.ensemble import GradientBoostingClassifier
gb_class = GradientBoostingClassifier(random_state=42,min_samples_leaf=4,min_samples_split=5)
# n_estimators=110,min_samples_leaf=2,min_samples_split=3,learning_rate=0.2
gb_class.fit(X_train_scale,y_train_resample)

#get the predicated label
y_pred_gb = gb_class.predict(X_test_scale)
#get the predicted probability
pred_prob = gb_class.predict_proba(X_test_scale)


# In[137]:


#check the training default parameters
gb_class.get_params()


# In[138]:


#Hyperparameter tuning 
# %time
# param_grid = dict({"n_estimators":[90,110],
#                    "min_samples_split":[2,4],
#                    "min_samples_leaf":[2,3],
#                    "learning_rate":[.2,.3]})

# gb_class_cv = GradientBoostingClassifier(random_state=40)
# gb_clf_rcv = RandomizedSearchCV(gb_class_cv,param_grid,cv=5,scoring="f1_micro")
# gb_clf_rcv.fit(X_train_scale,y_train_resample)


# In[139]:


#check the scores and best paramters 
# print(gb_clf_rcv.best_score_)
# print(gb_clf_rcv.best_estimator_)


# In[140]:


#Check the test scores
df4 = calculate_peformance_testdata("GradientBoosting",y_test,y_pred_gb,pred_prob[:,1])


# In[141]:


#Add the scores to dataframe
score_df=score_df.append(df4)
score_df


# In[143]:


#Plot the confusion matrix
#metrics.plot_confusion_matrix(gb_class, X_test, y_test)
#plt.show()
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
conf_mat = confusion_matrix(y_test, gb_class.predict(X_test))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[145]:


#plot the roc curve
#metrics.plot_roc_curve(gb_class, X_test_scale, y_test)
#plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, gb_class.predict_proba(X_test_scale)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# ## Xgboost

# In[146]:


# !pip install xgboost


# In[147]:


import xgboost as xgb


# In[148]:


# Model training with default paamters

xgb_class = xgb.XGBClassifier(max_depth=10)
xgb_class.fit(X_train_scale,y_train_resample)

#Model prediction 
y_pred_xgb = xgb_class.predict(X_test_scale)
#Model predict probability
pred_prob = xgb_class.predict_proba(X_test_scale)


# In[149]:


#check the model default paramters
xgb_class.get_params()


# In[150]:


# # check the time it takes for model training
# %time   
# #Hyperparamter tuning 
# param_grid = dict({"n_estimators":[90,110],
#                    "subsample":[0,1],
#                    "max_depth": [5,7,10],
#                    "learning_rate":[.2,.4]})

# xgb_class_cv = GradientBoostingClassifier(random_state=40)
# xgb_clf_rcv = RandomizedSearchCV(xgb_class_cv,param_grid,cv=5,scoring="f1_micro")
# xgb_clf_rcv.fit(X_train_scale,y_train_resample)


# In[151]:


#Check the best scores and best paramters
# print(xgb_clf_rcv.best_score_)
# print(xgb_clf_rcv.best_estimator_)


# In[152]:


#
# xgb_class = xgb.XGBClassifier()


# In[153]:


#predict the  labels of test data
# y_pred_xgb = xgb_class.predict(X_test_scale)
# #chekc the predict probability
# pred_prob = xgb_class.predict_proba(X_test_scale)


# In[154]:


#check the scores
df5 = calculate_peformance_testdata("XGBoost",y_test,y_pred_xgb,pred_prob[:,1])


# In[155]:


#add the score to dataframe
score_df= score_df.append(df5)
score_df.drop_duplicates()


# In[157]:


#Plot confusion matrix
#metrics.plot_confusion_matrix(xgb_class, X_test_scale, y_test)
#plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
conf_mat = confusion_matrix(y_test, xgb_class.predict(X_test_scale))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[159]:


#plot roc curve
#metrics.plot_roc_curve(xgb_class, X_test_scale, y_test)
#plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, xgb_class.predict_proba(X_test_scale)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[160]:


#check how various model is performing on test set on Churn=1.
score_df


#  - __The randomforest worked well on this data in churn with precision  close to 59%, recall close to 65% and f1_score close to 61%.__
#  - In Logistic regression we have used PCA.
#  -  In this scenario, Without PCA model works well.

# ## Fearure Importance and Model Interpretation
# 

# In[161]:


# Randomforest model training 
gb_object = RandomForestClassifier(random_state=40)
gb_object.fit(X_train_resample,y_train_resample)
y_pred = gb_object.predict(X_test)


# In[162]:


#check the performance on test data
calculate_peformance_testdata("RandomForest",y_test,y_pred,pred_prob[:,1])


# In[164]:


#plot confusion matrix
#from sklearn.metrics import plot_confusion_matrix
#plot_confusion_matrix(gb_object, X_test, y_test)
#plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
conf_mat = confusion_matrix(y_test, gb_object.predict(X_test))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[166]:


#plot ROC curve
#metrics.plot_roc_curve(gb_object, X_test, y_test) 
#plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, gb_object.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[167]:


#check the classification report
print(metrics.classification_report(y_test,y_pred))


# In[168]:


#Create a Feature importance dataframe
Feature_importance = pd.DataFrame({"columns":X_train.columns,"feature_importance":gb_object.feature_importances_})


# In[169]:


#check 40 important features
fi = Feature_importance.sort_values(by="feature_importance",ascending=False).head(40)
fi


# In[170]:


#Plot to show the feature importance
plt.figure(figsize=[20,15])
sns.barplot(x = "columns",y="feature_importance",data=fi)
plt.title("Feature Importance",size=15)
plt.xticks(rotation="vertical")
plt.ylabel("Coefficient Magnitude",size=15)
plt.xlabel("Features",size=15)
plt.tick_params(size=5,labelsize = 15) # Tick size in both X and Y axes
plt.grid(0.3)


# ## Conclusion:
#  - The most important features are as shown in above graph.
#  - Average revenue per user more, those are likely to churn  if they are not happy with the network.
#  - local calls minutes of usage has also has impact on churn .
#  - Large difference between recharge amount between 6th and 7th month, also impact churn.
#  - Users who are using more Roaminng in Outgoing and Incoming calls, are likely to churn.Compnay can focus on them too.
# 
#     
#  

# In[ ]:




