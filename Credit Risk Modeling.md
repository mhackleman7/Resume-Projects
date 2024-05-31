import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#Create a backup copy of data
loan_data_backup = pd.read_csv('C:/Users/mhack/OneDrive/Documents/Projects/loan_data_2007_2014.csv')

loan_data_backup

#Create our working dataset
loan_data = loan_data_backup.copy()

loan_data

pd.options.display.max_columns = None

loan_data

loan_data.columns.values

loan_data.info()

#Turning Emp_Length variable into a numeric value to work with
loan_data['emp_length'].unique()
loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('+ years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a', str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year', '')
type(loan_data['emp_length_int'][0])
loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])
type(loan_data['emp_length_int'][0])

#Turning Term variable into a numeric value to work with
loan_data['term'].unique()
loan_data['term_int'] = loan_data['term'].str.replace(' months', '')
type(loan_data['term_int'][25])
loan_data['term_int'] = pd.to_numeric(loan_data['term_int'])
type(loan_data['term_int'][0])

#Turning earliest_cr_line variable into a numeric value to work with
loan_data['earliest_cr_line']
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format = '%b-%y')
type(loan_data['earliest_cr_line_date'][0])
pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']
loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'D')))
loan_data['mths_since_earliest_cr_line']
reference_date = pd.to_datetime('2017-12-01')
loan_data['days_since_earliest_cr_line'] = (reference_date - loan_data['earliest_cr_line_date']).dt.days
loan_data['mths_since_earliest_cr_line'] = round(loan_data['days_since_earliest_cr_line'] / 30)
loan_data['mths_since_earliest_cr_line'].describe()
loan_data.loc[: , ['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']][loan_data['mths_since_earliest_cr_line'] < 0]
loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line'] < 0] = loan_data['mths_since_earliest_cr_line'].max()
min(loan_data['mths_since_earliest_cr_line'])

#Turning issue_d variable into a numeric value to work with
loan_data['issue_d']
loan_data['issue_d'] = pd.to_datetime(loan_data['issue_d'], format = '%b-%y')
type(loan_data['issue_d'][0])
pd.to_datetime('2017-12-01') - loan_data['issue_d']
loan_data['days_since_issue_d'] = (reference_date - loan_data['issue_d']).dt.days
loan_data['mths_since_issue_d'] = round(loan_data['days_since_issue_d'] / 30)
loan_data['mths_since_issue_d'].describe()

#Creating dummy variables for a logistic regression to predict default
loan_data_dummies = pd.concat([
    pd.get_dummies(loan_data['grade'], prefix='grade', prefix_sep=':'),
    pd.get_dummies(loan_data['sub_grade'], prefix='sub_grade', prefix_sep=':'),
    pd.get_dummies(loan_data['home_ownership'], prefix='home_ownership', prefix_sep=':'),
    pd.get_dummies(loan_data['verification_status'], prefix='verification_status', prefix_sep=':'),
    pd.get_dummies(loan_data['loan_status'], prefix='loan_status', prefix_sep=':'),
    pd.get_dummies(loan_data['purpose'], prefix='purpose', prefix_sep=':'),
    pd.get_dummies(loan_data['addr_state'], prefix='addr_state', prefix_sep=':'),
    pd.get_dummies(loan_data['initial_list_status'], prefix='initial_list_status', prefix_sep=':')
], axis=1)

#Adding back in the dummy variables into our dataset
loan_data = pd.concat([loan_data, loan_data_dummies], axis=1)

#Raplcing null values with logical values
loan_data.columns.values
loan_data.isnull()
pd.options.display.max_rows = None
loan_data.isnull().sum()
pd.options.display.max_rows = 100
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True)
loan_data['total_rev_hi_lim'].isnull()
loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)
loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
loan_data['acc_now_delinq'].fillna(0, inplace=True)
loan_data['total_acc'].fillna(0, inplace=True)
loan_data['pub_rec'].fillna(0, inplace=True)
loan_data['open_acc'].fillna(0, inplace=True)
loan_data['inq_last_6mths'].fillna(0, inplace=True)
loan_data['delinq_2yrs'].fillna(0, inplace=True)
loan_data['emp_length_int'].fillna(0, inplace=True)
pd.options.display.max_rows = None
loan_data.isnull().sum()

#Ratio of good vs bad loan status
loan_data['loan_status'].unique()
loan_data['loan_status'].value_counts()
loan_data['loan_status'].value_counts() / loan_data['loan_status'].count()
loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'Late (31-120 days)']),0, 1)
loan_data['good_bad']

#Splitting data into train/test to make sure our model is not under/over fit
train_test_split(loan_data.drop('good_bad', axis = 1), loan_data['good_bad'])
loan_data_inputs_train, loan_data_inputs_test,loan_data_targets_train, loan_data_targets_test = train_test_split(loan_data.drop('good_bad', axis = 1), loan_data['good_bad'], test_size = 0.2, random_state = 42)
loan_data_inputs_train.shape
loan_data_targets_train.shape
loan_data_inputs_test.shape
loan_data_targets_test.shape

#Creating dataframes for our preprocessing
#df_inputs_prepr = loan_data_inputs_train #Commented Out after running the code up to line 706
#df_targets_prepr = loan_data_targets_train #Commented Out after running the code up to line 706 and run below 2 lines
df_inputs_prepr = loan_data_inputs_test
df_targets_prepr = loan_data_targets_test

df_inputs_prepr['grade'].unique()
df1 = pd.concat([df_inputs_prepr['grade'], df_targets_prepr], axis = 1)
df1.head()
df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].count()
df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].mean()

#Merging into 1 dataframe
df1 = pd.concat([df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].count(), df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].mean()], axis = 1)
df1
df1 = df1.iloc[: , [0, 1, 3]]
df1
df1.columns = [df1.columns.values[0], 'n_obs', 'prop_good']
df1
#Probability of Observations
df1['prop_n_obs'] = df1['n_obs'] / df1['n_obs'].sum()
df1
#Good vs Bad observations
df1['n_good'] = df1['prop_good'] * df1['n_obs']
df1['n_bad'] = (1 - df1['prop_good']) * df1['n_obs']
df1

#Calculate proportion of good vs bad for each grade
df1['prop_n_good'] = df1['n_good'] / df1['n_good'].sum()
df1['prop_n_bad'] = df1['n_bad'] / df1['n_bad'].sum()
df1

#Weight of Evidence Function
df1['WoE'] = np.log(df1['prop_n_good'] / df1['prop_n_bad'])
df1

#Sort by probability of highest default rate first
df1 = df1.sort_values(['WoE'])
df1 = df1.reset_index(drop = True)
df1
#Difference of proportion of good loans in subsequent categories and WoE
df1['diff_prop_good'] = df1['prop_good'].diff().abs()
df1['diff_WoE'] = df1['WoE'].diff().abs()
df1

#Calculate Information Value
df1['IV'] = (df1['prop_n_good'] - df1['prop_n_bad']) * df1['WoE']
df1['IV'] = df1['IV'].sum()
df1

#Automating the calculation now for IV for our other variables

def woe_discrete(df, discrete_varaible_name, good_bad_variable_df):
    df = pd.concat([df[discrete_varaible_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[: , [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df
    
df_temp = woe_discrete(df_inputs_prepr, 'grade', df_targets_prepr)
df_temp

#Plotting our information for interpretation 
def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 0):
    x = np.array(df_WoE.iloc[: , 0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize = (18, 6))
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    plt.xticks(rotation = rotation_of_x_axis_labels)
    
plot_by_woe(df_temp)    

#Creating more dummy variables and finding what to use as reference variables    
df_temp = woe_discrete(df_inputs_prepr, 'home_ownership', df_targets_prepr) 
df_temp 
plot_by_woe(df_temp)  
#This will show us we need to combine a few dummy variables because independently they don't have material impact and would overfit the model
df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_inputs_prepr['home_ownership:RENT'], df_inputs_prepr['home_ownership:OTHER'],
                                                             df_inputs_prepr['home_ownership:NONE'], df_inputs_prepr['home_ownership:ANY']])   

#More advanced way to get dummy variables
df_inputs_prepr['addr_state'].unique()
df_temp = woe_discrete(df_inputs_prepr, 'addr_state', df_targets_prepr)
df_temp    
plot_by_woe(df_temp)
if ['addr_state:ND'] in df_inputs_prepr.columns.values:
    pass
else:
    df_inputs_prepr['addr_state:ND'] = 0
    
plot_by_woe(df_temp.iloc[2: -2, :])    
#Gives us info to plot WV, NH, WY, DC, ME, ID (High) together and NE, IA, NV, FL, HI, AL (Low)
    
plot_by_woe(df_temp.iloc[6: -6, :]) #TOTAL LIST OF CATEGORIES BELOW
#NE, IA, NV, FL, HI, AL (Low - will be our reference variable)
#NM, VA 
#NY (high # obs) 
#OK, TN, MO, LA, MD, NC 
#CA (high # of obs)
#UT, KY, AZ, NJ can be combined
#AR, MI, PA, OH, MN
#RI, MA, DE, SD, IN
#GA, WA, OR
#WI, MT
#TX (high # of obs)
#IL, CT
#KS, SC, CO, VT, AK, MS
#WV, NH, WY, DC, ME, ID (High)
    
# Create a dictionary to map the new combined categories
combined_addr_state_categories = {
    'addr_state:ND_NE_IA_NV_FL_HI_AL': ['addr_state:ND', 'addr_state:NE', 'addr_state:IA', 'addr_state:NV', 'addr_state:FL', 'addr_state:HI', 'addr_state:AL'],
    'addr_state:NM_VA': ['addr_state:NM', 'addr_state:VA'],
    'addr_state:OK_TN_MO_LA_MD_NC': ['addr_state:OK', 'addr_state:TN', 'addr_state:MO', 'addr_state:LA', 'addr_state:MD', 'addr_state:NC'],
    'addr_state:UT_KY_AZ_NJ': ['addr_state:UT', 'addr_state:KY', 'addr_state:AZ', 'addr_state:NJ'],
    'addr_state:AR_MI_PA_OH_MN': ['addr_state:AR', 'addr_state:MI', 'addr_state:PA', 'addr_state:OH', 'addr_state:MN'],
    'addr_state:RI_MA_DE_SD_IN': ['addr_state:RI', 'addr_state:MA', 'addr_state:DE', 'addr_state:SD', 'addr_state:IN'],
    'addr_state:GA_WA_OR': ['addr_state:GA', 'addr_state:WA', 'addr_state:OR'],
    'addr_state:WI_MT': ['addr_state:WI', 'addr_state:MT'],
    'addr_state:IL_CT': ['addr_state:IL', 'addr_state:CT'],
    'addr_state:KS_SC_CO_VT_AK_MS': ['addr_state:KS', 'addr_state:SC', 'addr_state:CO', 'addr_state:VT', 'addr_state:AK', 'addr_state:MS'],
    'addr_state:WV_NH_WY_DC_ME_ID': ['addr_state:WV', 'addr_state:NH', 'addr_state:WY', 'addr_state:DC', 'addr_state:ME', 'addr_state:ID']
}

# Loop through the dictionary to create new combined binary columns for addr_state
for new_col, old_cols in combined_addr_state_categories.items():
    df_inputs_prepr[new_col] = np.where(df_inputs_prepr[old_cols].sum(axis=1) > 0, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

#Verification Status
df_temp = woe_discrete(df_inputs_prepr, 'verification_status', df_targets_prepr)
df_temp
plot_by_woe(df_temp)
#No combining these - Verified will be our reference variable

#Purpose
df_temp = woe_discrete(df_inputs_prepr, 'purpose', df_targets_prepr)
df_temp
plot_by_woe(df_temp, 90)

# Create a dictionary to map the new combined categories for purpose
combined_purpose_categories = {
    'purpose:educ__sm_b__wedd__ren_en__mov__house': ['purpose:educational', 'purpose:small_business', 'purpose:wedding', 'purpose:renewable_energy', 'purpose:moving', 'purpose:house'],
    'purpose:oth__med__vacation': ['purpose:other', 'purpose:medical', 'purpose:vacation'],
    'purpose:major_purch__car__home_impr': ['purpose:major_purchase', 'purpose:car', 'purpose:home_improvement']
}

# Loop through the dictionary to create new combined binary columns for purpose
for new_col, old_cols in combined_purpose_categories.items():
    df_inputs_prepr[new_col] = np.where(df_inputs_prepr[old_cols].sum(axis=1) > 0, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()


#Initial_List_Status
df_temp = woe_discrete(df_inputs_prepr, 'initial_list_status', df_targets_prepr)
df_temp
plot_by_woe(df_temp)

df_inputs_prepr.head()

def woe_ordered_continuous(df, discrete_varaible_name, good_bad_variable_df):
    df = pd.concat([df[discrete_varaible_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[: , [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

#Creating dummy variable for loan term
df_inputs_prepr['term_int'].unique()
df_temp = woe_ordered_continuous(df_inputs_prepr, 'term_int', df_targets_prepr)
df_temp
plot_by_woe(df_temp)
term_categories = {
    'term:36': 36,
    'term:60': 60
}

# Loop through the dictionary to create new binary columns for term_int
for new_col, term in term_categories.items():
    df_inputs_prepr[new_col] = np.where(df_inputs_prepr['term_int'] == term, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

#Dummy variable for emp_length
df_inputs_prepr['emp_length_int'].unique()
df_temp = woe_ordered_continuous(df_inputs_prepr, 'emp_length_int', df_targets_prepr)
df_temp
plot_by_woe(df_temp)
#Create categories of 0, 1, 2-4yrs, 5-6y, 7-9, 10
#0 is reference category
#Coarse Classing the variables for employment length
emp_length_categories = {
    'emp_length:0': [0],
    'emp_length:1': [1],
    'emp_length:2-4': range(2, 5),
    'emp_length:5-6': range(5, 7),
    'emp_length:7-9': range(7, 10),
    'emp_length:10': [10]
}

# Loop through the dictionary to create new binary columns for emp_length_int
for new_col, length_range in emp_length_categories.items():
    df_inputs_prepr[new_col] = np.where(df_inputs_prepr['emp_length_int'].isin(length_range), 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

#months since issue_date classing
df_inputs_prepr['mths_since_issue_d'].unique()
#Fine classing
df_inputs_prepr['mths_since_issue_d_factor'] = pd.cut(df_inputs_prepr['mths_since_issue_d'], 50)
#Coarse Classing
df_temp = woe_ordered_continuous(df_inputs_prepr, 'mths_since_issue_d_factor', df_targets_prepr)
df_temp
plot_by_woe(df_temp, rotation_of_x_axis_labels=90)
#First 3 variables are separate

plot_by_woe(df_temp.iloc[3: , :], 90)
#Categorize
mths_since_issue_d_categories = {
    'mths_since_issue_d:<38': [i for i in range(0, 38)],
    'mths_since_issue_d:38-39': range(38, 40),
    'mths_since_issue_d:40-41': range(40, 42),
    'mths_since_issue_d:42-48': range(42, 49),
    'mths_since_issue_d:49-52': range(49, 53),
    'mths_since_issue_d:53-64': range(53, 65),
    'mths_since_issue_d:65-84': range(65, 85),
    'mths_since_issue_d:>84': range(85, int(df_inputs_prepr['mths_since_issue_d'].max()) + 1)
}

# Loop through the dictionary to create new binary columns for mths_since_issue_d
for new_col, month_range in mths_since_issue_d_categories.items():
    df_inputs_prepr[new_col] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(month_range), 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

#Int_rate_factor
df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'int_rate_factor', df_targets_prepr)
df_temp
plot_by_woe(df_temp, 90)

int_rate_categories = {
    'int_rate:<9.548': (df_inputs_prepr['int_rate'] <= 9.548),
    'int_rate:9.548-12.025': (df_inputs_prepr['int_rate'] > 9.548) & (df_inputs_prepr['int_rate'] <= 12.025),
    'int_rate:12.025-15.74': (df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate'] <= 15.74),
    'int_rate:15.74-20.281': (df_inputs_prepr['int_rate'] > 15.74) & (df_inputs_prepr['int_rate'] <= 20.281),
    'int_rate:>20.281': (df_inputs_prepr['int_rate'] > 20.281)
}

# Loop through the dictionary to create new binary columns for int_rate
for new_col, condition in int_rate_categories.items():
    df_inputs_prepr[new_col] = np.where(condition, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

#Funded Amt Factor
df_inputs_prepr['funded_amnt_factor'] = pd.cut(df_inputs_prepr['funded_amnt'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'funded_amnt_factor', df_targets_prepr)
df_temp
plot_by_woe(df_temp, 90)
#No association with IV, do not need to use the funded_amt_factor in our PD Model

#mths_since_earliest_cr_line
df_inputs_prepr['mths_since_earliest_cr_line_factor'] = pd.cut(df_inputs_prepr['mths_since_earliest_cr_line'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'mths_since_earliest_cr_line_factor', df_targets_prepr)
df_temp
plot_by_woe(df_temp, 90)
plot_by_woe(df_temp.iloc[6: , : ], 90)
# < 140, # 141 - 164, # 165 - 247, # 248 - 270, # 271 - 352, # > 352 categories
mths_since_earliest_cr_line_categories = {
    'mths_since_earliest_cr_line:<140': range(140),
    'mths_since_earliest_cr_line:141-200': range(140, 201),
    'mths_since_earliest_cr_line:201-262': range(201, 263),  # Corrected range from 200-262 to 201-263
    'mths_since_earliest_cr_line:263-285': range(263, 286),  # Corrected range from 262-286 to 263-286
    'mths_since_earliest_cr_line:286-441': range(286, 442),  # Corrected range from 286-441 to 286-442
    'mths_since_earliest_cr_line:>441': range(442, int(df_inputs_prepr['mths_since_earliest_cr_line'].max()) + 1)  # Corrected range from 441 to 442
}

# Loop through the dictionary to create new binary columns for mths_since_earliest_cr_line
for new_col, month_range in mths_since_earliest_cr_line_categories.items():
    df_inputs_prepr[new_col] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(month_range), 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

# delinq_2yrs
df_temp = woe_ordered_continuous(df_inputs_prepr, 'delinq_2yrs', df_targets_prepr)
df_temp
plot_by_woe(df_temp)
# Categories: 0, 1-3, >=4
delinq_2yrs_categories = {
    'delinq_2yrs:0': df_inputs_prepr['delinq_2yrs'] == 0,
    'delinq_2yrs:1-3': (df_inputs_prepr['delinq_2yrs'] >= 1) & (df_inputs_prepr['delinq_2yrs'] <= 3),
    'delinq_2yrs:>=4': df_inputs_prepr['delinq_2yrs'] >= 4  # Corrected from >=9 to >=4
}

# Loop through the dictionary to create new binary columns for delinq_2yrs
for new_col, condition in delinq_2yrs_categories.items():
    df_inputs_prepr[new_col] = np.where(condition, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

# inq_last_6mths\
df_temp = woe_ordered_continuous(df_inputs_prepr, 'inq_last_6mths', df_targets_prepr)
df_temp
plot_by_woe(df_temp)
# Categories: 0, 1 - 2, 3 - 6, > 6
inq_last_6mths_categories = {
    'inq_last_6mths:0': df_inputs_prepr['inq_last_6mths'] == 0,
    'inq_last_6mths:1-2': (df_inputs_prepr['inq_last_6mths'] >= 1) & (df_inputs_prepr['inq_last_6mths'] <= 2),
    'inq_last_6mths:3-6': (df_inputs_prepr['inq_last_6mths'] >= 3) & (df_inputs_prepr['inq_last_6mths'] <= 6),
    'inq_last_6mths:>6': df_inputs_prepr['inq_last_6mths'] > 6
}

# Loop through the dictionary to create new binary columns for inq_last_6mths
for new_col, condition in inq_last_6mths_categories.items():
    df_inputs_prepr[new_col] = np.where(condition, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

# open_acc
df_temp = woe_ordered_continuous(df_inputs_prepr, 'open_acc', df_targets_prepr) 
df_temp
plot_by_woe(df_temp, 90)
plot_by_woe(df_temp.iloc[ : 40, :], 90)
# Categories: '0', '1-3', '4-12', '13-17', '18-22', '23-25', '26-30', '>30'
open_acc_categories = {
    'open_acc:0': df_inputs_prepr['open_acc'] == 0,
    'open_acc:1-3': (df_inputs_prepr['open_acc'] >= 1) & (df_inputs_prepr['open_acc'] <= 3),
    'open_acc:4-12': (df_inputs_prepr['open_acc'] >= 4) & (df_inputs_prepr['open_acc'] <= 12),
    'open_acc:13-17': (df_inputs_prepr['open_acc'] >= 13) & (df_inputs_prepr['open_acc'] <= 17),
    'open_acc:18-22': (df_inputs_prepr['open_acc'] >= 18) & (df_inputs_prepr['open_acc'] <= 22),
    'open_acc:23-25': (df_inputs_prepr['open_acc'] >= 23) & (df_inputs_prepr['open_acc'] <= 25),
    'open_acc:26-30': (df_inputs_prepr['open_acc'] >= 26) & (df_inputs_prepr['open_acc'] <= 30),
    'open_acc:>=31': df_inputs_prepr['open_acc'] >= 31
}

# Loop through the dictionary to create new binary columns for open_acc
for new_col, condition in open_acc_categories.items():
    df_inputs_prepr[new_col] = np.where(condition, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

# pub_rec
df_temp = woe_ordered_continuous(df_inputs_prepr, 'pub_rec', df_targets_prepr)
df_temp
plot_by_woe(df_temp, 90)
# Categories '0-2', '3-4', '>=5'
pub_rec_categories = {
    'pub_rec:0-2': (df_inputs_prepr['pub_rec'] >= 0) & (df_inputs_prepr['pub_rec'] <= 2),
    'pub_rec:3-4': (df_inputs_prepr['pub_rec'] >= 3) & (df_inputs_prepr['pub_rec'] <= 4),
    'pub_rec:>=5': df_inputs_prepr['pub_rec'] >= 5
}

# Loop through the dictionary to create new binary columns for pub_rec
for new_col, condition in pub_rec_categories.items():
    df_inputs_prepr[new_col] = np.where(condition, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

# total_acc
df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'total_acc_factor', df_targets_prepr)
df_temp
plot_by_woe(df_temp, 90)
# Categories: '<=27', '28-51', '>51'
total_acc_categories = {
    'total_acc:<=27': df_inputs_prepr['total_acc'] <= 27,
    'total_acc:28-51': (df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51),
    'total_acc:>=52': df_inputs_prepr['total_acc'] >= 52
}

# Loop through the dictionary to create new binary columns for total_acc
for new_col, condition in total_acc_categories.items():
    df_inputs_prepr[new_col] = np.where(condition, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()


# acc_now_delinq
df_temp = woe_ordered_continuous(df_inputs_prepr, 'acc_now_delinq', df_targets_prepr)
df_temp
plot_by_woe(df_temp)
# Categories: '0', '>=1'
acc_now_delinq_categories = {
    'acc_now_delinq:0': df_inputs_prepr['acc_now_delinq'] == 0,
    'acc_now_delinq:>=1': df_inputs_prepr['acc_now_delinq'] >= 1
}

# Loop through the dictionary to create new binary columns for acc_now_delinq
for new_col, condition in acc_now_delinq_categories.items():
    df_inputs_prepr[new_col] = np.where(condition, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

# total_rev_hi_lim
df_inputs_prepr['total_rev_hi_lim_factor'] = pd.cut(df_inputs_prepr['total_rev_hi_lim'], 2000)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'total_rev_hi_lim_factor', df_targets_prepr) 
df_temp
plot_by_woe(df_temp.iloc[: 50, : ], 90)
# Categories'<=5K', '5K-10K', '10K-20K', '20K-30K', '30K-40K', '40K-55K', '55K-95K', '>95K'\n",
total_rev_hi_lim_categories = {
    'total_rev_hi_lim:<=5K': df_inputs_prepr['total_rev_hi_lim'] <= 5000,
    'total_rev_hi_lim:5K-10K': (df_inputs_prepr['total_rev_hi_lim'] > 5000) & (df_inputs_prepr['total_rev_hi_lim'] <= 10000),
    'total_rev_hi_lim:10K-20K': (df_inputs_prepr['total_rev_hi_lim'] > 10000) & (df_inputs_prepr['total_rev_hi_lim'] <= 20000),
    'total_rev_hi_lim:20K-30K': (df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000),
    'total_rev_hi_lim:30K-40K': (df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000),
    'total_rev_hi_lim:40K-55K': (df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000),
    'total_rev_hi_lim:55K-95K': (df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000),
    'total_rev_hi_lim:>95K': df_inputs_prepr['total_rev_hi_lim'] > 95000
}

# Loop through the dictionary to create new binary columns for total_rev_hi_lim
for new_col, condition in total_rev_hi_lim_categories.items():
    df_inputs_prepr[new_col] = np.where(condition, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

# installment
df_inputs_prepr['installment_factor'] = pd.cut(df_inputs_prepr['installment'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'installment_factor', df_targets_prepr)
df_temp
plot_by_woe(df_temp, 90)

#More complicated annual income processing
df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 100)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_targets_prepr)
df_temp
#Because a majority of dataset is in first 2 cuts of data, we are creating a dummy variable for high income earners >140K
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000, : ].copy()
df_inputs_prepr_temp['annual_inc_factor'] = pd.cut(df_inputs_prepr_temp['annual_inc'], 50).copy()
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'annual_inc_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp
plot_by_woe(df_temp, 90)
#increasing steadily, can split variable into roughly equal variables considering # of obs
annual_inc_categories = {
    'annual_inc:<20K': df_inputs_prepr['annual_inc'] <= 20000,
    'annual_inc:20K-30K': (df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000),
    'annual_inc:30K-40K': (df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000),
    'annual_inc:40K-50K': (df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000),
    'annual_inc:50K-60K': (df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000),
    'annual_inc:60K-70K': (df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000),
    'annual_inc:70K-80K': (df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000),
    'annual_inc:80K-90K': (df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000),
    'annual_inc:90K-100K': (df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000),
    'annual_inc:100K-120K': (df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000),
    'annual_inc:120K-140K': (df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000),
    'annual_inc:>140K': df_inputs_prepr['annual_inc'] > 140000
}

# Loop through the dictionary to create new binary columns for annual_inc
for new_col, condition in annual_inc_categories.items():
    df_inputs_prepr[new_col] = np.where(condition, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

#mths_since_last_delinq
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_delinq'])].copy()
df_inputs_prepr_temp['mths_since_last_delinq_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_delinq'], 50).copy()
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_delinq_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp
plot_by_woe(df_temp, 90)
mths_since_last_delinq_categories = {
    'mths_since_last_delinq:Missing': df_inputs_prepr['mths_since_last_delinq'].isnull(),
    'mths_since_last_delinq:0-3': (df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3),
    'mths_since_last_delinq:4-30': (df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30),
    'mths_since_last_delinq:31-56': (df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56),
    'mths_since_last_delinq:>=57': df_inputs_prepr['mths_since_last_delinq'] >= 57
}

# Loop through the dictionary to create new binary columns for mths_since_last_delinq
for new_col, condition in mths_since_last_delinq_categories.items():
    df_inputs_prepr[new_col] = np.where(condition, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

#dti factor
df_inputs_prepr['dti_factor'] = pd.cut(df_inputs_prepr['dti'], 100)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'dti_factor', df_targets_prepr)
df_temp
plot_by_woe(df_temp, 90)
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['dti'] <= 35, : ].copy()
df_inputs_prepr_temp['dti_factor'] = pd.cut(df_inputs_prepr_temp['dti'], 50).copy()
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'dti_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp
plot_by_woe(df_temp, 90)

dti_categories = {
    'dti:<=1.4': df_inputs_prepr['dti'] <= 1.4,
    'dti:1.4-3.5': (df_inputs_prepr['dti'] > 1.4) & (df_inputs_prepr['dti'] <= 3.5),
    'dti:3.5-7.7': (df_inputs_prepr['dti'] > 3.5) & (df_inputs_prepr['dti'] <= 7.7),
    'dti:7.7-10.5': (df_inputs_prepr['dti'] > 7.7) & (df_inputs_prepr['dti'] <= 10.5),
    'dti:10.5-16.1': (df_inputs_prepr['dti'] > 10.5) & (df_inputs_prepr['dti'] <= 16.1),
    'dti:16.1-20.3': (df_inputs_prepr['dti'] > 16.1) & (df_inputs_prepr['dti'] <= 20.3),
    'dti:20.3-21.7': (df_inputs_prepr['dti'] > 20.3) & (df_inputs_prepr['dti'] <= 21.7),
    'dti:21.7-22.4': (df_inputs_prepr['dti'] > 21.7) & (df_inputs_prepr['dti'] <= 22.4),
    'dti:22.4-35': (df_inputs_prepr['dti'] > 22.4) & (df_inputs_prepr['dti'] <= 35),
    'dti:>35': df_inputs_prepr['dti'] > 35
}

# Loop through the dictionary to create new binary columns for dti
for new_col, condition in dti_categories.items():
    df_inputs_prepr[new_col] = np.where(condition, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

#Mths Since Last Record
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_record'])].copy()
df_inputs_prepr_temp['mths_since_last_record_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_record'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_record_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp
plot_by_woe(df_temp, 90)

mths_since_last_record_categories = {
    'mths_since_last_record:Missing': df_inputs_prepr['mths_since_last_record'].isnull(),
    'mths_since_last_record:0-2': (df_inputs_prepr['mths_since_last_record'] >= 0) & (df_inputs_prepr['mths_since_last_record'] <= 2),
    'mths_since_last_record:3-20': (df_inputs_prepr['mths_since_last_record'] >= 3) & (df_inputs_prepr['mths_since_last_record'] <= 20),
    'mths_since_last_record:21-31': (df_inputs_prepr['mths_since_last_record'] >= 21) & (df_inputs_prepr['mths_since_last_record'] <= 31),
    'mths_since_last_record:32-80': (df_inputs_prepr['mths_since_last_record'] >= 32) & (df_inputs_prepr['mths_since_last_record'] <= 80),
    'mths_since_last_record:81-86': (df_inputs_prepr['mths_since_last_record'] >= 81) & (df_inputs_prepr['mths_since_last_record'] <= 86),
    'mths_since_last_record:>=86': df_inputs_prepr['mths_since_last_record'] >= 86
}

# Loop through the dictionary to create new binary columns for mths_since_last_record
for new_col, condition in mths_since_last_record_categories.items():
    df_inputs_prepr[new_col] = np.where(condition, 1, 0)

# Display the updated DataFrame
df_inputs_prepr.head()

df_inputs_prepr.shape

#Run line 706 then go back up to line 128
#preprocessing the test dataset
#loan_data_inputs_train = df_inputs_prepr #comment out after running and input line below
loan_data_inputs_test = df_inputs_prepr

folder_path = r'C:\Users\mhack\OneDrive\Documents\Projects'

loan_data_inputs_train.to_csv(f'{folder_path}\\loan_data_inputs_train.csv')
loan_data_targets_train.to_csv(f'{folder_path}\\loan_data_targets_train.csv', header=None)
loan_data_inputs_test.to_csv(f'{folder_path}\\loan_data_inputs_test.csv')
loan_data_targets_test.to_csv(f'{folder_path}\\loan_data_targets_test.csv', header=None)


loan_data_inputs_train = pd.read_csv('C:/Users/mhack/OneDrive/Documents/Projects/loan_data_inputs_train.csv', index_col = 0)
loan_data_targets_train = pd.read_csv('C:/Users/mhack/OneDrive/Documents/Projects/loan_data_targets_train.csv', index_col = 0, header = None)
loan_data_inputs_test = pd.read_csv('C:/Users/mhack/OneDrive/Documents/Projects/loan_data_inputs_test.csv', index_col = 0)
loan_data_targets_test = pd.read_csv('C:/Users/mhack/OneDrive/Documents/Projects/loan_data_targets_test.csv', index_col = 0, header = None)

loan_data_inputs_train.head()
loan_data_targets_train.head()
loan_data_inputs_train.shape
loan_data_targets_train.shape
loan_data_inputs_test.shape
loan_data_targets_test.shape


#selecting the features we need
inputs_train_with_ref_cat = loan_data_inputs_train.loc[: , ['grade:A','grade:B','grade:C','grade:D','grade:E','grade:F','grade:G',
'home_ownership:RENT_OTHER_NONE_ANY','home_ownership:OWN','home_ownership:MORTGAGE','addr_state:ND_NE_IA_NV_FL_HI_AL','addr_state:NM_VA',
'addr_state:NY','addr_state:OK_TN_MO_LA_MD_NC','addr_state:CA','addr_state:UT_KY_AZ_NJ','addr_state:AR_MI_PA_OH_MN','addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR','addr_state:WI_MT','addr_state:TX','addr_state:IL_CT','addr_state:KS_SC_CO_VT_AK_MS','addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified','verification_status:Source Verified','verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house','purpose:credit_card','purpose:debt_consolidation','purpose:oth__med__vacation','purpose:major_purch__car__home_impr',
'initial_list_status:f','initial_list_status:w',
'term:36','term:60',
'emp_length:0','emp_length:1','emp_length:2-4','emp_length:5-6','emp_length:7-9','emp_length:10',
'mths_since_issue_d:<38','mths_since_issue_d:38-39','mths_since_issue_d:40-41','mths_since_issue_d:42-48','mths_since_issue_d:49-52',
'mths_since_issue_d:53-64','mths_since_issue_d:65-84','mths_since_issue_d:>84',
'int_rate:<9.548','int_rate:9.548-12.025','int_rate:12.025-15.74','int_rate:15.74-20.281','int_rate:>20.281',
'mths_since_earliest_cr_line:<140','mths_since_earliest_cr_line:141-200','mths_since_earliest_cr_line:201-262','mths_since_earliest_cr_line:263-285',
'mths_since_earliest_cr_line:286-441','mths_since_earliest_cr_line:>441',
'delinq_2yrs:0','delinq_2yrs:1-3','delinq_2yrs:>=4',
'inq_last_6mths:0','inq_last_6mths:1-2','inq_last_6mths:3-6','inq_last_6mths:>6',
'open_acc:0','open_acc:1-3','open_acc:4-12','open_acc:13-17','open_acc:18-22','open_acc:23-25','open_acc:26-30','open_acc:>=31',
'pub_rec:0-2','pub_rec:3-4','pub_rec:>=5',
'total_acc:<=27','total_acc:28-51','total_acc:>=52',
'acc_now_delinq:0','acc_now_delinq:>=1',
'total_rev_hi_lim:<=5K','total_rev_hi_lim:5K-10K','total_rev_hi_lim:10K-20K','total_rev_hi_lim:20K-30K','total_rev_hi_lim:30K-40K',
'total_rev_hi_lim:40K-55K','total_rev_hi_lim:55K-95K','total_rev_hi_lim:>95K',
'annual_inc:<20K','annual_inc:20K-30K','annual_inc:30K-40K','annual_inc:40K-50K','annual_inc:50K-60K','annual_inc:60K-70K',
'annual_inc:70K-80K','annual_inc:80K-90K','annual_inc:90K-100K','annual_inc:100K-120K','annual_inc:120K-140K','annual_inc:>140K',
'dti:<=1.4','dti:1.4-3.5','dti:3.5-7.7','dti:7.7-10.5','dti:10.5-16.1','dti:16.1-20.3','dti:20.3-21.7','dti:21.7-22.4','dti:22.4-35','dti:>35',
'mths_since_last_delinq:Missing','mths_since_last_delinq:0-3','mths_since_last_delinq:4-30','mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57','mths_since_last_record:Missing',
'mths_since_last_record:0-2','mths_since_last_record:3-20','mths_since_last_record:21-31','mths_since_last_record:32-80',
'mths_since_last_record:81-86','mths_since_last_record:>=86']]

#reference categories to reduce our categories variables to k-1
ref_categories = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'delinq_2yrs:>=4',
'inq_last_6mths:>6',
'open_acc:0',
'pub_rec:0-2',
'total_acc:<=27',
'acc_now_delinq:0',
'total_rev_hi_lim:<=5K',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']

inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis = 1)
inputs_train.head()

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

reg = LogisticRegression(max_iter=5000)
pd.options.display.max_rows = None

reg.fit(inputs_train, loan_data_targets_train)
reg.intercept_
reg.coef_

#format the results
feature_name = inputs_train.columns.values
summary_table = pd.DataFrame(columns = ['Feature Name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table

from sklearn import linear_model
import scipy.stats as stat

class LogisticRegression_with_p_values:
    def __init__(self, *args, **kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X / denom).T, X)
        
        # Ensure F_ij is of type float64
        F_ij = F_ij.astype(np.float64)
        
        # Compute the inverse of F_ij in a numerically stable way
        try:
            Cramer_Rao = np.linalg.inv(F_ij)
        except np.linalg.LinAlgError:
            Cramer_Rao = np.linalg.pinv(F_ij)  # Use pseudo-inverse if regular inverse fails
        
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values

reg = LogisticRegression_with_p_values(max_iter=5000)
reg.fit(inputs_train, loan_data_targets_train)
#format the results
feature_name = inputs_train.columns.values
summary_table = pd.DataFrame(columns = ['Feature Name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table

p_values = reg.p_values
p_values = np.append(np.nan, np.array(p_values))
summary_table['p_values'] = p_values
summary_table


inputs_train_with_ref_cat2 = loan_data_inputs_train.loc[: , ['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-200',
'mths_since_earliest_cr_line:201-262',
'mths_since_earliest_cr_line:263-285',
'mths_since_earliest_cr_line:286-441',
'mths_since_earliest_cr_line:>441',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31',
'mths_since_last_record:32-80',
'mths_since_last_record:81-86',
'mths_since_last_record:>=86']]


ref_categories2 = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'inq_last_6mths:>6',
'acc_now_delinq:0',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']


inputs_train2 = inputs_train_with_ref_cat2.drop(ref_categories2, axis = 1)
inputs_train2.head()

reg2 = LogisticRegression_with_p_values(max_iter=5000)
reg2.fit(inputs_train2, loan_data_targets_train)
feature_name = inputs_train2.columns.values
summary_table = pd.DataFrame(columns = ['Feature Name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(reg2.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg2.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table

p_values2 = reg2.p_values
p_values2 = np.append(np.nan, np.array(p_values2))
summary_table['p_values'] = p_values2
summary_table

summary_table.to_csv(f'{folder_path}\\summary_table_crm_pd_model.csv')
