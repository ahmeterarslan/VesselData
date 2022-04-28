
"""
Summary:
    
    I developed two different models which were DecisionTreeRegressor and XGBoost.
    The reason why I preferred is data includes lots of categorical variable
    I used MSE as evaluation metric
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xg


# read the data

data = pd.read_excel("VesselData.xlsx")




# sort data according to ata and convert all object type data to categorical with numeric values

ordered_data = data.sort_values("ata").copy()
categorical = ordered_data.copy()

categorical.dtypes
cat_columns = categorical.select_dtypes(['object']).columns
categorical[cat_columns] = categorical[cat_columns].astype('category')
categorical[cat_columns] = categorical[cat_columns].apply(lambda x: x.cat.codes)

# train test split

train, test = train_test_split(categorical,shuffle=False,test_size = 0.2)


# the data includes lots of binary and categorical variables therefore I first check DT regressor
# for discharge3

X_discharge3 = train.copy()
del X_discharge3["discharge3"]
del X_discharge3["eta"]
del X_discharge3["ata"]
del X_discharge3["atd"]

X_discharge3["hasnohamis"].unique()
del X_discharge3["hasnohamis"]

X_discharge3 = X_discharge3.fillna(0)

Y_discharge3 = train["discharge3"].copy()


X_discharge3_test = test.copy()
del X_discharge3_test["discharge3"]
del X_discharge3_test["eta"]
del X_discharge3_test["ata"]
del X_discharge3_test["atd"]

X_discharge3_test["hasnohamis"].unique() #All nan
del X_discharge3_test["hasnohamis"]

X_discharge3_test = X_discharge3_test.fillna(0)

Y_discharge3_test = test["discharge3"].copy()

regressor = DecisionTreeRegressor(random_state=2,max_depth=1000,min_samples_split=10,min_impurity_decrease=0.01)
regressor.fit(X_discharge3, Y_discharge3)
y_pred = regressor.predict(X_discharge3)

mean_squared_error(Y_discharge3,y_pred)

y_pred_test = model.predict(X_discharge3_test)

mean_squared_error(Y_discharge3_test,y_pred_test)

#XGboost

model = xg.XGBRegressor(max_depth=10)
model.fit(X_discharge3, Y_discharge3)
y_pred_xg = model.predict(X_discharge3)
mean_squared_error(Y_discharge3,y_pred_xg)
y_pred_test_xg = model.predict(X_discharge3_test)

mean_squared_error(Y_discharge3_test,y_pred_test_xg)

# I found lower mse value for test data of DT for the prediction of discharge3
# I preferred use DT Reg.

# discharge1


X_discharge1 = train.copy()
del X_discharge1["discharge1"]
del X_discharge1["eta"]
del X_discharge1["ata"]
del X_discharge1["atd"]

X_discharge1["hasnohamis"].unique()
del X_discharge1["hasnohamis"]

X_discharge1 = X_discharge1.fillna(0)

Y_discharge1 = train["discharge1"].copy()


X_discharge1_test = test.copy()
del X_discharge1_test["discharge1"]
del X_discharge1_test["eta"]
del X_discharge1_test["ata"]
del X_discharge1_test["atd"]

X_discharge1_test["hasnohamis"].unique() #All nan
del X_discharge1_test["hasnohamis"]

X_discharge1_test = X_discharge1_test.fillna(0)

Y_discharge1_test = test["discharge1"].copy()

#model

regressor = DecisionTreeRegressor(random_state=2,max_depth=1000,min_samples_split=3,min_impurity_decrease=0.01)
regressor.fit(X_discharge1, Y_discharge1)
y_pred = regressor.predict(X_discharge1)

mean_squared_error(Y_discharge1,y_pred)

y_pred_test = model.predict(X_discharge1_test)

mean_squared_error(Y_discharge1_test,y_pred_test)


# discharge2


X_discharge2 = train.copy()
del X_discharge2["discharge2"]
del X_discharge2["eta"]
del X_discharge2["ata"]
del X_discharge2["atd"]

X_discharge2["hasnohamis"].unique()
del X_discharge2["hasnohamis"]

X_discharge2 = X_discharge2.fillna(0)

Y_discharge2 = train["discharge2"].copy()


X_discharge2_test = test.copy()
del X_discharge2_test["discharge2"]
del X_discharge2_test["eta"]
del X_discharge2_test["ata"]
del X_discharge2_test["atd"]

X_discharge2_test["hasnohamis"].unique() #All nan
del X_discharge2_test["hasnohamis"]

X_discharge2_test = X_discharge2_test.fillna(0)

Y_discharge2_test = test["discharge2"].copy()

#model

regressor = DecisionTreeRegressor(random_state=2,max_depth=1000,min_samples_split=10,min_impurity_decrease=0.01)
regressor.fit(X_discharge2, Y_discharge2)
y_pred = regressor.predict(X_discharge2)

mean_squared_error(Y_discharge2,y_pred)

y_pred_test = model.predict(X_discharge2_test)

mean_squared_error(Y_discharge2_test,y_pred_test)


# discharge4


X_discharge4 = train.copy()
del X_discharge4["discharge4"]
del X_discharge4["eta"]
del X_discharge4["ata"]
del X_discharge4["atd"]

X_discharge4["hasnohamis"].unique()
del X_discharge4["hasnohamis"]

X_discharge4 = X_discharge4.fillna(0)

Y_discharge4 = train["discharge4"].copy()


X_discharge4_test = test.copy()
del X_discharge4_test["discharge4"]
del X_discharge4_test["eta"]
del X_discharge4_test["ata"]
del X_discharge4_test["atd"]

X_discharge4_test["hasnohamis"].unique() #All nan
del X_discharge4_test["hasnohamis"]

X_discharge4_test = X_discharge4_test.fillna(0)

Y_discharge4_test = test["discharge4"].copy()

#model

regressor = DecisionTreeRegressor(random_state=2,max_depth=1000,min_samples_split=10,min_impurity_decrease=0.01)
regressor.fit(X_discharge4, Y_discharge4)
y_pred = regressor.predict(X_discharge4)

mean_squared_error(Y_discharge4,y_pred)

y_pred_test = model.predict(X_discharge4_test)

mean_squared_error(Y_discharge4_test,y_pred_test)



# load1


X_load1 = train.copy()
del X_load1["load1"]
del X_load1["eta"]
del X_load1["ata"]
del X_load1["atd"]

X_load1["hasnohamis"].unique()
del X_load1["hasnohamis"]

X_load1 = X_load1.fillna(0)

Y_load1 = train["load1"].copy()


X_load1_test = test.copy()
del X_load1_test["load1"]
del X_load1_test["eta"]
del X_load1_test["ata"]
del X_load1_test["atd"]

X_load1_test["hasnohamis"].unique() #All nan
del X_load1_test["hasnohamis"]

X_load1_test = X_load1_test.fillna(0)

Y_load1_test = test["load1"].copy()

#model

regressor = DecisionTreeRegressor(random_state=2,max_depth=1000,min_samples_split=10,min_impurity_decrease=0.01)
regressor.fit(X_load1, Y_load1)
y_pred = regressor.predict(X_load1)

mean_squared_error(Y_load1,y_pred)

y_pred_test = model.predict(X_load1_test)

mean_squared_error(Y_load1_test,y_pred_test)



# load2


X_load2 = train.copy()
del X_load2["load2"]
del X_load2["eta"]
del X_load2["ata"]
del X_load2["atd"]

X_load2["hasnohamis"].unique()
del X_load2["hasnohamis"]

X_load2 = X_load2.fillna(0)

Y_load2 = train["load2"].copy()


X_load2_test = test.copy()
del X_load2_test["load2"]
del X_load2_test["eta"]
del X_load2_test["ata"]
del X_load2_test["atd"]

X_load2_test["hasnohamis"].unique() #All nan
del X_load2_test["hasnohamis"]

X_load2_test = X_load2_test.fillna(0)

Y_load2_test = test["load2"].copy()

#model

regressor = DecisionTreeRegressor(random_state=2,max_depth=1000,min_samples_split=10,min_impurity_decrease=0.01)
regressor.fit(X_load2, Y_load2)
y_pred = regressor.predict(X_load2)

mean_squared_error(Y_load2,y_pred)

y_pred_test = model.predict(X_load2_test)

mean_squared_error(Y_load2_test,y_pred_test)



# load3


X_load3 = train.copy()
del X_load3["load3"]
del X_load3["eta"]
del X_load3["ata"]
del X_load3["atd"]

X_load3["hasnohamis"].unique()
del X_load3["hasnohamis"]

X_load3 = X_load3.fillna(0)

Y_load3 = train["load3"].copy()


X_load3_test = test.copy()
del X_load3_test["load3"]
del X_load3_test["eta"]
del X_load3_test["ata"]
del X_load3_test["atd"]

X_load3_test["hasnohamis"].unique() #All nan
del X_load3_test["hasnohamis"]

X_load3_test = X_load3_test.fillna(0)

Y_load3_test = test["load3"].copy()

#model

regressor = DecisionTreeRegressor(random_state=2,max_depth=1000,min_samples_split=10,min_impurity_decrease=0.01)
regressor.fit(X_load3, Y_load3)
y_pred = regressor.predict(X_load3)

mean_squared_error(Y_load3,y_pred)

y_pred_test = model.predict(X_load3_test)

mean_squared_error(Y_load3_test,y_pred_test)



# load4


X_load4 = train.copy()
del X_load4["load4"]
del X_load4["eta"]
del X_load4["ata"]
del X_load4["atd"]

X_load4["hasnohamis"].unique()
del X_load4["hasnohamis"]

X_load4 = X_load4.fillna(0)

Y_load4 = train["load4"].copy()


X_load4_test = test.copy()
del X_load4_test["load4"]
del X_load4_test["eta"]
del X_load4_test["ata"]
del X_load4_test["atd"]

X_load4_test["hasnohamis"].unique() #All nan
del X_load4_test["hasnohamis"]

X_load4_test = X_load4_test.fillna(0)

Y_load4_test = test["load4"].copy()

#model

regressor = DecisionTreeRegressor(random_state=2,max_depth=1000,min_samples_split=10,min_impurity_decrease=0.01)
regressor.fit(X_load4, Y_load4)
y_pred = regressor.predict(X_load4)

mean_squared_error(Y_load4,y_pred)

y_pred_test = model.predict(X_load4_test)

mean_squared_error(Y_load4_test,y_pred_test)