import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np
import json

app=Flask(__name__)
## Load the Model
model=pickle.load(open("./pickle Files\model.pkl",'rb'))
scaler=pickle.load(open("./pickle Files\scaler.pkl",'rb'))
ohe=pickle.load(open("./pickle Files\ohe.pkl",'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    #data = json.loads(request.get_data())
    input_data=request.json["data"]
    print(input_data)
    # convert the input data into a pandas DataFrame
    df = pd.DataFrame(input_data, index=[0])
    # select numeric columns
    numeric_df = df.select_dtypes(include=['int', 'float'])
    # select categorical columns
    categorical_df = df.select_dtypes(include=['object', 'category'])

    def perform_standardization_one_df(test_df, scaler):
        #scaler = StandardScaler()
        num_attr = test_df.select_dtypes(['int','float']).columns
        print(num_attr)
        test_df[num_attr]=scaler.transform(test_df[num_attr])
        return None

    perform_standardization_one_df(numeric_df, scaler)

    def perform_one_hot_encoding_one_df(X_test,cols,ohe):
        # ohe = OneHotEncoder(drop='first',sparse=False,dtype=np.int32)
        X_test_new = ohe.transform(X_test[cols])
        # Print Transformed
        print("#### Printing Transformed Records #### \n")
        print(X_test_new[:],'\n')
        for i in range(len(ohe.categories_)):
            if i == 0:
                label = ohe.categories_[i]
                label_combined = label[1:]
            elif i > 0:
                label = ohe.categories_[i]
                label_combined = np.append(label_combined,label[1:])
        print("Labels: \n")
        print(label_combined, " \n")
        # Adding Transformed X_test back to the main DataFrame
        X_test[label_combined] = X_test_new
        # Dropping the Encoded Column
        print("#### Dropping Encoded Column in Test ####" '\n')
        X_test.drop(cols,axis=1,inplace=True)
        print("#### Test Columns After Dropping ####" '\n')
        # print(X_test_new.columns,'\n')
        return None

    cols = ['sex','smoker','region']
    perform_one_hot_encoding_one_df(categorical_df,cols,ohe)

    # Combine numerical data and categorical data
    def combine_num_df_cat_df(num_df, cat_df):
        print(num_df.head())
        print(cat_df.head())
        result = pd.concat([num_df,cat_df],axis=1) # Using concat funtion in pandas we join the numerical columns and categorical columns
        print(result.head()) # Printing the head of combined dataset
        return result

    final_df = combine_num_df_cat_df(numeric_df, categorical_df)

    # final_list = final_df.values.tolist()

    new_data = final_df.values.reshape(1,-1)
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    df = pd.DataFrame(form_data, index=[0])
    # Set appropriate datatype for each column
    dtype_dict = {'age': int, 'sex': object, 'bmi': float, 'children': int, 'smoker':object, 'region':object}
    df = df.astype(dtype_dict)
    print(df.head())
    print(df.dtypes)

    def get_num_cat_dataframes(DataFrame):
        num_df = DataFrame.select_dtypes(include=['int','float']) # Assigning the columns which are of 'int' & 'float' type to num_df
        print(num_df.columns)
        cat_df = DataFrame.select_dtypes(exclude=['int','float']) # Assigning the columns which are of 'category' type to cat_df
        print(cat_df.columns)
        return num_df, cat_df

    numeric_df, categorical_df = get_num_cat_dataframes(df)

    def perform_standardization_one_df(test_df, scaler):
        #scaler = StandardScaler()
        num_attr = test_df.select_dtypes(['int','float']).columns
        print("num attr",num_attr)
        test_df[num_attr]=scaler.transform(test_df[num_attr])
        return None

    perform_standardization_one_df(numeric_df, scaler)

    print("Done")

    def perform_one_hot_encoding_one_df(X_test,cols,ohe):
        # ohe = OneHotEncoder(drop='first',sparse=False,dtype=np.int32)
        X_test_new = ohe.transform(X_test[cols])
        # Print Transformed
        print("#### Printing Transformed Records #### \n")
        print(X_test_new[:],'\n')
        for i in range(len(ohe.categories_)):
            if i == 0:
                label = ohe.categories_[i]
                label_combined = label[1:]
            elif i > 0:
                label = ohe.categories_[i]
                label_combined = np.append(label_combined,label[1:])
        print("Labels: \n")
        print(label_combined, " \n")
        # Adding Transformed X_test back to the main DataFrame
        X_test[label_combined] = X_test_new
        # Dropping the Encoded Column
        print("#### Dropping Encoded Column in Test ####" '\n')
        X_test.drop(cols,axis=1,inplace=True)
        print("#### Test Columns After Dropping ####" '\n')
        # print(X_test_new.columns,'\n')
        return None

    cols = ['sex','smoker','region']
    perform_one_hot_encoding_one_df(categorical_df,cols,ohe)

    # Combine numerical data and categorical data
    def combine_num_df_cat_df(num_df, cat_df):
        print(num_df.head())
        print(cat_df.head())
        result = pd.concat([num_df,cat_df],axis=1) # Using concat funtion in pandas we join the numerical columns and categorical columns
        print(result.head()) # Printing the head of combined dataset
        return result

    final_df = combine_num_df_cat_df(numeric_df, categorical_df)

    # final_list = final_df.values.tolist()

    new_data = final_df.values.reshape(1,-1)
    output = model.predict(new_data)
    return render_template("home.html",prediction_text="The House Price Prediction is : {}".format(output[0]))

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
