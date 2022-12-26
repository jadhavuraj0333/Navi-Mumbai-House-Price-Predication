
#Import Libraries
from flask import Flask, request, render_template
import pickle
import numpy as np

svrmodel=pickle.load(open('svrmodel.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
app = Flask(__name__)
 
# render htmp page
@app.route('/')
def home():
    return render_template('index.html')
 
# get user input and the predict the output and return to user
@app.route('/predict',methods=['POST'])
def predict():
     
    #take data from form and store in each feature    
    input_features = [x for x in request.form.values()]
    new_list=[]
    c=0
    Loc=input_features[13]
    for i in (input_features[0:14]):
        c+=1
        if c<14:
          new_list.append(int(i))
          print(i)
        
        else:
           for i in range (0,23):
               new_list.append(0)
    new_list[3]=(np.sqrt(new_list[3]))
    new_list[4]=(np.log(new_list[4]))
    new_list[5]=(np.sqrt(new_list[5]))
    allfeatures=['Bedrooms', 'Transaction_Type', 'New_Furnishing',
       'sqrt_PropertyOnFloor', 'Log_Totals_Floors', 'sqrt_Carpet_Area', 'Lift',
       'Cover Parking', 'Security', 'GYM', 'Swimming Pool', 'Park',
       'Club House', 'airoli', 'belapur', 'dronagiri',
       'ghansoli', 'juinagar', 'kalamboli', 'kamothe',
       'karanjade', 'khanda', 'khandeshwar', 'kharghar',
       'koparkhairane', 'navade', 'nerul', 'panvel',
       'rabale', 'roadpali', 'sanpada', 'seawoods',
       'taloja', 'turbhe', 'ulwe', 'vashi']
    index = allfeatures.index(Loc)
    new_list[index]=1
    final_input=scaler.transform(np.array(new_list).reshape(1,-1))
    output=svrmodel.predict(final_input)[0]
    pred_result=np.exp(output)
    final_pred_result=round(pred_result/100000,2)


    
    # render the html page and show the output
    return render_template ('index.html',prediction_text='Predicted Price of  House is {} {}'.format(final_pred_result,"Lakhs"))
 
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port="8080")
     
if __name__ == "__main__":
    app.run(debug=True)
    