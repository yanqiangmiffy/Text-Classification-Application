import numpy as np
from flask import Flask,render_template,request,redirect
from keras.models import load_model
from data_helpers import build_predict_input

app=Flask(__name__)


en_model = load_model('model/weights.007-0.7618.hdf5')
# load 进来模型紧接着就执行一次 predict 函数
print('test model...')
print(en_model.predict(np.zeros((1, 56))))
print('test done.')

def en_predict(input_x):
    sentence=input_x
    input_x = build_predict_input(input_x)
    y_pred = en_model.predict(input_x)
    result = list(y_pred[0])
    result={'sentence':sentence,'positive':result[1],'negative':result[0]}
    return result

@app.route('/english',methods=['POST','GET'])
def english():
    if request.method=='POST':
        review=request.form['review']
        result=en_predict(review)
        return render_template('index.html',result=result)
    return render_template('index.html')





if __name__ == '__main__':
    app.run(debug=True)