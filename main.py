import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import re
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from data_helpers_english import build_input_english
from data_helpers_chinese import build_input_chinese

app = Flask(__name__)


en_model = load_model('results/weights.007-0.7618.hdf5')
ch_model = load_model('results/chinese.weights.003-0.9083.hdf5')
# load 进来模型紧接着就执行一次 predict 函数
print('test train...')
print(en_model.predict(np.zeros((1, 56))))
print(ch_model.predict(np.zeros((1, 50))))
print('test done.')

def en_predict(input_x):
    sentence = input_x
    input_x = build_input_english(input_x)
    y_pred = en_model.predict(input_x)
    result = list(y_pred[0])
    result = {'sentence': sentence, 'positive': result[1], 'negative': result[0]}
    return result

def ch_predict(input_x):
    sentence = input_x
    input_x = build_input_chinese(input_x)
    y_pred = ch_model.predict(input_x)
    result = list(y_pred[0])
    result = {'sentence': sentence, 'positive': result[1], 'negative': result[0]}
    return result

@app.route('/classification', methods=['POST', 'GET'])
def english():
    if request.method == 'POST':
        review = request.form['review']
        # 来判断是中文句子/还是英文句子
        review_flag = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", review)  # 去除数字
        review_flag = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", review_flag)
        if review_flag:
            result = en_predict(review)
            # result = {'sentence': 'hello', 'positive': '03.87878', 'negative': '03.64465'}
            return render_template('index.html', result=result)
        else:
            result = ch_predict(review)
            # result = {'sentence': 'hello', 'positive': '03.87878', 'negative': '03.64465'}
            return render_template('index.html', result=result)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
