from keras.models import load_model
from data_helpers import build_predict_input



model = load_model('model/weights.007-0.7618.hdf5')
def predict():
    while True:
        input_x = input("请输入预测的句子:\n")
        if input_x == 'exit':
            exit()

        input_x = build_predict_input(input_x)
        print(input_x.shape)
        y_pred = model.predict(input_x)
        result=list(y_pred[0])
        if result[1]>result[0]:
            print('positive:', result[1])
        else:
            print('negative:', result[0])

if __name__ == '__main__':
    predict()