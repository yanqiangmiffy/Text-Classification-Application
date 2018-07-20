from keras.models import load_model
from data_helpers_chinese import build_input_chinese
from data_helpers_english import build_input_english

def predict_chinese(model):
    while True:
        input_x = input("请输入预测的句子:\n")
        if input_x == 'exit':
            exit()

        input_x = build_input_chinese(input_x)
        print(input_x.shape)
        y_pred = model.predict(input_x)
        result=list(y_pred[0])
        if result[1]>result[0]:
            print('positive:', result[1])
        else:
            print('negative:', result[0])

def predict_english(model):
    while True:
        input_x = input("请输入预测的句子:\n")
        if input_x == 'exit':
            exit()

        input_x = build_input_english(input_x)
        print(input_x.shape)
        y_pred = model.predict(input_x)
        result=list(y_pred[0])
        if result[1]>result[0]:
            print('positive:', result[1])
        else:
            print('negative:', result[0])

if __name__ == '__main__':
    english_model = load_model('results/weights.007-0.7618.hdf5')
    predict_english(english_model)
    # chinese_model = load_model('results/chinese.weights.003-0.9083.hdf5')
    # predict_chinese(chinese_model)