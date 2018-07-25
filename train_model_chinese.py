from keras.layers import Input,Dense,Embedding,Conv2D,MaxPool2D
from keras.layers import Reshape,Flatten,Dropout,Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers_chinese import load_data

print("正在加载数据....")
x,y,vocabulary,vocabulary_inv=load_data()

# x.shape -> (54568, 50)
# y.shape -> (54568, 2)
# len(vocabulary) -> 52822
# len(vocabulary_inv) -> 52822

# 划分数据集
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
# X_train.shape -> (43654, 50)
# y_train.shape -> (43654, 2)
# X_test.shape -> (10914, 50)
# y_test.shape -> (10914, 2)


sequence_length=X_train.shape[1]
vocabulary_size=len(vocabulary_inv)
embedding_dim=256
filter_sizes=[3,4,5]
num_filters=512
drop=0.5

epochs=100
batch_size=30

# 创建tensor
print("正在创建模型...")
inputs=Input(shape=(sequence_length,),dtype='int32')
embedding=Embedding(input_dim=vocabulary_size,output_dim=embedding_dim,input_length=sequence_length)(inputs)
reshape=Reshape((sequence_length,embedding_dim,1))(embedding)

# cnn
conv_0=Conv2D(num_filters,kernel_size=(filter_sizes[0],embedding_dim),padding='valid',kernel_initializer='normal',activation='relu')(reshape)
conv_1=Conv2D(num_filters,kernel_size=(filter_sizes[1],embedding_dim),padding='valid',kernel_initializer='normal',activation='relu')(reshape)
conv_2=Conv2D(num_filters,kernel_size=(filter_sizes[2],embedding_dim),padding='valid',kernel_initializer='normal',activation='relu')(reshape)

maxpool_0=MaxPool2D(pool_size=(sequence_length-filter_sizes[0]+1,1),strides=(1,1),padding='valid')(conv_0)
maxpool_1=MaxPool2D(pool_size=(sequence_length-filter_sizes[1]+1,1),strides=(1,1),padding='valid')(conv_1)
maxpool_2=MaxPool2D(pool_size=(sequence_length-filter_sizes[2]+1,1),strides=(1,1),padding='valid')(conv_2)


concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=2, activation='softmax')(dropout)
model=Model(inputs=inputs,outputs=output)


checkpoint = ModelCheckpoint('results/chinese.weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training

