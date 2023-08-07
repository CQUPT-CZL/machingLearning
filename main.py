import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 假设我们有一些时间序列数据，输入序列为10个时间步长，每个时间步长有1个特征
train_data = np.random.randn(100, 10, 1)
train_labels = np.random.randn(100,)

# 创建LSTM模型
model = Sequential()
model.add(LSTM(64, input_shape=(10, 1)))  # 64是LSTM单元数，input_shape是输入数据形状
model.add(Dense(1))  # 输出层，无激活函数

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(train_data, train_labels, epochs=10, batch_size=32)

# 使用模型进行预测
test_data = np.random.randn(10, 10, 1)
predictions = model.predict(test_data)
print(predictions)
