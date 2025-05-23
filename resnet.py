import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
import shap
import seaborn as sns


seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)


data = pd.read_excel('data.xlsx', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[0:1213, -1].values

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

X_scaled = X_s[0:1213]

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3,
                                                    random_state=seed_value)  
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=seed_value)  # 分别20%


def resnet_block(x, filters, dropout_rate=0.5):
    shortcut = x

    x = layers.Conv1D(filters, kernel_size=3, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)  

    x = layers.Conv1D(filters, kernel_size=3, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    # 加入shortcut
    x = layers.add([x, shortcut])
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def build_improved_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)


    x = layers.Conv1D(64, kernel_size=7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # 第一组残差块 
    for _ in range(2):  
        x = resnet_block(x, 64, dropout_rate=0.5)  # 增加Dropout


    x = layers.Conv1D(128, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    
    for _ in range(2):  
        x = resnet_block(x, 128, dropout_rate=0.5)

   
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

num_classes = len(np.unique(y))
model = build_improved_resnet((X_train.shape[1], 1), num_classes)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

history = model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train,
                    epochs=200,
                    batch_size=32,
                    validation_data=(X_val.reshape(X_val.shape[0], X_val.shape[1], 1), y_val),
                    callbacks=[early_stopping, reduce_lr])

loss, accuracy = model.evaluate(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test)
print(f'Test Accuracy: {accuracy:.2f}')

y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

y_test_one_hot = pd.get_dummies(y_test).values
auc = roc_auc_score(y_test_one_hot, y_pred, multi_class='ovr')
print(f'AUC: {auc:.2f}')

plt.rcParams['font.family'] = 'Times New Roman'
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Greens)
plt.savefig('confusion_matrixxiuceshi.png', dpi=300, bbox_inches='tight')
plt.show()

print(classification_report(y_test, y_pred_classes))

val_loss, val_accuracy = model.evaluate(X_val.reshape(X_val.shape[0], X_val.shape[1], 1), y_val)
print(f'Validation Accuracy: {val_accuracy:.2f}')

y_val_pred = model.predict(X_val.reshape(X_val.shape[0], X_val.shape[1], 1))
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

val_accuracy = accuracy_score(y_val, y_val_pred_classes)
val_precision = precision_score(y_val, y_val_pred_classes, average='weighted')
val_recall = recall_score(y_val, y_val_pred_classes, average='weighted')
val_f1 = f1_score(y_val, y_val_pred_classes, average='weighted')

print(f'Validation Accuracy: {val_accuracy:.2f}')
print(f'Validation Precision: {val_precision:.2f}')
print(f'Validation Recall: {val_recall:.2f}')
print(f'Validation F1 Score: {val_f1:.2f}')

y_val_one_hot = pd.get_dummies(y_val).values
val_auc = roc_auc_score(y_val_one_hot, y_val_pred, multi_class='ovr')
print(f'Validation AUC: {val_auc:.2f}')

K = 100 
background_data = shap.kmeans(X_train.reshape(X_train.shape[0], X_train.shape[1]), K)

explainer = shap.KernelExplainer(lambda x: model.predict(x.reshape(x.shape[0], x.shape[1], 1)), background_data)

shap_values = explainer.shap_values(X_test.reshape(X_test.shape[0], X_test.shape[1]))

shap.summary_plot(shap_values, X_test.reshape(X_test.shape[0], X_test.shape[1]), plot_type="bar")

class_index = 2  
shap_values_class_2 = shap_values[class_index]

shap.summary_plot(shap_values_class_2, X_test.reshape(X_test.shape[0], X_test.shape[1]), plot_type="bar")

class_index = 1  
shap_values_class_2 = shap_values[class_index]

shap.summary_plot(shap_values_class_2, X_test.reshape(X_test.shape[0], X_test.shape[1]), plot_type="bar")

class_index = 0  
shap_values_class_2 = shap_values[class_index]

shap.summary_plot(shap_values_class_2, X_test.reshape(X_test.shape[0], X_test.shape[1]), plot_type="bar")

fig_all_samples = shap.summary_plot(shap_values, X_test.reshape(X_test.shape[0], X_test.shape[1]),  show=False)
plt.savefig("shap_summary_all_samples.png", dpi=300, bbox_inches='tight')
plt.close()

