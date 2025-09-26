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
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.cluster import Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import shap
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
import os

data = pd.read_excel("LCM-PBT.xlsx")
X = data.iloc[:, 1:-1].values  


threshold = 0.07
birch = Birch(threshold=threshold, branching_factor=30, n_clusters=None)
birch.fit(X)
subcluster_centers = birch.subcluster_centers_
n_subclusters = len(subcluster_centers)
print(f" {n_subclusters}")

best_score = -1
best_n = 2
best_labels = None
linkage_options = ['ward', 'complete', 'average']

for linkage in linkage_options:
    for n_clusters in range(2, min(11, n_subclusters+1)):
        agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels_temp = agglo.fit_predict(subcluster_centers)
        score = silhouette_score(subcluster_centers, labels_temp)
        if score > best_score:
            best_score = score
            best_n = n_clusters
            best_labels = labels_temp
            best_linkage = linkage

print(f" {best_n}, linkage={best_linkage}, silhouette={best_score:.3f}")

first_level_labels = birch.predict(X)
final_labels = np.zeros(X.shape[0], dtype=int)
for i in range(len(X)):
    sub_idx = first_level_labels[i]
    final_labels[i] = best_labels[sub_idx]

plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], c=final_labels, cmap='viridis', s=50)
plt.title('Two-Step Clustering (SPSS-like)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()

data = pd.read_excel(file_path, header=None)

data_cleaned = data.loc[:, (data != 0).any(axis=0)]

X = data.iloc[:,0:]

selector = VarianceThreshold(threshold=0.01)
X_new = selector.fit_transform(X)

# Extract the relevant data for the model
X = data.iloc[:, 1:-1]  # Features (from column B to BCO, rows 2 to 19)
y = data.iloc[:, -1]    # Target (column BCP, rows 2 to 1+9)
model = RandomForestClassifier(random_state=0)
model.fit(X, y)


selector = RFECV(model, step=200, cv=10, n_jobs=-1)     
selector = selector.fit(X, y)
X_wrapper = selector.transform(X)         
score =cross_val_score(model , X_wrapper, y, cv=10,n_jobs=-1).mean()   
print(score)
print(selector.support_)                                
print(selector.n_features_)                       
print(selector.ranking_)                                
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

lasso = Lasso(alpha=0.1)  
selector = SelectFromModel(lasso)
selector.fit(X, y)

selected_features = X.columns[selector.get_support()]
print( selected_features) 


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



X_scaled = data.select_dtypes(include=[np.number]) 

n_neighbors = 20

thresholds = [-1.5, -1.4, -1.3, -1.2]

results = []

lof = LocalOutlierFactor(n_neighbors=n_neighbors)

y_pred = lof.fit_predict(X_scaled)

lof_scores = lof.negative_outlier_factor_

for threshold in thresholds:
    outlier_mask = lof_scores < threshold
    outlier_count = np.sum(outlier_mask)
    results.append({
        'threshold': threshold,
        'outlier_count': outlier_count
    })

results_df = pd.DataFrame(results)
print(results_df)
selected_threshold = -1.23

data['prediction'] = y_pred  
data['lof_scores'] = lof_scores

data['outlier'] = lof_scores < selected_threshold

normal_data = data[~data['outlier']]

plt.figure(figsize=(10, 6))

plt.rcParams['font.family'] = 'Times New Roman'


plt.hist(lof_scores, bins=50, edgecolor='k', color='skyblue', alpha=0.7)

plt.title('Distribution of LOF Decision Function Scores', fontsize=16)
plt.xlabel('Local Outlier Factor (LOF)', fontsize=14)
plt.ylabel('Number of Samples', fontsize=14)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('lof_distribution.png', dpi=300, bbox_inches='tight')

plt.tight_layout() 
plt.show()
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
