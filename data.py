from enum import auto
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics  import classification_report, confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
import numpy as np


# Bỏ qua hàng đầu tiên, dùng hàng 2 làm header
df = pd.read_csv("SWAT.csv",header=1, low_memory=False)


#-------------------------------------------------------------------------------
# Check missing values
#-------------------------------------------------------------------------------

# print("Số lượng missing values mỗi cột:")
# print(df.isnull().sum())

# print("\nTỷ lệ missing values mỗi cột:")
# print(df.isnull().mean() * 100)

# missing_rows = df[df.isnull().any(axis=1)]
# print("\nCác cột có missing values:")
# print(missing_rows.head())


# df.columns = df.columns.str.strip().str.replace(" ", "_")


#-------------------------------------------------------------------------------
#Process timestamp column
#-------------------------------------------------------------------------------

time_candidates = [c for c in df.columns if 'time' in c.lower() or 'timestamp' in c.lower()]
if not time_candidates:
    timestamp_col = df.columns[0]
else:
    timestamp_col = time_candidates[0]

df = df[~df[timestamp_col].astype(str).str.lower().eq(timestamp_col.lower())]

df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)

df = df.dropna(subset=[timestamp_col])

df = df.set_index(timestamp_col) 

print("Timestamp column used:", timestamp_col)


#-------------------------------------------------------------------------------# Check columns
# print("\nAll columns (count={}):".format(len(df.columns)))
# print(df.columns.tolist())  

#-------------------------------------------------------------------------------
# identify attack time intervals
#-------------------------------------------------------------------------------



attack_periods = [
    # Attack 1: 15h (GMT+8) -> 7h (GMT+0)
    ('2019-07-20 07:08:46', '2019-07-20 07:10:31'),
    # Attack 2: 15h (GMT+8) -> 7h (GMT+0)
    ('2019-07-20 07:15:00', '2019-07-20 07:19:32'),
    # Attack 3: 15h (GMT+8) -> 7h (GMT+0)
    ('2019-07-20 07:26:57', '2019-07-20 07:30:48'),
    # Attack 4: 15h (GMT+8) -> 7h (GMT+0)
    ('2019-07-20 07:38:50', '2019-07-20 07:46:20'),
    # Attack 5: 15h (GMT+8) -> 7h (GMT+0)
    ('2019-07-20 07:54:00', '2019-07-20 07:56:00'),
    # Attack 6: 16h (GMT+8) -> 8h (GMT+0)
    ('2019-07-20 08:02:56', '2019-07-20 08:16:18')
]

attack_datetime_periods = [
    (pd.to_datetime(start).tz_localize("UTC"), pd.to_datetime(end).tz_localize("UTC"))
    for start, end in attack_periods
]

df['Attack'] = 0
for start, end in attack_datetime_periods:
    df.loc[start:end, 'Attack'] = 1

print("\n--- Thống kê số điểm dữ liệu Normal (0) và Attack (1) ---")

print(df['Attack'].value_counts())

print("Tỷ lệ Attack trong toàn bộ dữ liệu: {:.2f}%".format(df['Attack'].mean() * 100))



#-------------------------------------------------------------------------------
# Basic statistics
#-------------------------------------------------------------------------------

plt.rcParams['figure.figsize'] = (15, 5)


# print("\n--- Thống kê cơ bản về dữ liệu số ---")
#print(df.describe())        # Thống kê mô tả

status_cols = [col for col in df.columns if df[col].astype(str).str.contains("Active|Inactive", case=False).any()]

# print("\n Các cột có Active/Inactive:", status_cols)

for col in status_cols:
    df[col] = df[col].map({'Active': 1, 'Inactive': 0})

print(df[status_cols].head())

target_column = 'Attack'
feature_columns = df.columns.drop(target_column)

for col in feature_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

feature_std = df[feature_columns].std()
useless_columns = feature_std[feature_std == 0].index

if not useless_columns.empty:
    print("\nCác cột không có biến thiên (std=0) và có thể loại bỏ:")
    print(list(useless_columns))
    print("Số cột bị loại bỏ:", len(useless_columns))
    # df.drop(columns=useless_columns, inplace=True)
else:
    print("\nKhông có cột nào bị loại bỏ do không có biến thiên.")

#-------------------------------------------------------------------------------
# Visualize distributions
#-------------------------------------------------------------------------------

# Show some distributions
# features_to_plot = ['LIT 101', 'FIT 101', 'P101 Status']
# df[features_to_plot].plot(subplots=True, title='Biến thiên của các cảm biến theo thời gian')
# plt.xlabel('Timestamp')
# plt.show()

# # --------------------------------------------------------------------------------
# # Visualize attack periods on a specific sensor
# fig, ax = plt.subplots()

# df['LIT 301'].plot(ax=ax, label='LIT301', color='blue')

# start,end = attack_datetime_periods[1]
# ax.axvspan(start, end, color='red', alpha=0.3, label ='_nolegend_')

# ax.set_title('Giá trị cảm biến LIT301 và cuộc tấn công')
# ax.set_ylabel('Giá trị LIT301')
# ax.legend()
# plt.show()



#--------------------------------------------------------------------------------\
# Feature correlations analysis 

nan_columns = df.columns[df.isnull().all()]

if not nan_columns.empty:
    print("\nCÁC CỘT BỊ TRỐNG TRÊN HEATMAP (do toàn giá trị NaN):")
    print(list(nan_columns))
else:
    print("\nKhông tìm thấy cột nào chứa toàn giá trị NaN.")

corr_matrix = df.corr() 

# plt.figure(figsize=(20, 15))

# sns.heatmap(corr_matrix, cmap='coolwarm', center=0)

# plt.title('Bản đồ nhiệt tương quan giữa các đặc trưng')

# plt.show()


# print("\n--- 20 features tương quan mạnh nhất với cột 'Attack' ---")
# attack_correlation = corr_matrix['Attack'].abs().sort_values(ascending=False)
# print(attack_correlation.head(20))

# print("\n--- 3 features tương quan mạnh nhất với cột  ---")
# correlation = corr_matrix['FIT 502'].abs().sort_values(ascending=False)
# print(correlation.head(10))

#---------------------------------------------------------------------------------
#Compare distributions of selected features during normal and attack periods

# feature_to_compare = 'AIT 201'

# plt.figure(figsize=(10, 6))    

# sns.kdeplot(data=df, x=feature_to_compare, hue='Attack', common_norm=False, fill=True)

# plt.title(f'Phân phối của {feature_to_compare} trong các giai đoạn bình thường và bị tấn công')
# plt.show()


#---------------------------------------------------------------------------------
#Visualize outliers in selected features

# feature_for_boxplot = attack_correlation.head(10).index.drop('Attack')

# print(f"Biểu đồ hộp của các đặc trưng hàng đầu: {list(feature_for_boxplot)}")



# scaler = StandardScaler()

# df_scaled_features = pd.DataFrame(scaler.fit_transform(df[feature_for_boxplot]), columns=feature_for_boxplot, index=df.index)

# plt.figure(figsize=(20, 8))

# sns.boxplot(data=df_scaled_features)

# plt.title('Biểu đồ hộp của các đặc trưng đã chuẩn hóa')

# plt.ylabel('Giá trị chuẩn hóa')

# plt.xticks(rotation=45)

# plt.grid(True)

# plt.show()

#---------------------------------------------------------------------------------
# Select top features based on correlation with 'Attack'

selected_features = ['FIT 101', 'LIT 101', 'MV 101', 'P1_STATE', 'P101 Status', 'AIT 201', 'AIT 202', 'AIT 203', 'FIT 201', 'MV201', 'P203 Status', 'P205 Status',
                    'AIT 301', 'AIT 302', 'AIT 303', 'DPIT 301', 'FIT 301', 'LIT 301', 'MV 301', 'MV 302', 'MV 303', 'MV 304', 'P3_STATE', 'P301 Status', 
                    'AIT 402', 'FIT 401', 'LIT 401', 'P401 Status', 'UV401', 'AIT 501', 'AIT 502', 'AIT 503', 'AIT 504', 'FIT 501', 'FIT 502', 'FIT 503',
                    'FIT 504', 'MV 501', 'PIT 501', 'PIT 502', 'PIT 503', 'FIT 601', 'LSH 601', 'P601 Status']


df_model = df[selected_features + ['Attack']].copy()

print(f"Đã tạo DataFrame mới với {len(selected_features)} đặc trưng được chọn.")

print(df_model.head())


#---------------------------------------------------------------------------------
# missing value imputation


print(f"Số giá trị NaN trước khi xử lý: {df_model.isnull().sum().sum()}")

df_model.ffill(inplace=True)

df_model.bfill(inplace=True)

print(f"Số giá trị NaN sau khi xử lý: {df_model.isnull().sum().sum()}")


#- ---------------------------------------------------------------------------
# Split data into train and test sets based on time

split_timestamp = pd.to_datetime('2019-07-20 07:00:00').tz_localize("UTC")

df_train = df_model.loc[df_model.index <= split_timestamp]

df_test = df_model.loc[df_model.index > split_timestamp]

print("split timestamp:", split_timestamp)
print(f"Train set: {df_train.shape}, Test set: {df_test.shape}")


# label distribution

print("\n--- Phân bố nhãn trong tập train ---")
print(df_train['Attack'].value_counts())

print("\n--- Phân bố nhãn trong tập test ---")
print(df_test['Attack'].value_counts())

X_train = df_train[selected_features]
y_train = df_train['Attack']

X_test = df_test[selected_features]
y_test = df_test['Attack']

print("Completed data preprocessing and splitting!")

#---------------------------------------------------------------------------------
# Chuẩn hóa dữ liệu(data scaling)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Dữ liệu đã được chuẩn hóa.")



#---------------------------------------------------------------------------------
# Train Isolation Forest model

iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

iforest.fit(X_train_scaled)

predictions = iforest.predict(X_test_scaled)

# map predictions to 0 (normal) and 1 (anomaly)

predictions_mapped = np.where(predictions == -1, 1, 0)

# print(f"Ví dụ 10 dự đoán đầu tiên (đã chuyển đổi): {predictions_mapped[:10]}")

#---------------------------------------------------------------------------------
# Evaluate model performance

print("\n--- Báo cáo phân loại ---")  

print(classification_report(y_test, predictions_mapped, target_names=['Normal', 'Attack']))

accuracy = accuracy_score(y_test, predictions_mapped)

precision = precision_score(y_test, predictions_mapped)

recall = recall_score(y_test, predictions_mapped)

f1 = f1_score(y_test, predictions_mapped)

# Visualize confusion matrix

print("\n--- Ma trận nhầm lẫn ---")

cm = confusion_matrix(y_test, predictions_mapped)

plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Normal', 'Predicted Attack'], yticklabels=['Normal', 'Attack'])

plt.title('Confusion Matrix for Isolation Forest on SWaT Dataset')

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.show()