import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bỏ qua hàng đầu tiên, dùng hàng 2 làm header
df = pd.read_csv("SWAT.csv",header=1, low_memory=False)

# # Chỉ lấy 1000 dòng đầu
# df = df[:100]

# print(df.columns.tolist()) 


# cols = ["FIT 101", "FIT 201", "FIT 301"]

# plt.figure(figsize=(10,6))
# for c in cols:
#     if c in df.columns:  
#         plt.scatter(df.index, df[c], label=c, s=10)

# plt.xlabel("Thời gian (index)")
# plt.ylabel("Lưu lượng nước (Flow)")
# plt.title("Biểu đồ lưu lượng nước SWaT - 3 vòi")
# plt.legend()
# plt.show()



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

print("\n Các cột có Active/Inactive:", status_cols)

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
    # df.drop(columns=useless_columns, inplace=True)
else:
    print("\nKhông có cột nào bị loại bỏ do không có biến thiên.")

#-------------------------------------------------------------------------------
# Visualize distributions
#-------------------------------------------------------------------------------