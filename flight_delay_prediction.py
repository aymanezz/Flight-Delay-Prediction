data = pd.read_csv('/kaggle/input/flight-delays/flights.csv',low_memory = False)

df = data[0:100000]
print(df.shape)

df.info()

df.value_counts('DIVERTED') 

sb.jointplot(data=df, x="SCHEDULED_ARRIVAL", y="ARRIVAL_TIME")

df.corr()


df.corr()['ARRIVAL_DELAY'].sort_values()

print(len(df.columns))
df = df.drop(['YEAR','FLIGHT_NUMBER','AIRLINE','DISTANCE','TAIL_NUMBER','TAXI_OUT',
         'SCHEDULED_TIME','DEPARTURE_TIME','WHEELS_OFF',
         'ELAPSED_TIME', 'AIR_TIME','WHEELS_ON','DAY_OF_WEEK','TAXI_IN','CANCELLATION_REASON'],axis =1)
print(len(df.columns))

df.isna().sum()


df = df.fillna(df.mean())
df.isna().sum()

result =[]
for cell in df['ARRIVAL_DELAY']:
    if cell > 15:
        result.append(1)
    else:
        result.append(0)

df['result']  = result

df['result'][:10]

df['result'].value_counts()

df=df.drop(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ARRIVAL_TIME', 'ARRIVAL_DELAY'],axis=1)


print(df.columns)

x = df.iloc[:,:-1]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

scaled_features = StandardScaler()

X_train_scaled = scaled_features.fit_transform(X_train)
X_test_scaled = scaled_features.fit_transform(X_test)


clf = DecisionTreeClassifier()
clf.fit(X_train_scaled,y_train)

pred = clf.predict_proba(X_test_scaled)

from sklearn.metrics import roc_auc_score


auc_score = roc_auc_score(y_test, pred[:,1])
print(auc_score)
