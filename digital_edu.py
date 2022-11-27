import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 


df = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

print(df.info())

to_del = ['bdate', 'followers_count', 'graduation',
          'relation', 'life_main', 'people_main', 'city', 'last_seen',
          'occupation_name', 'career_start', 'career_end', 'education_status', 'occupation_type', 'langs']

df.drop(to_del, axis = 1, inplace = True)
df2.drop(to_del, axis = 1, inplace = True)

df["has_mobile"] = df["has_mobile"].apply(int)
df2["has_mobile"] = df2["has_mobile"].apply(int)

def fill_sex(sex):
    if sex == 2:
        return 1
    return 0

df['sex'] = df['sex'].apply(fill_sex)
df2['sex'] = df2['sex'].apply(fill_sex)

df[list(pd.get_dummies(df["education_form"]).columns)] = pd.get_dummies(df["education_form"])
df.drop('education_form', axis = 1, inplace = True)
df2[list(pd.get_dummies(df2["education_form"]).columns)] = pd.get_dummies(df2["education_form"])
df2.drop('education_form', axis = 1, inplace = True)

#------------------

x_train = df.drop('result', axis = 1)
y_train = df['result']
x_test = df2

sc = StandardScaler ()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

id = df2['id']
result = pd.DataFrame({'id': id, 'result': y_pred})

result.to_csv('answer.csv', index = False)