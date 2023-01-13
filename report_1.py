import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
import pickle

lab_enc = preprocessing.LabelEncoder()

path = "ODI_data.csv"


def change_stringValue(df):
    list = []
    for x in df:
        if str(x).find("-") == -1:
            if str(x).find("*") == -1:
                list.append(float(x))
            else:
                x = str(x).replace("*", "")
                list.append(float(x))
        else:
            list.append(0.00)
    return list


original_dataFrame = pd.read_csv(path)

df = pd.read_csv(path)

df = df.dropna(axis=1)

df.drop('Unnamed: 0', axis=1, inplace=True)

df.drop("Player", axis=1, inplace=True)

df[['Start Year', 'Last Year']] = df['Span'].str.split('-', n=1, expand=True)

df.drop("Span", axis=1, inplace=True)

df.drop("Start Year", axis=1, inplace=True)
df.drop("Last Year", axis=1, inplace=True)

for column in df.columns:
    df[column] = change_stringValue(df[column])

pd_df = pd.DataFrame.from_dict(df)

print(pd_df.columns)

train_set, test_set = train_test_split(pd_df, test_size=0.25)

train_y_set = lab_enc.fit_transform(train_set['100'])
train_set.drop('100', axis=1)

test_y_set = lab_enc.fit_transform(test_set['100'])
test_set.drop('100', axis=1)

model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=100)

model.fit(train_set, train_y_set)
predictions = model.predict(test_set)
model_score = model.score(test_set, test_y_set)
rec_score = recall_score(test_y_set, predictions, average="macro")
score = precision_score(test_y_set, predictions, average='macro')
f1Score = f1_score(test_y_set, predictions, average="macro")

print("LR Model", model_score)
print("LR Precision", score)
print("LR Recall", rec_score)
print("LR F1", f1Score)
#
svm_model = SVC(random_state=0)

svm_model.fit(train_set, train_y_set)
predictions = svm_model.predict(test_set)
model_score = svm_model.score(test_set, test_y_set)
rec_score = recall_score(test_y_set, predictions, average="macro")
score = precision_score(test_y_set, predictions, average='macro')
f1Score = f1_score(test_y_set, predictions, average="macro")

print("SVC Model", model_score)
print("SVC Precision", score)
print("SVC Recall", rec_score)
print("SVC F1", f1Score)



# analysis={}
#
# for ele in df.columns:
#     analysis[ele]= {}
#     analysis[ele]["Range"] = [df[ele].max(), df[ele].min()]
#     analysis[ele]["Mean"] = df[ele].mean()
#     analysis[ele]["Mode"] = statistics.multimode(df[ele])
#     analysis[ele]["Median"] = df[ele].median()
#     analysis[ele]["Standard Deviation"] = df[ele].std()
#
# for column in df.columns:
#     print("        " + column + "  ", end=" ")
#
# analysis_df = pd.DataFrame.from_dict(analysis)
# analysis_df.to_csv("analysis.csv")
#
#
#
# print(analysis_df)
# print("Range :", end="")
# for column in df.columns:
#     print(str(analysis[column]["Range"]), end=" ")
#
# print()
# print("Mean :", end="")
# for column in df.columns:
#     print(str(analysis[column]["Mean"]), end=" ")
#
# print()
# print("Mode :", end="")
# for column in df.columns:
#     print(str(analysis[column]["Mode"]), end=" ")
#
# print()
# print("Median :", end="")
# for column in df.columns:
#     print(str(analysis[column]["Median"]), end=" ")
#
#
# print()
# print("Standard Deviation :", end="")
# for column in df.columns:
#     print(str(analysis[column]["Standard Deviation"]), end=" ")
#
#
#
# # plt.show(plotting_df["Player"], plotting_df["Mat"])
#
#
# original_dataFrame[['Player', 'Region']] = original_dataFrame['Player'].str.split('(', n=1, expand=True)
# original_dataFrame['Region'] = original_dataFrame['Region'].map(lambda x: x.rstrip(')'))
# original_dataFrame[['Region1', 'Region']] = original_dataFrame['Region'].str.split('/', n=1, expand=True)
# original_dataFrame.dropna()


# plt.figure()
# regionMap = {}
# for region in original_dataFrame["Region1"]:
#     if region is None:
#         continue
#     if region in regionMap:
#         regionMap[region] = regionMap[region] + 1
#     else:
#         regionMap[region] = 1
#
# print(regionMap)
#
# plt.scatter(regionMap.keys(), regionMap.values())
# plt.xticks(rotation=90)
# plt.savefig('region_count.png', bbox_inches='tight')
#
# plt.figure()
# regionMap = {}
# for index, hundred in enumerate(original_dataFrame["100"]):
#     if hundred is None:
#         continue
#     if str(hundred).find("-") != -1:
#         continue
#     region = original_dataFrame["Region1"][index]
#     if region in regionMap:
#         regionMap[region] = regionMap[region] + int(hundred)
#     else:
#         regionMap[region] = int(hundred)
#
#
# plt.title("Region with Number of 100s")
# plt.plot(regionMap.keys(), regionMap.values())
# plt.xlabel("Region")
# plt.xticks(rotation=90)
# plt.ylabel("100s")
# plt.savefig("region_100_count.png", bbox_inches='tight')
#
# plt.figure()
# regionMap = {}
# for index, hundred in enumerate(original_dataFrame["50"]):
#     if hundred is None:
#         continue
#     if str(hundred).find("-") != -1:
#         continue
#     region = original_dataFrame["Region1"][index]
#     if region in regionMap:
#         regionMap[region] = regionMap[region] + int(hundred)
#     else:
#         regionMap[region] = int(hundred)
#
#
# plt.title("Region with Number of 50s")
# plt.stem(regionMap.keys(), regionMap.values(), use_line_collection=True)
# plt.xlabel("Region")
# plt.xticks(rotation=90)
# plt.ylabel("50s")
# plt.savefig("region_50_count.png", bbox_inches='tight')
#
#
#
# plt.figure()
# regionMap = {}
# for index, hundred in enumerate(original_dataFrame["0"]):
#     if hundred is None:
#         continue
#     if str(hundred).find("-") != -1:
#         continue
#     region = original_dataFrame["Region1"][index]
#     if region in regionMap:
#         regionMap[region] = regionMap[region] + int(hundred)
#     else:
#         regionMap[region] = int(hundred)
#
#
# plt.title("Region with Number of 0s")
# fig1, ax1 = plt.subplots()
# ax1.pie(regionMap.values(), labels=regionMap.keys(), autopct='%1.1f%%',
#         shadow=True, startangle=90)
# plt.xticks(rotation=90)
# ax1.axis('equal')
# plt.savefig("region_0s_count.png", bbox_inches='tight')
#
#
#
#
# list = [];
# for x in original_dataFrame["Player"]:
#     list.append(x[:1])
#
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(list, original_dataFrame["Mat"])
# ax.set_ylabel("Matches")
# ax.set_xlabel("Players")
# ax.set_title('Matches and Players')
# plt.savefig('matches_players.png', bbox_inches='tight')
