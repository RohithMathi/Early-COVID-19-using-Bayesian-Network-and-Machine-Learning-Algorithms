import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import seaborn as sns

def navie_bayes(train_df_features, train_df_target):
    gnb = GaussianNB()
    lab = LabelEncoder()
    train_df_features['immunity'] = lab.fit_transform(train_df_features['immunity'].astype('str'))

    # print(train_df_features.head())

    gnb.fit(train_df_features, train_df_target)
    return gnb


def navie_bayes_accuracy(gnb, test_df_features, test_df_target):
    lab = LabelEncoder()
    test_df_features['immunity'] = lab.fit_transform(test_df_features['immunity'].astype('str'))

    y_pred = gnb.predict(test_df_features)
    plot4 = plt.figure(4)
    print('accuracy: ',accuracy_score(test_df_target, y_pred))
    matrix = confusion_matrix(test_df_target, y_pred)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
                cmap=plt.cm.Greens, linewidths=0.2)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for naivebayes Model')



if __name__ == '__main__':
    df = pd.read_csv("corona_tested_individuals_modified.csv", encoding='utf-8')
    print(df.head())

    df_features = df.drop('corona_result', inplace=False, axis=1)
    df_target = df.corona_result

    # print(df_features.head())
    # print(df_target.head())
    train_df_features, test_df_features, train_df_target, test_df_target = train_test_split(df_features, df_target,
                                                                                            test_size=0.05,
                                                                                            random_state=1)
    clf=navie_bayes(train_df_features,train_df_target)
    navie_bayes_accuracy(clf,test_df_features, test_df_target)