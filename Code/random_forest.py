import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


def random_forest(train_df_features, train_df_target):
    forest = RandomForestClassifier(n_estimators=100, random_state=0)

    lab = LabelEncoder()
    train_df_features['immunity'] = lab.fit_transform(train_df_features['immunity'].astype('str'))

    forest.fit(train_df_features, train_df_target)

    return forest


def random_forest_accuracy(forest, test_df_features, test_df_target):

    lab = LabelEncoder()
    test_df_features['immunity'] = lab.fit_transform(test_df_features['immunity'].astype('str'))

    y_pred = forest.predict(test_df_features)
    plot3 = plt.figure(3)
    print('accuracy: ', accuracy_score(test_df_target, y_pred))
    matrix = confusion_matrix(test_df_target, y_pred)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10},cmap=plt.cm.Greens, linewidths=0.2)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')




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
    clf=random_forest(train_df_features,train_df_target)
    random_forest_accuracy(clf,test_df_features, test_df_target)