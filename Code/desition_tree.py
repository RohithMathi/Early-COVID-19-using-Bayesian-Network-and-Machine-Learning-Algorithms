import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns


def desition_tree(train_df_features,train_df_target):
    desition = DecisionTreeClassifier()
    # print(train_df_features.head(20))

    lab = LabelEncoder()
    train_df_features['immunity'] = lab.fit_transform(train_df_features['immunity'].astype('str'))

    desition.fit(train_df_features, train_df_target)

    return desition


def desition_tree_accuracy(clf,test_df_features, test_df_target):
    lab = LabelEncoder()
    test_df_features['immunity'] = lab.fit_transform(test_df_features['immunity'].astype('str'))
    # Predict the response for test dataset
    y_pred = clf.predict(test_df_features)
    #print(y_pred)
    print('accuracy: ', accuracy_score(test_df_target, y_pred))

    matrix = confusion_matrix(test_df_target, y_pred)
    plot2 = plt.figure(2)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for decision tree Model')



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
    clf=desition_tree(train_df_features,train_df_target)
    desition_tree_accuracy(clf,test_df_features, test_df_target)