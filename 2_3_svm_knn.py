import numpy as np
import pandas as pd
from sklearn import preprocessing, svm, neighbors
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as pl


for j in ['Two', 'Three']:
    accuracy_j_svm=[]
    accuracy_j_knn=[]

    for i in range(1, 16):
        path= '/home/machinelearningstation/PycharmProjects/2/'+j+'/data'+str(i)+'.csv'
        df0=pd.read_csv(path, error_bad_lines=False)
        df=df0.replace([np.inf, -np.inf], np.nan).dropna()
        df_features = df.drop('marker', axis=1)
        x=preprocessing.maxabs_scale(df_features)

        le=preprocessing.LabelEncoder()
        y=le.fit_transform(df['marker'])

        svc=svm.SVC(C=1, kernel='rbf')
        knn=neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
        kfold=KFold(n_splits=10)

        accuracy_j_svm.append(np.mean(cross_val_score(svc, x, y, cv=kfold)))
        accuracy_j_knn.append(np.mean(cross_val_score(knn, x, y, cv=kfold)))

    n=range(1,16)
    pl.plot(n, accuracy_j_svm, 'b')
    pl.plot(n, accuracy_j_knn, 'r')
    pl.ylim((0,1))
    pl.title('Accuracy of '+j+' Classes')
    pl.xlabel('Dataset')
    pl.ylabel('Accuracy')
    pl.legend(['SVM', 'KNN'], loc=8, shadow=True)
    pl.savefig('/home/machinelearningstation/PycharmProjects/2/'+j+'.jpg')
    pl.cla()


for k in ['precision', 'recall', 'f1']:
    k_svm=[]
    k_knn=[]

    for i in range(1, 16):
        path = '/home/machinelearningstation/PycharmProjects/2/Two/data' + str(i) + '.csv'
        df = pd.read_csv(path, error_bad_lines=False)
        df_features = df.drop('marker', axis=1)
        x = preprocessing.maxabs_scale(np.nan_to_num(df_features))

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(np.nan_to_num(df['marker']))

        svc = svm.SVC(C=1, kernel='rbf')
        knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
        kfold = KFold(n_splits=10)

        k_svm.append(np.mean(cross_val_score(svc, x, y, cv=kfold, scoring=k)))
        k_knn.append(np.mean(cross_val_score(knn, x, y, cv=kfold, scoring=k)))

    n=range(1,16)
    pl.plot(n, k_svm, 'bo')
    pl.plot(n, k_knn, 'ro')
    pl.ylim((0,1))
    pl.title(j)
    pl.xlabel('Dataset')
    pl.ylabel(j)
    pl.legend(['SVM', 'KNN'], loc=2, shadow=True)
    pl.savefig('/home/machinelearningstation/PycharmProjects/2/2_'+k+'.jpg')
    pl.cla()