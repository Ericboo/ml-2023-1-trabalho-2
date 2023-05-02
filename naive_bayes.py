import numpy as np
from sklearn.naive_bayes import GaussianNB

def dropLabel(dataset):
    aux = [row[:1] + row[2:] for row in dataset]
    dataset = [row for row in aux]
    return dataset

def subdivide(dataset, divisao):
    if divisao == 'treinamento':
        treinamento_positivos = dataset[:100]
        treinamento_negativos = dataset[600:1001]        
        return np.concatenate((treinamento_positivos, treinamento_negativos), axis=0)
    elif divisao == 'teste':
        teste_negativos = dataset[1001:2000]
        teste_positivos = dataset[101:500]
        teste = np.concatenate((teste_positivos, teste_negativos), axis=0)
        return teste
    elif divisao == 'label treinamento':
        label = []
        treinamento = subdivide(dataset, 'treinamento')
        for i in range(len(treinamento)):
           label.append(treinamento[i][1])
        return label
    elif divisao == 'label teste':
        label = []
        teste = subdivide(dataset, 'teste')
        for i in range(len(teste)):
           label.append(teste[i][1])
        return label

def execute(dataset):
    
    X_train = subdivide(dataset, 'treinamento')
    X_test = subdivide(dataset, 'teste')
    y_train = subdivide(dataset, 'label treinamento')
    y_test = subdivide(dataset, 'label teste')
    X_train = dropLabel(X_train)
    X_test = dropLabel(X_test)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
        % (len(X_test), (y_test != y_pred).sum()))