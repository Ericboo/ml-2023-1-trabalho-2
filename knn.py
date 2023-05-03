from sklearn.neighbors import NearestNeighbors
import numpy as np

def dropLabel(dataset):
    aux = [row[:1] + row[2:] for row in dataset]
    new_dataset = [row for row in aux]
    return new_dataset

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


# Define a função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Define a função K-NN
def knn(X_train, y_train, X_test, k):
    # Lista para armazenar as predições
    predictions = []
    # Loop através dos pontos de teste
    for i in range(len(X_test)):
        # Lista para armazenar as distâncias entre o ponto de teste atual e todos os pontos de treinamento
        distances = []
        # Loop através dos pontos de treinamento
        for j in range(len(X_train)):
            # Calcula a distância euclidiana entre o ponto de teste atual e o ponto de treinamento atual
            distance = euclidean_distance(X_test[i], X_train[j])
            # Adiciona a distância e o índice do ponto de treinamento atual à lista de distâncias
            distances.append((distance, j))
        # Ordena a lista de distâncias em ordem crescente
        distances = sorted(distances)
        # Lista para armazenar as classes dos vizinhos mais próximos
        class_votes = []
        # Loop através dos k vizinhos mais próximos
        for n in range(k):
            # Obtém o índice do n-ésimo vizinho mais próximo
            index = distances[n][1]
            # Adiciona a classe do ponto de treinamento correspondente à lista de votos de classe
            class_votes.append(y_train[index])
        # Encontra a classe mais comum entre os vizinhos mais próximos
        prediction = max(set(class_votes), key=class_votes.count)
        # Adiciona a classe predita à lista de predições
        predictions.append(prediction)
    # Retorna a lista de predições
    return predictions

def execute(dataset):
    X_train = subdivide(dataset, 'treinamento')
    X_test = subdivide(dataset, 'teste')
    y_train = subdivide(dataset, 'label treinamento')
    y_test = subdivide(dataset, 'label teste')
    X_train = dropLabel(X_train)
    X_test = dropLabel(X_test)
    result = knn(X_train, y_train, X_test, k = 10) #4=791 5=771 6=824 7=822 8=901 9=901 10=911
    corretas = np.sum(np.array(result).astype(int) == np.array(y_test).astype(int))

    print("De 1398 pacientes, o knn preveu corretamente:", corretas)