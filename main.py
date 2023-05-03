#TRABALHO 2 DE APRENDIZADO DE MÁQUINA 2023.1 POR ERIC JONAI COSTA SOUZA

import pandas
import naive_bayes as nb
import arvore_decisao as ad
import knn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

#Abre o arquivo
df = pandas.read_excel('./Dataset.xlsx')

#Guarda o quantitativo de linhas
df_lines = df.shape[0]

def valor_chave(dicionario):
    return dicionario['SARS-Cov-2 exam result']

#Guarda todas as informações num dicionário e ordena para deixar os positivos primeiro
dataset = df.to_dict('records')
#Coloca os positivos em primeiro para facilitar as divisões
dataset = sorted(dataset, key=valor_chave, reverse=True)
#Remove colunas indesejadas
for i in range(len(dataset)):
    dataset[i].pop('Patient ID', None)
    dataset[i].pop('Parainfluenza 1', None)
    dataset[i].pop('Parainfluenza 2', None)
    dataset[i].pop('Parainfluenza 3', None)
    dataset[i].pop('Bordetella pertussis', None)
    dataset[i].pop('Fio2 (venous blood gas analysis)', None)
    dataset[i].pop('Myeloblasts', None)
    dataset[i].pop('Vitamin B12', None)
    dataset[i].pop('Lipase dosage', None)

#Reduz o dataset de dicionários para vetores
keys = dataset[0].keys()
dataset = [[d[key] for key in keys] for d in dataset]

print('\nRESULTADO DA EXECUÇÃO DO KNN:')
result = knn.execute(dataset)
print('Matriz de Confusão:\n', confusion_matrix(result[0], result[1]))
print('Acurácia:', accuracy_score(result[0], result[1]))
print('Precisão:', precision_score(result[0], result[1], zero_division=1))
print('Revocação:', recall_score(result[0], result[1]))

print('\nRESULTADO DA EXECUÇÃO DO NAIVE BAYES:')
result = nb.execute(dataset)
print('Matriz de Confusão:\n', confusion_matrix(result[0], result[1]))
print('Acurácia:', accuracy_score(result[0], result[1]))
print('Precisão:', precision_score(result[0], result[1], zero_division=1))
print('Revocação:', recall_score(result[0], result[1]))

print('\nRESULTADO DA EXECUÇÃO DA ÁRVORE DE DECISÃO:')
result = ad.execute(dataset)
print('Matriz de Confusão:\n', confusion_matrix(result[0], result[1]))
print('Acurácia:', accuracy_score(result[0], result[1]))
print('Precisão:', precision_score(result[0], result[1], zero_division=0))
print('Revocação:', recall_score(result[0], result[1]))