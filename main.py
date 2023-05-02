#TRABALHO 1 DE APRENDIZADO DE MÁQUINA 2023.1 POR ERIC JONAI COSTA SOUZA

import pandas
import numpy as np
from statistics import mode
import naive_bayes as nb

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

#naive bayes
nb.execute(dataset)