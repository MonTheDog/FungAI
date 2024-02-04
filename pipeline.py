import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Ignoriamo i FutureWarning di Pandas essendo questo un progetto accademico
warnings.simplefilter(action='ignore', category=FutureWarning)

# Data Understanding and Preparation (Graphs, Summary Statistics, Data Cleaning, Data Transformation, Feature Engineering)

# Importiamo il dataset e visualizziamo le prime righe
dataset = pd.read_csv("mushroom.csv", index_col=0)
print(dataset.head())  # 8124 righe, 23 colonne

# Visualizziamo i tipi delle variabili del dataset
print(dataset.dtypes)

# Notiamo che tutte le colonne sono di tipo object, andiamo a sostituire i valori con stringhe per
# evitare problemi futuri con i grafici

for column in dataset:
    dataset[column] = dataset[column].astype(str)

# Dalla documentazione del dataset ci rendiamo conto che solo una colonna (stalk-root) presenta
# valori mancanti, andiamo a studiarla meglio
print(dataset["stalk-root"].unique())  # ['e' 'c' 'b' 'r' 'nan']

# Possiamo notare che i valori mancanti sono segnati come nan, andiamo a contare quanti sono
print("Stalk-root mancanti:", dataset["stalk-root"].value_counts()["nan"])  # 2480

# Controlliamo quanto la feature sia importante per il nostro modello mostrando l'istogramma
dataset_edible = dataset[dataset["poisonous"] == "e"]
dataset_poisonous = dataset[dataset["poisonous"] == "p"]

plt.hist([dataset_edible["stalk-root"], dataset_poisonous["stalk-root"]], color=["#776754", "#CA0B00"],
         stacked=True, bins=range(6), align="left", rwidth=0.7)
plt.legend(["Edible", "Poisonous"])
plt.xticks([0, 1, 2, 3, 4], ["e", "c", "b", "r", "nan"])
plt.show()

# Potremmo decidere di eliminare la colonna, ma ciò comporterebbe una perdita di informazioni, in particolare alcuni
# di questi valori ("e" e "r") hanno una buona potenza predittiva, mentre i valori mancanti hanno una distribuzione
# più varia. Andiamo a sostituire i valori mancanti con un valore "missing"
dataset["stalk-root"] = dataset["stalk-root"].replace(["nan"], "m")

# Andiamo a controllare cbe la sostituzione sia andata a buon fine
print(dataset["stalk-root"].unique())  # ['e' 'c' 'b' 'r' 'm']


# Modeling (Training, Testing, Optimization)



# Evaluation (Accuracy, Precision, Recall, F1, AUC, ROC, Confusion Matrix)



# Deployment (Small web app using streamlit)


