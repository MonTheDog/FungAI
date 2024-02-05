import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix)
import pickle
from sklearn.tree import plot_tree

# Ignoriamo i FutureWarning di Pandas essendo questo un progetto accademico
warnings.simplefilter(action='ignore', category=FutureWarning)


def fix_dataset_types(df):
    # Notiamo che tutte le colonne sono di tipo object, andiamo a sostituire i valori con stringhe per
    # evitare problemi futuri con i grafici
    for column in df:
        df[column] = df[column].astype(str)


def fix_stalk_root_missing_values(df):
    # Dalla documentazione del dataset ci rendiamo conto che solo una colonna (stalk-root) presenta
    # valori mancanti, andiamo a studiarla meglio
    print("Possibili valori di stalk-root:", df["stalk-root"].unique())  # ['e' 'c' 'b' 'r' 'nan']

    # Possiamo notare che i valori mancanti sono segnati come nan, andiamo a contare quanti sono
    print("Stalk-root mancanti:", df["stalk-root"].value_counts()["nan"])  # 2480

    # Controlliamo quanto la feature sia importante per il nostro modello mostrando l'istogramma
    dataset_edible = df[df["poisonous"] == "e"]
    dataset_poisonous = df[df["poisonous"] == "p"]

    plt.hist([dataset_edible["stalk-root"], dataset_poisonous["stalk-root"]], color=["#776754", "#CA0B00"],
             stacked=True, bins=range(6), align="left", rwidth=0.7, alpha=0.8)
    plt.legend(["Edible", "Poisonous"])
    plt.xticks([0, 1, 2, 3, 4], ["e", "c", "b", "r", "nan"])
    plt.xlabel("stalk-root")
    plt.ylabel("Frequenza")
    plt.savefig("images/stalk-root_histogram.png")
    plt.show()
    plt.close("all")

    # Potremmo decidere di eliminare la colonna, ma ciò comporterebbe una perdita di informazioni, in particolare alcuni
    # di questi valori ("e" e "r") hanno una buona potenza predittiva, mentre i valori mancanti hanno una distribuzione
    # più varia. Andiamo a sostituire i valori mancanti con un valore "missing"
    df["stalk-root"] = df["stalk-root"].replace(["nan"], "m")


def dataset_balancement(df):
    # Controlliamo se il dataset è bilanciato
    plt.pie(df["poisonous"].value_counts(), labels=["Edible", "Poisonous"], explode=(0, 0.05), autopct="%0.2f",
            colors=["#776754", "#CA0B00"])
    plt.savefig("images/poisonous_pie_chart.png")
    plt.show()
    plt.close("all")
    print("Bilanciamento del dataset: ",
          df["poisonous"].value_counts(normalize=True) * 100)  # 51.8% commestibili, 48.2% velenosi


def produce_histograms(df):
    # Andiamo a visualizzare gli istogrammi delle varie feature per capire come sono distribuiti i valori
    for column in df:
        plt.hist([df[df["poisonous"] == "e"][column], df[df["poisonous"] == "p"][column]], color=["#776754", "#CA0B00"],
                 stacked=True, bins=range(len(df[column].unique()) + 1), align="left", rwidth=0.7, alpha=0.8)
        plt.legend(["Edible", "Poisonous"])
        plt.xticks(range(len(df[column].unique())), df[column].unique())
        plt.xlabel(column)
        plt.ylabel("Frequenza")
        plt.savefig("images/" + column + "_histogram.png")
        plt.show()
        plt.close("all")


def fix_equal_columns(df):
    # Andiamo a rimuovere le colonne praticamente identiche e a rinominare l'altra
    df = df.drop(["stalk-surface-above-ring", "stalk-color-above-ring"], axis=1)
    df = df.rename(columns={"stalk-surface-below-ring": "stalk-surface",
                            "stalk-color-below-ring": "stalk-color"})
    return df


def encode_variables(df):
    # Andiamo a codificare le variabili categoriche in modo da poterle utilizzare nei nostri modelli
    # Utilizziamo OneHotEncoder per evitare di dare un peso maggiore a valori con un valore numerico più alto
    encoder = OneHotEncoder()
    encoded_df = pd.DataFrame(encoder.fit_transform(df).toarray(), columns=encoder.get_feature_names_out(df.columns))
    return encoded_df


def train_models(models, encoded_dataset, y_train):
    # Convertiamo il dataset in un array numpy per poterlo utilizzare nei modelli e addestriamoli
    X_train = encoded_dataset.to_numpy()
    for key in models.keys():
        models[key].fit(X_train, y_train)


# def train_models(models, encoded_dataset, y_train):
#     skf = StratifiedKFold(n_splits=10)  # Applicazione della Stratified K-Fold Validation (k=10)
#     accuracy, precision, recall, f1 = {}, {}, {}, {}  # Dictionary relativi alle metriche di valutazione
#
#     X_train = encoded_dataset.to_numpy()
#
#     for key in models.keys():
#         accuracy[key] = []
#         precision[key] = []
#         recall[key] = []
#         f1[key] = []
#
#     for train_index, test_index in skf.split(X_train, y_train):
#         for key in models.keys():
#             models[key].fit(X_train[train_index], y_train[train_index])
#
#             y_pred = models[key].predict(X_train[test_index])
#
#             accuracy[key].append(accuracy_score(y_true = y_train[test_index], y_pred = y_pred))
#             precision[key].append(precision_score(y_true = y_train[test_index], y_pred = y_pred))
#             recall[key].append(recall_score(y_true = y_train[test_index], y_pred = y_pred))
#             f1[key].append(f1_score(y_true = y_train[test_index], y_pred = y_pred))
#
#     mean_accuracy, mean_precision, mean_recall, mean_f1 = [], [], [], []
#
#     for key in models.keys():
#         mean_accuracy.append(round(np.mean(accuracy[key]),4))
#         mean_precision.append(round(np.mean(precision[key]),4))
#         mean_recall.append(round(np.mean(recall[key]),4))
#         mean_f1.append(round(np.mean(f1[key]),4))
#
#     scores = pd.DataFrame({"Accuracy": mean_accuracy, "Precision": mean_precision, "Recall": mean_recall, "F1": mean_f1}, index=models.keys())
#     print(scores)
#
#     for key in models.keys():
#         models[key].fit(X_train, y_train)


def test_models(models, encoded_dataset, y_test):
    # Prepariamo le liste per le metriche
    accuracy_list, precision_list, recall_list, f1_list, auc_list = [], [], [], [], []
    X_test = encoded_dataset.to_numpy()

    # Per ogni modello andiamo a calcolare le metriche e a visualizzare la confusion matrix e la ROC curve
    for key in models.keys():

        y_pred = models[key].predict(X_test)

        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        precision = precision_score(y_true=y_test, y_pred=y_pred)
        recall = recall_score(y_true=y_test, y_pred=y_pred)
        f1 = f1_score(y_true=y_test, y_pred=y_pred)
        auc = roc_auc_score(y_true=y_test, y_score=y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_pred)

        plt.plot(fpr, tpr, color="darkorange", lw=2)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve " + key)
        plt.savefig(key + "_roc_curve.png")
        plt.show()
        plt.close("all")

        matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        sns.heatmap(matrix, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix " + key)
        plt.savefig(key + "_confusion_matrix.png")
        plt.show()
        plt.close("all")

        accuracy_list.append(round(accuracy, 4))
        precision_list.append(round(precision, 4))
        recall_list.append(round(recall, 4))
        f1_list.append(round(f1, 4))
        auc_list.append(round(auc, 4))

    # Visualizziamo le performance sotto forma di tabella
    scores = pd.DataFrame({"Accuracy": accuracy_list, "Precision": precision_list, "Recall": recall_list, "F1": f1_list, "AUC": auc_list}, index=models.keys())
    print(scores)


def naive_bayes_fine_tuning(X, y):
    X = X.to_numpy()

    # Andiamo a fare un fine tuning dei parametri utilizzando GridSearchCV
    param_grid = {
        "var_smoothing": np.logspace(0, -9, num=100)
    }

    naive_bayes = GaussianNB()
    grid_search = GridSearchCV(estimator=naive_bayes, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, refit="True", scoring="accuracy")
    grid_search.fit(X, y)

    print(grid_search.best_params_)

    # Restituiamo il modello con i parametri ottimizzati
    return grid_search.best_estimator_


def decision_tree_fine_tuning(X, y):
    X = X.to_numpy()

    # Andiamo a fare un fine tuning dei parametri utilizzando GridSearchCV
    param_grid = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    tree = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, refit="True", scoring="accuracy")
    grid_search.fit(X, y)

    print(grid_search.best_params_)

    # Restituiamo il modello con i parametri ottimizzati
    return grid_search.best_estimator_


def plot_tree_graph(model, columns):
    # Visualizziamo il grafico dell'albero di decisione
    plt.figure(figsize=(10, 10))
    plot_tree(model, filled=True, feature_names=columns, class_names=["Edible", "Poisonous"])
    plt.savefig("images/Decision_Tree.png")
    plt.show()
    plt.close("all")


def get_decision_explanation(classifier, input_features, columns):
    # Cambiamo la forma dell'input in modo che sia compatibile con il modello
    input_features = input_features.reshape(1, -1)

    # Otteniamo il decision path e l'id dei nodi foglia
    node_indicator = classifier.decision_path(input_features)
    leaf_id = classifier.apply(input_features)

    result = []

    sample_id = 0
    # Ottiene gli id dei nodi attraversati dal campione
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]]

    for node_id in node_index:
        # Se il nodo è una foglia, non lo consideriamo
        if leaf_id[sample_id] == node_id:
            continue

        # Altrimenti otteniamo il nome della feature e il valore utilizzato per la decisione
        feature_index = classifier.tree_.feature[node_id]
        feature_name = columns[feature_index-1] if columns else str(feature_index)

        # Aggiungiamo la spiegazione alla lista
        explanations = "%s %s %s" % (feature_name, "=", input_features[sample_id, classifier.tree_.feature[node_id]])
        result.append(explanations)

    return result

# Main
if __name__ == "__main__":

    # Importiamo il dataset e visualizziamo le informazioni
    dataset = pd.read_csv("mushroom.csv", index_col=0)
    print("Informazioni sul dataset")
    print(dataset.info())  # 8124 righe, 23 colonne

    # Notiamo che tutte le colonne sono di tipo object e solo la colonna "stalk-root" presenta valori mancanti

    # Andiamo a correggere i tipi di dato
    fix_dataset_types(dataset)

    # Andiamo a correggere i valori mancanti
    fix_stalk_root_missing_values(dataset)

    # Controlliamo se ci sono valori duplicati
    print("Valori duplicati nel dataset: ", dataset.duplicated().sum())  # 0

    # Non ci sono valori duplicati, quindi proseguiamo

    # Controlliamo se il dataset è bilanciato
    dataset_balancement(dataset)

    # Il dataset è già molto bilanciato, quindi proseguiamo

    # Andiamo a visualizzare gli istogrammi delle varie feature per capire come sono distribuiti i valori
    produce_histograms(dataset)

    # Da questi istogrammi notiamo che alcune colonne hanno bassissima varianza, e quindi non hanno molta potenza
    # predittiva. Andiamo quindi a rimuoverle
    dataset = dataset.drop(["gill-attachment", "veil-type", "veil-color"], axis=1)

    # Inoltre notiamo come alcune colonne sono praticamente identiche, andiamo a rimuovere ogni volta una delle due
    # e a rinominare l'altra
    dataset = fix_equal_columns(dataset)

    # Creiamo un test set e un training set
    X = dataset.drop("poisonous", axis=1)
    y = dataset["poisonous"]
    y = y.replace(["e", "p"], [0, 1]).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Andiamo a codificare le variabili categoriche in modo da poterle utilizzare nei nostri modelli,
    # sia per il training set che per il test set utilizzando il OneHotEncoding
    encoded_dataset_train = encode_variables(X_train)
    encoded_dataset_train = encoded_dataset_train.astype(int)

    encoded_dataset_test = encode_variables(X_test)
    encoded_dataset_test["cap-shape_c"] = 0
    encoded_dataset_test = encoded_dataset_test.astype(int)

    # Il nostro dataset è ora pronto per essere utilizzato nei modelli di machine learning
    # Testeremo Naive Bayes, Decision Tree, Random Forest, KNN e SVM
    models = {"Naive_Bayes": GaussianNB(), "Decision_Tree": DecisionTreeClassifier(),
              "Random_Forest": RandomForestClassifier(), "KNN": KNeighborsClassifier(), "SVM": SVC(kernel="rbf")}

    # Andiamo ad addestrare i modelli
    train_models(models, encoded_dataset_train, y_train)

    # Andiamo a testare i modelli
    test_models(models, encoded_dataset_test, y_test)

    # I modelli con le performance migliori sono Decision Tree e Naive Bayes, ma hanno comunque performance poco
    # desiderabili. Andiamo quindi a fare un fine tuning dei parametri utilizzando GridSearchCV
    tuned_models = {"Tuned_Naive_Bayes": naive_bayes_fine_tuning(encoded_dataset_train, y_train),
                    "Tuned_Decision_Tree": decision_tree_fine_tuning(encoded_dataset_train, y_train)}

    # Andiamo a testare i modelli con i parametri ottimizzati
    test_models(tuned_models, encoded_dataset_test, y_test)

    # TODO ricontrollare ed eventualmente rimuovere
    # Curiosamente il Decision Tree non ha migliorato le sue performance, mentre Naive Bayes ha avuto un miglioramento
    # leggero. Nonostante questo però, Decision Tree ha una Recall sensibilmente migliore rispetto a Naive Bayes, e
    # considerando che in questo caso è più importante avere una bassa percentuale di falsi negativi, Decision Tree
    # risulta essere il modello migliore. Inoltre Decision Tree ha una explainability maggiore rispetto a Naive Bayes,
    # rendendolo il modello migliore per questo caso specifico.

    # Andiamo a visualizzare il grafico dell'albero di decisione
    plot_tree_graph(tuned_models["Tuned_Decision_Tree"], encoded_dataset_train.columns)

    # Salviamo il modello
    with open("Decision_Tree.pickle", "wb") as f:
        pickle.dump(tuned_models["Tuned_Decision_Tree"], f)

    # column_names = encoded_dataset_test.columns.to_list()
    # result = get_decision_explanation(tuned_models["Tuned_Decision_Tree"], encoded_dataset_test.iloc[123].to_numpy(), column_names)
    # print(result)
