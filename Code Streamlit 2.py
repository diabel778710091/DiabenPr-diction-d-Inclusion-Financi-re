# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Charger les données
file_path = 'Financial_inclusion_dataset.csv'
data = pd.read_csv(file_path)

# Afficher des informations générales sur l'ensemble de données
print(data.info())

# Gérer les valeurs manquantes en supprimant les lignes incomplètes
data = data.dropna()

# Vérifier les valeurs manquantes après la suppression
print(data.isnull().sum())

# Supprimer les doublons, s'ils existent
data = data.drop_duplicates()

# Afficher les premières lignes de l'ensemble de données
print(data.head(1999))

# Définir la liste des colonnes catégorielles
colonnes_categorielles = [
    'country', 'year', 'uniqueid', 'location_type', 'relationship_with_head',
    'marital_status', 'education_level', 'job_type'
]

# Créer un OneHotEncoder pour les colonnes catégorielles
encodeur_one_hot = OneHotEncoder(drop='first', handle_unknown='ignore')
donnees_encodees = encodeur_one_hot.fit_transform(data[colonnes_categorielles]).toarray()
colonnes_encodees = encodeur_one_hot.get_feature_names_out(colonnes_categorielles)

# Créer un DataFrame à partir des données encodées
df_encode = pd.DataFrame(donnees_encodees, columns=colonnes_encodees)

# Concaténer le DataFrame original avec le DataFrame encodé
data_encoded = pd.concat([data.drop(colonnes_categorielles, axis=1), df_encode], axis=1)

# Encoder la colonne 'cellphone_access' avec un Label Encoding
data_encoded["cellphone_access"] = data_encoded["cellphone_access"].map({"Yes": 1, "No": 0})
data_encoded["bank_account"] = data_encoded["bank_account"].map({"Yes": 1, "No": 0})
data_encoded["gender_of_respondent"] = data_encoded["gender_of_respondent"].map({"Male": 1, "Female": 0})

# Sélectionner la variable cible et les fonctionnalités
cible = 'bank_account'
fonctionnalites = data_encoded.drop(columns=[cible])

X = fonctionnalites
y = data_encoded[cible]

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle de régression logistique
modele_regression_logistique = LogisticRegression(max_iter=1000)
modele_regression_logistique.fit(X_train, y_train)

# Tester les performances du modèle
y_pred = modele_regression_logistique.predict(X_test)
exactitude = accuracy_score(y_test, y_pred)
print(f"Précision : {exactitude:.2f}")

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Créer la matrice de confusion
matrice_de_confusion = confusion_matrix(y_test, y_pred)
sns.heatmap(matrice_de_confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.show()

# Enregistrer le modèle de régression logistique au format pkl
with open('logistic_regression_model.pkl', 'wb') as fichier:
    pickle.dump(modele_regression_logistique, fichier)

# Enregistrer l'encodeur one-hot dans un fichier
with open('one_hot_encoder.pkl', 'wb') as fichier:
    pickle.dump(encodeur_one_hot, fichier)
