# Importer les bibliothèques nécessaires
import streamlit as st
import pickle
import pandas as pd

# Charger le modèle de régression logistique entraîné
with open('logistic_regression_model.pkl', 'rb') as fichier:
    modele = pickle.load(fichier)

# Charger l'encodeur one-hot
with open('one_hot_encoder.pkl', 'rb') as fichier:
    encodeur_one_hot = pickle.load(fichier)

# Définir une fonction pour obtenir les valeurs des fonctionnalités à partir des entrées utilisateur
def obtenir_valeurs_fonctionnalites():
    valeurs_fonctionnalites = {
        'country': st.selectbox('Pays', options=['Kenya', 'Rwanda', 'Tanzania', 'Uganda']),
        'year': st.slider('Année', min_value=2000, max_value=2020, value=2010),
        'uniqueid': st.text_input('ID Unique'),
        'location_type': st.selectbox('Type de Lieu', options=['Urban', 'Rural']),
        'relationship_with_head': st.selectbox('Relation avec le Chef de Famille',
                                               options=['Child', 'Head of Household', 'Other non-relatives',
                                                        'Other relative', 'Parent', 'Spouse']),
        'marital_status': st.selectbox('Statut Marital',
                                       options=['Divorced/Seperated', 'Dont know', 'Married/Living together',
                                                'Single/Never Married', 'Widowed']),
        'education_level': st.selectbox('Niveau d\'Éducation',
                                        options=['No formal education', 'Other/Dont know/RTA', 'Primary education',
                                                 'Secondary education', 'Tertiary education',
                                                 'Vocational/Specialised training']),
        'job_type': st.selectbox('Type d\'Emploi', options=['Dont Know/Refuse to answer', 'Farming and Fishing',
                                                      'Formally employed Government', 'Formally employed Private',
                                                      'Government Dependent', 'Informally employed', 'No Income',
                                                      'Other Income', 'Remittance Dependent', 'Self employed']),
        'cellphone_access': st.selectbox('Accès au Téléphone', options=['Yes', 'No']),
        'gender_of_respondent': st.selectbox('Sexe', options=['Male', 'Female']),
        'age_of_respondent': st.number_input('Âge du Répondant', min_value=0, max_value=120, value=30),
        'household_size': st.number_input('Taille du Ménage', min_value=1, max_value=50, value=5)
    }

    # Encoder les caractéristiques binaires catégorielles
    valeurs_fonctionnalites['cellphone_access'] = 1 if valeurs_fonctionnalites['cellphone_access'] == 'Yes' else 0
    valeurs_fonctionnalites['gender_of_respondent'] = 1 if valeurs_fonctionnalites['gender_of_respondent'] == 'Male' else 0

    return valeurs_fonctionnalites

# Fonction principale pour exécuter l'application
def main():
    st.title("Application de Prédiction d'Inclusion Financière")
    st.write("Entrez les détails pour prédire l'inclusion financière (possession d'un compte bancaire).")

    # Obtenir les valeurs des fonctionnalités à partir de l'entrée utilisateur
    valeurs_fonctionnalites = obtenir_valeurs_fonctionnalites()

    # Bouton pour effectuer la prédiction
    if st.button('Prédire'):
        # Créer un DataFrame à partir des valeurs des fonctionnalités
        donnees_entrees = pd.DataFrame([valeurs_fonctionnalites])

        # Colonnes catégorielles à encoder
        colonnes_categorielles = ['country', 'year', 'uniqueid', 'location_type', 'relationship_with_head',
                                  'marital_status', 'education_level', 'job_type']

        # Transformer les données d'entrée
        entrees_encodees = encodeur_one_hot.transform(donnees_entrees[colonnes_categorielles]).toarray()
        colonnes_encodees = encodeur_one_hot.get_feature_names_out(colonnes_categorielles)
        df_encode = pd.DataFrame(entrees_encodees, columns=colonnes_encodees)

        # Combiner avec d'autres caractéristiques
        donnees_entrees_encodees = pd.concat([donnees_entrees.drop(colonnes_categorielles, axis=1), df_encode], axis=1)

        # Assurez-vous que toutes les colonnes sont présentes dans le même ordre que lors de l'entraînement
        colonnes_entrainement = modele.feature_names_in_
        donnees_entrees_encodees = donnees_entrees_encodees.reindex(columns=colonnes_entrainement, fill_value=0)

        # Prédire en utilisant le modèle chargé
        prediction = modele.predict(donnees_entrees_encodees)
        probabilite_prediction = modele.predict_proba(donnees_entrees_encodees)

        # Afficher le résultat de la prédiction
        if prediction[0] == 1:
            st.success(f"Le modèle prédit : Compte Bancaire Possédé (Probabilité : {probabilite_prediction[0][1]:.2f})")
        else:
            st.error(f"Le modèle prédit : Pas de Compte Bancaire (Probabilité : {probabilite_prediction[0][0]:.2f})")

if __name__ == "__main__":
    main()
