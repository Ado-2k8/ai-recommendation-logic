"""
Project 3 - AI Recommendation Logic (Tech Stack Recommender)
DecodeLabs Industrial Training Kit - Batch 2026

Pipeline IPO :
1. Ingestion  -> capter les compétences de l'utilisateur
2. Scoring    -> TF-IDF + similarité cosinus (en cours)
3. Sorting    -> trier par score décroissant (à venir)
4. Filtering  -> garder le Top-N (à venir)
"""

import csv
import math


def load_jobs(csv_path):
    """
    Charge le dataset des métiers depuis raw_skills.csv.
    Retourne une liste de dicts : [{"job_role": ..., "skills": [...]}, ...]
    """
    jobs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skills = [s.strip() for s in row["skills"].split(",")]
            jobs.append({"job_role": row["job_role"], "skills": skills})
    return jobs


def ingest_user_skills():
    """
    Étape 1 : Ingestion.
    Demande à l'utilisateur de saisir au minimum 3 compétences.
    Retourne une liste de compétences (strings, nettoyées).
    """
    print("=== Tech Stack Recommender ===")
    print("Entrez vos compétences une par une (minimum 3).")
    print("Tapez 'fin' pour terminer la saisie.\n")

    user_skills = []
    while True:
        skill = input(f"Compétence #{len(user_skills) + 1} : ").strip()

        if skill.lower() == "fin":
            if len(user_skills) < 3:
                print(f"⚠️  Il faut au moins 3 compétences (actuellement {len(user_skills)}). Continuez.\n")
                continue
            break

        if skill == "":
            print("⚠️  Compétence vide, réessayez.\n")
            continue

        user_skills.append(skill)

    return user_skills


# --------------------------------------------------------------------------
# ÉTAPE 2 : VECTORISATION TF-IDF
# --------------------------------------------------------------------------
# Rappel (page 9-12 du PDF) :
#   - Vocabulaire : toutes les compétences uniques (métiers + utilisateur)
#   - TF  = (occurrences du terme dans le "document") / (nb total de termes)
#   - IDF = log(nb total de documents / nb de documents contenant le terme)
#   - Poids TF-IDF = TF * IDF
# Chaque "document" ici = la liste de compétences d'un métier (ou du profil
# utilisateur). Comme chaque compétence apparaît une seule fois par métier,
# TF = 1 / (nombre de compétences du métier).


def build_vocabulary(documents):
    """
    Construit le vocabulaire (liste triée de compétences uniques)
    à partir d'une liste de documents (chaque document = liste de compétences).
    """
    vocab = set()
    for doc in documents:
        vocab.update(doc)
    return sorted(vocab)


def compute_tf(document, vocabulary):
    """
    Calcule le vecteur TF (Term Frequency) d'un document.
    Retourne un dict {terme: score_tf}.
    """
    total_terms = len(document)
    tf = {}
    for term in vocabulary:
        count = document.count(term)
        tf[term] = count / total_terms if total_terms > 0 else 0
    return tf


def compute_idf(documents, vocabulary):
    """
    Calcule l'IDF (Inverse Document Frequency) de chaque terme du vocabulaire
    sur l'ensemble des documents.
    Retourne un dict {terme: score_idf}.
    """
    n_docs = len(documents)
    idf = {}
    for term in vocabulary:
        docs_with_term = sum(1 for doc in documents if term in doc)
        # +1 au dénominateur pour éviter la division par zéro (lissage)
        idf[term] = math.log(n_docs / (docs_with_term + 1)) + 1
    return idf


def build_tfidf_vector(document, vocabulary, idf):
    """
    Construit le vecteur TF-IDF final d'un document (liste de floats,
    dans l'ordre du vocabulaire).
    """
    tf = compute_tf(document, vocabulary)
    return [tf[term] * idf[term] for term in vocabulary]


if __name__ == "__main__":
    # Test des étapes 1 et 2 : ingestion + vectorisation TF-IDF
    jobs = load_jobs("raw_skills.csv")
    print(f"{len(jobs)} métiers chargés depuis le dataset.\n")

    user_skills = ingest_user_skills()
    print("\n✅ Compétences saisies :", user_skills)

    # Tous les documents = compétences de chaque métier + profil utilisateur
    all_documents = [job["skills"] for job in jobs] + [user_skills]

    vocabulary = build_vocabulary(all_documents)
    print(f"\n📖 Taille du vocabulaire : {len(vocabulary)} termes uniques")

    idf = compute_idf(all_documents, vocabulary)

    user_vector = build_tfidf_vector(user_skills, vocabulary, idf)
    print(f"\n🔢 Vecteur TF-IDF utilisateur (premiers termes non nuls) :")
    for term, weight in zip(vocabulary, user_vector):
        if weight > 0:
            print(f"   {term}: {round(weight, 4)}")
