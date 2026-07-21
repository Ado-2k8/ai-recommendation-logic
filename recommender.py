"""
Project 3 - AI Recommendation Logic (Tech Stack Recommender)
DecodeLabs Industrial Training Kit - Batch 2026

Pipeline IPO :
1. Ingestion  -> capter les compétences de l'utilisateur
2. Scoring    -> TF-IDF + similarité cosinus
3. Sorting    -> trier par score décroissant
4. Filtering  -> garder le Top-N
"""

import csv
import math
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich import box

console = Console()


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
    console.print()
    console.print(
        Panel(
            "[bold]Enter your skills one at a time.[/bold]\n"
            "Type [cyan]done[/cyan] when finished (minimum 3 skills required).",
            title="[bold gold3]Tech Stack Recommender[/bold gold3]",
            subtitle="DecodeLabs — Project 3",
            border_style="gold3",
            box=box.ROUNDED,
            padding=(1, 3),
        )
    )
    console.print()

    user_skills = []
    while True:
        skill = Prompt.ask(f"[bold cyan]Skill #{len(user_skills) + 1}[/bold cyan]").strip()

        if skill.lower() == "done":
            if len(user_skills) < 3:
                console.print(
                    f"[bold red]At least 3 skills are required (currently {len(user_skills)}).[/bold red]\n"
                )
                continue
            break

        if skill == "":
            console.print("[yellow]Empty skill, please try again.[/yellow]\n")
            continue

        user_skills.append(skill)
        console.print(f"  [dim]added:[/dim] [green]{skill}[/green]")

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


# --------------------------------------------------------------------------
# ÉTAPE 3 : SIMILARITÉ COSINUS
# --------------------------------------------------------------------------
# Rappel (page 13-16 du PDF) :
#   cos(θ) = (A · B) / (||A|| * ||B||)
#   - A · B      = produit scalaire (dot product) des deux vecteurs
#   - ||A||,||B|| = norme (longueur) de chaque vecteur
#   Score entre 0 (aucun rapport) et 1 (profils parfaitement alignés).
#   Contrairement à la distance euclidienne, la similarité cosinus ignore
#   la magnitude (taille) des vecteurs et se concentre sur leur orientation.


def dot_product(vector_a, vector_b):
    """Produit scalaire entre deux vecteurs de même longueur."""
    return sum(a * b for a, b in zip(vector_a, vector_b))


def magnitude(vector):
    """Norme (longueur) d'un vecteur."""
    return math.sqrt(sum(x ** 2 for x in vector))


def cosine_similarity(vector_a, vector_b):
    """
    Calcule la similarité cosinus entre deux vecteurs.
    Retourne un score entre 0 et 1 (0 = aucune similarité, 1 = identiques).
    """
    mag_a = magnitude(vector_a)
    mag_b = magnitude(vector_b)

    if mag_a == 0 or mag_b == 0:
        # Cas du "Cold Start" (page 20) : vecteur nul -> similarité nulle
        return 0.0

    return dot_product(vector_a, vector_b) / (mag_a * mag_b)


def score_all_jobs(user_vector, jobs, vocabulary, idf):
    """
    Calcule le score de similarité cosinus entre le profil utilisateur
    et chaque métier du dataset.
    Retourne une liste de dicts : [{"job_role": ..., "score": ...}, ...]
    """
    results = []
    for job in jobs:
        job_vector = build_tfidf_vector(job["skills"], vocabulary, idf)
        score = cosine_similarity(user_vector, job_vector)
        results.append({"job_role": job["job_role"], "score": score})
    return results


# --------------------------------------------------------------------------
# ÉTAPE 4 : SORTING & FILTERING
# --------------------------------------------------------------------------
# Rappel (page 17 et 19 du PDF) :
#   Step 3 - Sorting  : trier les scores par ordre décroissant
#   Step 4 - Filtering: ne garder que le Top-N (ici Top-3) pour éviter
#                       le "choice overload" de l'utilisateur


def sort_by_score(results):
    """
    Trie les résultats par score de similarité décroissant.
    """
    return sorted(results, key=lambda r: r["score"], reverse=True)


def top_n_recommendations(results, n=3):
    """
    Filtre : ne garde que les N métiers les plus pertinents.
    Suppose que `results` est déjà trié (voir sort_by_score).
    """
    return results[:n]


def render_results_table(top_results):
    """
    Construit un tableau rich à partir du Top-N de recommandations.
    """
    table = Table(
        title="Top Career Matches",
        box=box.ROUNDED,
        border_style="gold3",
        header_style="bold white on grey15",
        title_style="bold gold3",
        show_lines=False,
    )
    table.add_column("Rank", justify="center", style="bold gold3", width=6)
    table.add_column("Job Role", style="bold white")
    table.add_column("Match", justify="right", style="cyan", width=8)
    table.add_column("", width=24)  # barre de progression visuelle

    bar_width = 20
    for rank, result in enumerate(top_results, start=1):
        pct = result["score"] * 100
        filled = round((pct / 100) * bar_width)
        bar = "[green]" + "█" * filled + "[/green]" + "[grey35]" + "░" * (bar_width - filled) + "[/grey35]"
        table.add_row(str(rank), result["job_role"], f"{pct:.1f}%", bar)

    return table


if __name__ == "__main__":
    jobs = load_jobs("raw_skills.csv")

    user_skills = ingest_user_skills()

    with Progress(
        SpinnerColumn(style="gold3"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Building vocabulary...", total=None)
        all_documents = [job["skills"] for job in jobs] + [user_skills]
        vocabulary = build_vocabulary(all_documents)
        time.sleep(0.3)

        progress.update(task, description="Computing TF-IDF weights...")
        idf = compute_idf(all_documents, vocabulary)
        user_vector = build_tfidf_vector(user_skills, vocabulary, idf)
        time.sleep(0.3)

        progress.update(task, description="Scoring jobs with cosine similarity...")
        scores = score_all_jobs(user_vector, jobs, vocabulary, idf)
        time.sleep(0.3)

        progress.update(task, description="Sorting and filtering top matches...")
        sorted_scores = sort_by_score(scores)
        top_3 = top_n_recommendations(sorted_scores, n=3)
        time.sleep(0.3)

    console.print()
    console.print(render_results_table(top_3))
    console.print()
    console.print(
        f"[dim]Profile analyzed: {', '.join(user_skills)} | "
        f"{len(jobs)} job roles compared | {len(vocabulary)} unique skills in vocabulary[/dim]"
    )
    console.print()
