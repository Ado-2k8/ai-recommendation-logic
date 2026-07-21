# Tech Stack Recommender

A content-based recommendation engine that matches a user's skill set to the tech job role it aligns with most closely. Built for **Project 3 — AI Recommendation Logic** of the DecodeLabs Industrial Training Kit (Batch 2026).

The engine ships in two forms that share the same underlying algorithm:

- **`recommender.py`** — an interactive command-line application with a styled terminal interface.
- **`web/index.html`** — a standalone, self-contained web interface with a visual "Match Compass."

Both implementations independently reproduce the same input-process-output pipeline, one in Python and one in JavaScript, so the results are identical for the same input skills.

## Table of contents

- [Concept](#concept)
- [How the algorithm works](#how-the-algorithm-works)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
  - [Command-line version](#command-line-version)
  - [Web version](#web-version)
- [Dataset](#dataset)
- [Design notes](#design-notes)
- [Known limitations](#known-limitations)
- [Possible extensions](#possible-extensions)

## Concept

Recommendation systems generally fall into two families: **collaborative filtering**, which relies on the behavior of similar users, and **content-based filtering**, which relies on the intrinsic attributes of the items themselves. This project uses content-based filtering exclusively, which means it works from the first interaction and requires no historical user data.

Given a small set of skills typed in by the user (for example `Python`, `Docker`, `SQL`), the engine treats each of the sixteen job roles in the dataset as a candidate item, scores every candidate against the user's profile, and returns the three closest matches.

## How the algorithm works

The engine follows a four-stage input-process-output pipeline.

### 1. Ingestion

The user provides a minimum of three skills. Each skill is treated as a raw string token; no external skill taxonomy is required, but consistent naming matters (see [Known limitations](#known-limitations)).

### 2. Vectorization (TF-IDF)

Machines cannot compare the words "Python" or "Docker" directly, so every profile — the user's and each job role's — is converted into a numerical vector inside a shared vocabulary space built from every skill that appears anywhere in the dataset.

Each vector position is weighted using **TF-IDF** (Term Frequency - Inverse Document Frequency):

```
TF(term, doc)  = occurrences of term in doc / total terms in doc
IDF(term)      = log(total documents / documents containing term) + 1
weight(term)   = TF(term, doc) * IDF(term)
```

TF rewards skills that make up a larger share of a given profile. IDF penalizes skills that appear across many job roles (for example "Python," which is common) and rewards skills that are rare and therefore more discriminative (for example "TensorFlow" or "Kubernetes"). This prevents generic, high-frequency skills from dominating the match purely because they appear everywhere.

### 3. Scoring (cosine similarity)

With every profile expressed as a TF-IDF vector, the engine measures how closely the user's vector and each job's vector point in the same direction using **cosine similarity**:

```
cos(θ) = (A · B) / (||A|| * ||B||)
```

Cosine similarity was chosen over Euclidean distance because it is invariant to vector magnitude. A job role with a long list of tagged skills should not automatically score worse than one with a short list; what matters is the *orientation* of the vector, not its length. The resulting score sits between 0 (no shared direction) and 1 (perfectly aligned profiles).

### 4. Sorting and filtering

All sixteen job roles are scored, sorted in descending order of similarity, and truncated to the top three. Returning a short, ranked list avoids overwhelming the user with the full list of candidates.

### The cold start problem

A brand-new profile with zero skills produces a zero vector, and cosine similarity against a zero vector is undefined by division — the engine returns a similarity of `0.0` in that case rather than raising an error. In production, this class of problem is typically bypassed with onboarding surveys, trending/popularity fallbacks, or metadata inference; this project does not implement those fallbacks, since the ingestion step already enforces a three-skill minimum.

## Project structure

```
ai-recommendation-logic/
├── recommender.py       Interactive CLI (Python, rich-based interface)
├── raw_skills.csv        Dataset: job roles and their associated skills
├── web/
│   └── index.html        Standalone web interface (HTML/CSS/JS, no build step)
└── README.md              This file
```

## Getting started

### Command-line version

**Requirements:** Python 3.8+ and the `rich` package.

```bash
pip install rich
python3 recommender.py
```

The application prompts for skills one at a time. Type a skill and press Enter to add it; type `done` once at least three skills have been entered to run the analysis. The terminal displays a live progress indicator while the pipeline runs, followed by a ranked results table with visual match bars.

Example session:

```
Skill #1: Python
Skill #2: Machine Learning
Skill #3: TensorFlow
Skill #4: done

Top Career Matches
 Rank   Job Role          Match
   1    Data Scientist    53.7%
   2    Data Analyst       9.9%
   3    Backend Developer  9.3%
```

### Web version

**Requirements:** any modern web browser. No server, build step, or dependency installation is needed — the dataset and recommendation logic are embedded directly in the HTML file.

Open `web/index.html` directly in a browser, or serve it locally:

```bash
cd web
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

Type a skill into the input field and press Enter to add it as a tag (or click one of the suggested skills below the field). Once at least three skills are added, the "Find my matches" button becomes active. Results appear below with:

- A **Match Compass**, a circular gauge whose needle angle visually represents cos(θ) between the user's vector and the top match — a fully aligned profile points the needle straight up.
- A ranked list of the top three job roles with animated match-strength bars.

## Dataset

`raw_skills.csv` contains sixteen tech job roles spanning five families: Data, Cloud, Cybersecurity, Software Development, and Infrastructure & Operations. Each row lists a job role and a comma-separated list of representative skills, tools, and technologies.

| Family | Job roles |
|---|---|
| Data | Data Scientist, Data Analyst, Data Engineer |
| Cloud | Cloud Developer, Cloud Architect |
| Cybersecurity | Security Administrator, Security Auditor, Cybersecurity Analyst |
| Software Development | Frontend Developer, Backend Developer, FullStack Developer, Software Architect, QA Engineer |
| Infrastructure & Operations | DevOps Engineer, Site Reliability Engineer, System and Network Administrator |

The list was compiled from the DecodeLabs training material and cross-referenced against a regional tech job market reference to keep the skill sets realistic.

## Design notes

The web interface's visual identity is built around the project's actual mechanism rather than generic dashboard styling: the signature element, the Match Compass, is a direct visualization of the cosine angle the algorithm computes, so the interface teaches the same concept the code implements. The palette pairs a dark charcoal-navy background with a gold accent (primary actions, best-match highlighting) and a teal accent (similarity and alignment), avoiding default AI-generated look-alikes. Type is set in Space Grotesk for display headings, Inter for body copy, and IBM Plex Mono for data, tags, and technical labels.

## Known limitations

- **Vocabulary sensitivity.** The matching only works if skill names in the user's input and in the dataset are written consistently. "Web Design" and "Frontend Development" are treated as two unrelated terms, even though a human would recognize them as related. There is no synonym resolution or fuzzy matching.
- **No collaborative signal.** Because the engine is purely content-based, it cannot learn from what similar users have chosen, and it cannot improve its recommendations over time without changes to the dataset itself.
- **Small, static dataset.** Sixteen job roles is enough to demonstrate the pipeline but is not representative of the full tech job market. Adding or removing rows in `raw_skills.csv` immediately changes the vocabulary and, therefore, all TF-IDF weights.

## Possible extensions

- Fuzzy or embedding-based skill matching to resolve near-synonyms.
- A "why this match" breakdown showing which specific skills contributed most to a given score.
- Persisting user profiles to support hybrid (content-based plus collaborative) filtering.
- Expanding the dataset with salary bands and regional demand, as captured in the source job-market reference material.
