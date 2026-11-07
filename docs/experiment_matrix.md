
# Experiment Matrix – Track E (G11)

## 1. Experimental Overview

This experiment matrix defines seven experiments (E1–E7) for evaluating lexical, semantic, and hybrid recommendation approaches.

All experiments follow a shared warm-start evaluation protocol to ensure fair comparisons across models.

## 2. Shared Evaluation Protocol

| Component | Configuration |
|---|---|
| Training split | 427,957 interactions |
| Validation split | 41,284 interactions |
| Test split | 42,595 interactions |
| Warm candidate universe | 22,753 items |
| Evaluable users | 11,850 users |
| Relevant user-item pairs | 24,707 pairs |
| Positive relevance | Rating >= 4 |
| Recommendation cutoff | Top-10 |
| Primary metrics | Recall@10, NDCG@10 |
| Supporting metric | HitRate@10 |
| Random seed | 42 |

The official temporal split is preserved without reshuffling. Training data is used to fit models and build user profiles, while validation data is used for hyperparameter tuning and model selection. The test split is reserved for the final evaluation in Week 13.

For each user, all items already present in their training history are excluded from the candidate set, regardless of rating.

## 3. Experiment Matrix

| ID | Model / Variant | Configuration / Search Space | Purpose |
|---|---|---|---|
| E1 | Popularity | Positive training interaction count (rating >= 4); no tuning | Establishes the popularity baseline |
| E2 | Biased Matrix Factorization (MF) | 16 configurations: factors {32,64}; regularization {0.02,0.05}; learning rate {0.005,0.01}; epochs {10,20} | Establishes the collaborative filtering baseline |
| E3 | TF-IDF CBF | max_features {10k,20k,all}; ngram_range {(1,1),(1,2)}; min_df {2,5} | Evaluates lexical content-based recommendation |
| E4 | Embedding CBF | Configured semantic encoders: MiniLM-L6-v2 and MPNet-base; maximum sequence length 256 | Evaluates semantic content-based recommendation |
| E5 | TF-IDF + Popularity | Content weight alpha {0.25,0.50,0.75}; popularity weight 1-alpha | Evaluates lexical hybrid recommendation |
| E6 | Embedding + Popularity | Content weight alpha {0.25,0.50,0.75}; popularity weight 1-alpha | Evaluates semantic hybrid recommendation |
| E7 | TF-IDF CBF vs. Embedding CBF by user activity | Low (<=5), Medium (6-8), High (>=9) training interactions | Compares lexical and semantic performance across user activity groups |

## 4. Evaluation Metrics

All experiments use the same evaluation metrics:

- Recall@10: Measures the proportion of relevant items retrieved in the top-10 recommendations.
- NDCG@10: Measures ranking quality while assigning higher importance to relevant items ranked near the top.
- HitRate@10: Measures the proportion of users receiving at least one relevant item in their top-10 recommendations. This is reported as a supporting metric.

Recall@10 and NDCG@10 are the primary metrics for model selection and evaluation.

## 5. Hybrid Scoring Configuration

For E5 and E6, content scores and popularity scores are separately min-max normalized over each user's unseen warm candidate items.

The combined score is defined as:

s(u,i) = alpha * s_content(u,i) + (1-alpha) * s_pop(i)

For E5, the content component uses TF-IDF scores. For E6, it uses cosine similarity from semantic embeddings.

If a user has no usable content profile, the content score is set to zero.

## 6. User Activity Groups (E7)

User activity groups are defined using the number of training interactions:

| Group | Training Interactions | Evaluable Users |
|---|---|---:|
| Low | <= 5 | 6,295 |
| Medium | 6-8 | 2,365 |
| High | >= 9 | 3,190 |
| Total | - | 11,850 |

These groups are used only for the separate RQ2 analysis and do not filter users in the main evaluation.

## 7. Model Selection and Reproducibility

- Hyperparameter tuning is performed using the validation split only.
- The configuration with the highest validation NDCG@10 is selected.
- Ties are broken by higher Recall@10, followed by the lower grid-combination index.
- Random seed 42 is used wherever randomness occurs.
- Each experimental run records its configuration, parameters, Git commit, software environment, validation metrics, number of evaluated users, and runtime.
- The test split is not used for tuning or model selection.

## 8. Implementation Status

The following table distinguishes existing implementation evidence from planned configurations in the repository snapshot.

| Experiment | Repository Evidence | Status |
|---|---|---|
| E1 | Popularity implementation and demonstration notebook | Baseline implementation available |
| E2 | MF implementation and mf_search.json | Baseline available; full 16-configuration grid not evidenced |
| E3 | TF-IDF implementation and validation metrics | Baseline evaluated; 2 grid combinations recorded |
| E4 | Semantic embedding implementation and validation metrics | MiniLM baseline evaluated; full encoder search not evidenced |
| E5 | hybrid_search.json | Configuration available; hybrid implementation not evidenced |
| E6 | hybrid_embedding_search.json | Configuration available; hybrid implementation not evidenced |
| E7 | User groups implementation and validation metrics | Activity-group analysis available |

The current grid runner does not register MF or hybrid model builders. Therefore, the existence of their configuration files does not establish that their full parameter searches have been executed.

## 9. Research Questions

- RQ1: How does semantic content-based recommendation compare with lexical content-based recommendation, and how do their hybrid variants perform?
- RQ2: How does user activity level affect the comparative performance of lexical and semantic recommendation approaches?
