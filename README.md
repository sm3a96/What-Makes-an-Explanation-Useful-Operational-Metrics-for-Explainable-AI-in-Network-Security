# What Makes an Explanation Useful in Network Security? Towards Operational Evaluation Metrics for Explainable AI in Network Intrusion Detection
 We are proposing a rigorous evaluation framework that decomposes XAI quality into three distinct quantifiable metrics (Explanation Power, Actionability, Explanation of Accuracy)  .

Our work give a clear, practical tests so teams can pick explainers that are true, useful, and actionable in real IDS work.
Thia repo for XAI in intrusion detection and XAI Evalaution Metrics for XAI methods (SHAP and LIME).  
We test how useful explanations are in real security work.

## This Research about: 
We ask three simple questions:
- Can explanations replicate the model's output? (fidelity/additivity)  
- Do explanations point to things operators can change? (actionability)  
- Do top features really change predictions when we perturb them? (causal faithfulness)

## Main contributions
- We propose three metrics: Explanatory Power, Actionability, Explanation Accuracy.  
- We give a model-aware test of XAI for linear and tree models.  
- We use deletion AUC and rank tests to show causal faithfulness.  
- We run a full study: 4 model families, 2 explainers (SHAP, LIME), 2 datasets.

## Our metrics 
- Explanatory Power  
  Measures how well the explanation rebuilds the model output.  
  We test if the signed feature scores add up to the model score.  
  We report R^2 (higher is better).  

- Actionability  
  Shows how many top features an operator can change.  
  We mark features that can be changed (e.g., firewall rules).  
  The metric is the share of top-k features that are controllable.

- Explanation Accuracy  
  Tests if top features truly change predictions.  
  We remove or change top features and watch the model.  
  We compare to random and least-important baselines.  
  We use deletion AUC and rank tests to measure effect.

## Why this work is different

- IDS-first: We design metrics for network security.  
- Multi-view: We do not use a single score. We use three separate checks.  
- Model-aware: Tests fit both linear and tree models.  
- Causal check: We measure real change via perturbations.  
- Action-focused: We measure if operators can act on the features.  
- Empirical: We test on two real IDS datasets, four model types, and two explainers (SHAP, LIME).  
- Stat-validated: We use paired non-parametric tests to show the differences matter.





## Why use this repo

- Tests XAI for real IDS datasets(CICIDS-2017 and Simargel-2022).  
- Gives measurable metrics.  
- Works with SHAP and LIME methods.  
- Compares 4 model types(DT, RF, LR, XGBoost).  

## General View of Our Framework
![Framework Overview](https://github.com/sm3a96/What-Makes-an-Explanation-Useful-Operational-Metrics-for-Explainable-AI-in-Network-Security/blob/main/Models/Framework_Final.jpg?raw=true)


