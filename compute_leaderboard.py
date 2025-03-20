import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from tqdm import tqdm
from models import Models, ModelCategory
import math
from scipy.optimize import minimize
from scipy.special import expit
from datetime import datetime
import random

def compute_model_ratings(votes_df, bootstraps=100):
    """
    Compute BT ratings, Elo ratings, confidence intervals, and win rates for models.

    Args:
    - votes_df (pd.DataFrame): DataFrame with columns ['preference', 'model1', 'model2'].
    - bootstraps (int): Number of bootstrap iterations for confidence intervals.

    Returns:
    - pd.DataFrame: A DataFrame with each row containing:
        - Model name
        - BT rating (median and CI)
        - Elo rating (median and CI)
        - Win rate (with CI)
    """
    def compute_pairwise_significance(df, model1, model2, n_bootstraps=1000):
        """
        Compute p-value for the hypothesis that model1 has a win rate > 50% against model2.
        
        Args:
            df: DataFrame with preference data
            model1: First model name (higher ranked)
            model2: Second model name (lower ranked)
            n_bootstraps: Number of bootstrap samples
            
        Returns:
            p_value: p-value for the hypothesis that model1's win rate against model2 is > 50%
            matchup_count: Number of direct comparisons between the two models
        """
        matchups = df[((df['model1'] == model1) & (df['model2'] == model2)) | 
                      ((df['model1'] == model2) & (df['model2'] == model1))].copy()
        
        if len(matchups) == 0:
            return 1.0, 0  # No data, return p=1.0 and count=0
        
        matchups['temp_pref'] = matchups['preference']
        mask = matchups['model1'] == model2
        matchups.loc[mask, 'temp_pref'] = matchups.loc[mask, 'temp_pref'].map({'A': 'B', 'B': 'A', 'tie': 'tie'})
        
        model1_wins = sum(matchups['temp_pref'] == 'A')
        model2_wins = sum(matchups['temp_pref'] == 'B')
        ties = sum(matchups['temp_pref'] == 'tie')
        
        total_decisive = model1_wins + model2_wins
        if total_decisive == 0:
            return 1.0, len(matchups)  # All ties, return p=1.0
        
        observed_win_rate = model1_wins / total_decisive
        
        win_rates = []
        for _ in range(n_bootstraps):
            bootstrap_sample = matchups.sample(n=len(matchups), replace=True)
            
            m1_wins = sum(bootstrap_sample['temp_pref'] == 'A')
            m2_wins = sum(bootstrap_sample['temp_pref'] == 'B')
            total = m1_wins + m2_wins
            
            if total > 0:
                win_rates.append(m1_wins / total)
            else:
                win_rates.append(0.5)  # If no decisive outcomes, use 0.5
        
        p_value = sum(wr <= 0.5 for wr in win_rates) / n_bootstraps
        
        return p_value, len(matchups)


    def compute_bt_ratings(df, SCALE=400, BASE=10, INIT_RATING=1000):
        """Compute Bradley-Terry ratings using logistic regression."""
        if len(df) == 0:
            return {}

        def preprocess_for_bt(df):
            n_rows = len(df)
            model_indices, models = pd.factorize(pd.concat([df["model1"], df["model2"]]))
            matchups = np.column_stack([model_indices[:n_rows], model_indices[n_rows:]])
            
            outcomes = np.full(len(df), 0.5)
            outcomes[df["preference"] == "A"] = 1.0
            outcomes[df["preference"] == "B"] = 0.0
            
            schedule = np.column_stack([matchups, outcomes])
            unique_matchups, weights = np.unique(schedule, return_counts=True, axis=0)
            
            return (
                unique_matchups[:, :2].astype(np.int32),  # matchups
                unique_matchups[:, 2],                    # outcomes
                models.to_list(),                         # model names
                weights.astype(np.float64)                # weights
            )

        def bt_loss_and_grad(ratings, matchups, outcomes, weights, alpha=1.0):
            matchup_ratings = ratings[matchups]
            logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
            probs = expit(logits)
            loss = -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes)) * weights).sum()
            
            matchups_grads = -alpha * (outcomes - probs) * weights
            model_grad = np.zeros_like(ratings)
            np.add.at(
                model_grad,
                matchups[:, [0, 1]],
                matchups_grads[:, None] * np.array([1.0, -1.0], dtype=np.float64)
            )
            return loss, model_grad

        def fit_bt(matchups, outcomes, weights, n_models, alpha, tol=1e-6):
            initial_ratings = np.zeros(n_models, dtype=np.float64)
            result = minimize(
                fun=bt_loss_and_grad,
                x0=initial_ratings,
                args=(matchups, outcomes, weights, alpha),
                jac=True,
                method='L-BFGS-B',
                options={'disp': False, 'maxiter': 100, 'gtol': tol}
            )
            return result['x']

        def scale_and_offset(ratings, models, scale=SCALE, init_rating=INIT_RATING):
            scaled_ratings = (ratings * scale) + init_rating
            return scaled_ratings

        matchups, outcomes, models, weights = preprocess_for_bt(df)
        ratings = fit_bt(matchups, outcomes, weights, len(models), math.log(BASE))
        scaled_ratings = scale_and_offset(ratings, models)
        
        return dict(zip(models, scaled_ratings))

    def compute_elo_ratings(df, K=4, INIT=1000):
        """Compute Elo ratings."""
        ratings = defaultdict(lambda: INIT)
        for _, row in df.iterrows():
            m1, m2, pref = row["model1"], row["model2"], row["preference"]
            r1, r2 = ratings[m1], ratings[m2]
            prob1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
            if pref == "A":
                delta = K * (1 - prob1)
            elif pref == "B":
                delta = K * (-prob1)
            else:
                delta = K * (0.5 - prob1)
            ratings[m1] += delta
            ratings[m2] -= delta
        return ratings

    def compute_confidence_intervals(df, compute_func, rounds):
        """Bootstrap confidence intervals."""
        scores = pd.DataFrame(
            [compute_func(df.sample(frac=1, replace=True)) for _ in tqdm(range(rounds))]
        )
        medians = scores.median()
        lower = scores.quantile(0.025)
        upper = scores.quantile(0.975)
        lower = medians-lower
        upper = upper-medians
        
        ci_strings = [f"-{l:.3f}/+{u:.3f}" for l, u in zip(lower, upper)]
        return pd.DataFrame({"median": medians, "ci": ci_strings})

    def compute_win_rates(df):
        """Compute win rates for each model."""
        total_games = pd.concat([df["model1"], df["model2"]]).value_counts()
        wins = (
            df[df["preference"] == "A"]["model1"]
            .value_counts()
            .add(df[df["preference"] == "B"]["model2"].value_counts(), fill_value=0)
        )
        return (wins / total_games).fillna(0)
    
    def compute_win_rates_with_ci(df, rounds=1000):
        """Compute win rates with 95% confidence intervals using bootstrap."""
        all_models = pd.concat([df["model1"], df["model2"]]).unique()
        
        win_rate_samples = {model: [] for model in all_models}
        
        for _ in tqdm(range(rounds), desc="Computing win rate CIs"):
            bootstrap_sample = df.sample(frac=1, replace=True)
            
            total_games = pd.concat([bootstrap_sample["model1"], bootstrap_sample["model2"]]).value_counts()
            wins = (
                bootstrap_sample[bootstrap_sample["preference"] == "A"]["model1"]
                .value_counts()
                .add(bootstrap_sample[bootstrap_sample["preference"] == "B"]["model2"].value_counts(), fill_value=0)
            )
            win_rates = (wins / total_games).fillna(0)
            
            for model in all_models:
                win_rate_samples[model].append(win_rates.get(model, 0.0))
        
        results = {}
        for model in all_models:
            samples = win_rate_samples[model]
            median = np.median(samples)
            lower = np.percentile(samples, 2.5)
            upper = np.percentile(samples, 97.5)
            results[model] = {
                'win_rate': median,
                'ci_lower': lower,
                'ci_upper': upper
            }
        
        return results

    def compute_lose_rates(df):
        """Compute lose rates for each model, counting 'neither' as a loss."""
        total_games = pd.concat([df["model1"], df["model2"]]).value_counts()
        losses = (
            df[df["preference"] == "B"]["model1"]
            .value_counts()
            .add(df[df["preference"] == "A"]["model2"].value_counts(), fill_value=0)
            .add(df[df["preference"] == "neither"]["model1"].value_counts(), fill_value=0)
            .add(df[df["preference"] == "neither"]["model2"].value_counts(), fill_value=0)
        )
        return (losses / total_games).fillna(0)

    def compute_vote_counts(df):
        """Compute total number of votes for each model."""
        return pd.concat([df["model1"], df["model2"]]).value_counts()
    
    votes_df["preference"] = votes_df["preference"].map(
        {"A": "A", "B": "B", "tie": "tie", "neither": "neither"}
    )
    vote_counts = compute_vote_counts(votes_df)

    votes_df = votes_df[votes_df["preference"].isin(["A", "B", "tie"])]
    
    bt_ratings = compute_bt_ratings(votes_df)
    elo_ratings = compute_elo_ratings(votes_df)
    bt_cis = compute_confidence_intervals(votes_df, compute_bt_ratings, bootstraps)
    elo_cis = compute_confidence_intervals(votes_df, compute_elo_ratings, bootstraps)
    
    win_rates_with_ci = compute_win_rates_with_ci(votes_df, rounds=bootstraps)
    
    win_rates = compute_win_rates(votes_df)
    lose_rates = compute_lose_rates(votes_df)
    
    results = pd.DataFrame(
        {
            "Model": list(bt_ratings.keys()),
            "BT Rating": [round(rating) for rating in bt_ratings.values()],
            "BT CI (95%)": [f"-{round(float(ci.split('/')[0][1:]))}/+{round(float(ci.split('/')[1]))}" 
                           for ci in [bt_cis["ci"].get(model, "-0.0/+0.0") for model in bt_ratings.keys()]],
            "Elo Rating": [round(elo_ratings.get(model, 1000)) for model in bt_ratings.keys()],
            "Elo CI (95%)": [f"-{round(float(ci.split('/')[0][1:]))}/+{round(float(ci.split('/')[1]))}"
                            for ci in [elo_cis["ci"].get(model, "-0.0/+0.0") for model in bt_ratings.keys()]],
            "Win Rate": [round(win_rates_with_ci.get(model, {'win_rate': 0.0})['win_rate'], 3) for model in bt_ratings.keys()],
            "Win Rate CI (95%)": [f"{round(win_rates_with_ci.get(model, {'ci_lower': 0.0})['ci_lower'], 3)}-{round(win_rates_with_ci.get(model, {'ci_upper': 0.0})['ci_upper'], 3)}" 
                                 for model in bt_ratings.keys()],
            "Lose Rate": [round(lose_rates.get(model, 0.0), 3) for model in bt_ratings.keys()],
            "Battle Count": [vote_counts.get(model, 0) for model in bt_ratings.keys()]
        }
    ).sort_values("Elo Rating", ascending=False)
    
    results['P-value vs Next'] = None
    results['Matchups vs Next'] = None
    
    for i in range(len(results) - 1):
        higher_model = results.iloc[i]['Model']
        lower_model = results.iloc[i + 1]['Model']
        p_value, matchup_count = compute_pairwise_significance(votes_df, higher_model, lower_model)
        
        if p_value < 0.001:
            p_value_str = f"{p_value:.3f}***"
        elif p_value < 0.01:
            p_value_str = f"{p_value:.3f}**"
        elif p_value < 0.05:
            p_value_str = f"{p_value:.3f}*"
        else:
            p_value_str = f"{p_value:.3f}"
            
        results.loc[results.index[i], 'P-value vs Next'] = p_value_str
        results.loc[results.index[i], 'Matchups vs Next'] = matchup_count
    
    return results

def generate_synthetic_preferences(model_names, n=1000):
    """
    Generate synthetic preferences data for pairs of models, with different model strengths.
    
    Args:
        model_names (list): List of model names
        n (int): Number of synthetic preferences to generate
    
    Returns:
        pd.DataFrame: DataFrame with columns ['preference', 'model1', 'model2']
    """
    model_strengths = {model: random.uniform(0.3, 0.9) for model in model_names}
    
    model_pairs = [(m1, m2) for m1 in model_names for m2 in model_names if m1 != m2]
    
    synthetic_data = []
    for _ in range(n):
        model1, model2 = random.choice(model_pairs)
        
        strength1 = model_strengths[model1]
        strength2 = model_strengths[model2]
        
        total_strength = strength1 + strength2
        prob_model1 = (strength1 / total_strength) * 0.9
        prob_model2 = (strength2 / total_strength) * 0.9
        
        preference = random.choices(
            ['A', 'B', 'tie', 'neither'],
            weights=[prob_model1, prob_model2, 0.05, 0.05]
        )[0]
        
        synthetic_data.append({
            'preference': preference,
            'model1': model1,
            'model2': model2
        })
    
    return pd.DataFrame(synthetic_data)


if __name__ == "__main__":
    today = datetime.now().strftime("%Y%m%d")
    
    model_names = []
    model_name_mapping = {}
    for model in Models.ALL_MODELS:
        name = model
        if model in Models.MODELS[ModelCategory.RAG_CAPABLE]:
            name += " 📚"
        if model in Models.MODELS[ModelCategory.VISION_CAPABLE]:
            name += " 📷"
        model_names.append(name)
        model_name_mapping[model] = name
    model_name_mapping["google/gemini-2.0-flash"] = "google/gemini-2.0-flash 📚"
    model_name_mapping["google/gemini-flash-1.5"] = "google/gemini-flash-1.5 📚"
    model_name_mapping["meta-llama/llama-3.2-90b-vision-instruct"] = "meta-llama/llama-3.2-90b-vision-instruct 📷"
    
    preferences_df = generate_synthetic_preferences(model_names)
    
    leaderboard_df = compute_model_ratings(preferences_df, bootstraps=1000)
    leaderboard_df['Model'] = leaderboard_df['Model'].apply(lambda x: model_name_mapping.get(x, x))
    leaderboard_df.to_csv(f"leaderboard_{today}.csv", index=False)
    print(leaderboard_df)
    # import pdb; pdb.set_trace()
