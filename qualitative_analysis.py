import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, creating it if necessary."""
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        raise

def compute_conditional_probabilities(data, predictor_col, target_col, threshold):
    """Compute conditional probability P(Target=1 | Predictor>=threshold)."""
    # Cases where predictor is high (above threshold)
    high_predictor = data[data[predictor_col] >= threshold]
    
    print(f"Cases with {predictor_col} >= {threshold}: {len(high_predictor)}")
    if len(high_predictor) == 0:
        print(f"No cases where {predictor_col} >= {threshold}. Returning probability 0.")
        return 0.0

    # Probability P(Target=1 | Predictor>=threshold)
    prob = (high_predictor[target_col] == 1).mean()
    print(f"P({target_col}=1 | {predictor_col}>={threshold}) = {prob}")
    return prob if not pd.isna(prob) else 0.0

def sdt_combined_analysis(tier_2_folder="tier_2_outputs", output_base_folder="tier_2_outputs"):
    """
    
    Notee: decided to discontinue this since it didnt work
    Compute conditional probabilities:
    1. P(Funniness=low | Var=High) for each Tier 2 variable.
    2. P(llm_humor_classification=no | Var=High) for each Tier 2 variable.
    Analysis are performed on Reddit (overall) and ColBERT datasets.

    """
    low_funniness_output_folder = os.path.join(output_base_folder, "prob_low_funniness")
    llm_classification_no_output_folder = os.path.join(output_base_folder, "prob_llm_classification_no")
    ensure_directory_exists(low_funniness_output_folder)
    ensure_directory_exists(llm_classification_no_output_folder)

    llms = ["grok", "deepseek", "chatgpt", "llama"]
    datasets = ["Reddit", "ColBERT"]
    predictors = ["appropriateness", "offensiveness", "confidence", "clarity", "originality"]

    low_funniness_results = []
    llm_classification_no_results = []

    for llm in llms:
        print(f"\nProcessing dataset for {llm}")
        reddit_dataframes = []
        colbert_dataframes = []

        colbert_file = os.path.join(tier_2_folder, f"{llm}_colbert_tier_2_quantitative.csv")
        if os.path.exists(colbert_file):
            try:
                df = pd.read_csv(colbert_file)
                df["dataset"] = "ColBERT"
                colbert_dataframes.append(df)
            except Exception as e:
                print(f"Error processing {colbert_file}: {e}")

        for file in os.listdir(tier_2_folder):
            if file.startswith(f"{llm}_") and file.endswith("_tier_2_quantitative.csv") and "colbert" not in file:
                file_path = os.path.join(tier_2_folder, file)
                try:
                    df = pd.read_csv(file_path)
                    df["dataset"] = "Reddit"
                    reddit_dataframes.append(df)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        dataset_dfs = {
            "Reddit": pd.concat(reddit_dataframes, ignore_index=True) if reddit_dataframes else None,
            "ColBERT": pd.concat(colbert_dataframes, ignore_index=True) if colbert_dataframes else None
        }

        for dataset_name in datasets:
            data = dataset_dfs[dataset_name]
            if data is None or data.empty:
                print(f"No {dataset_name} data for {llm}. Skipping.")
                continue

            required_cols = predictors + ["llm_classification", "true_label", "funniness"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                print(f"Missing columns for {llm} in {dataset_name}: {missing_cols}. Skipping.")
                continue

            # Create a copy to avoid SettingWithCopyWarning
            data = data.copy()

            for var in predictors + ["funniness"]:
                try:
                    data.loc[:, var] = pd.to_numeric(data[var], errors='coerce')
                    nan_count = data[var].isna().sum()
                    if nan_count > 0:
                        print(f"Warning: {var} contains {nan_count} NaN values. Filling with median.")
                        data.loc[:, var] = data[var].fillna(data[var].median())
                    if data[var].isna().all():
                        print(f"Error: {var} contains only NaN after conversion. Filling with 0.")
                        data.loc[:, var] = 0
                except Exception as e:
                    print(f"Error converting {var} to numeric: {e}. Skipping.")
                    continue

            # Prepare target variables with dynamic threshold for low_funniness
            if dataset_name == "Reddit":
                threshold = 2  # Median funniness is often 1 or 2 for Reddit
            else:
                threshold = 3  # Median funniness is higher for ColBERT (e.g., 6)

            data.loc[:, "low_funniness"] = (data["funniness"] <= threshold).astype(int)
            data.loc[:, "llm_classification_no"] = (data["llm_classification"].apply(lambda x: 1 if x == "YES" or x == 1 else 0) == 0).astype(int)
            data.loc[:, "true_label"] = data["true_label"].apply(lambda x: True if x == "True" or x == 1 else False)

            # Log distribution before dropping NaNs
            print(f"\nPre-dropna class distribution for {llm} in {dataset_name}:")
            print(f"funniness distribution: {data['funniness'].describe().to_dict()}")
            print(f"low_funniness (threshold <= {threshold}): {data['low_funniness'].value_counts().to_dict()}")
            print(f"llm_classification: {data['llm_classification'].value_counts().to_dict()}")
            print(f"llm_classification_no: {data['llm_classification_no'].value_counts().to_dict()}")
            print(f"true_label: {data['true_label'].value_counts().to_dict()}")

            # Drop NaNs for critical columns only
            sdt_data = data.dropna(subset=["llm_classification", "true_label", "funniness", "low_funniness", "llm_classification_no"]).copy()
            for col in predictors:
                sdt_data.loc[:, col] = sdt_data[col].fillna(sdt_data[col].median())

            if len(sdt_data) < 10:
                print(f"Insufficient data for {llm} in {dataset_name} ({len(sdt_data)} samples). Skipping.")
                continue

            # Log distribution after dropping NaNs
            print(f"\nPost-dropna class distribution for {llm} in {dataset_name}:")
            print(f"low_funniness: {sdt_data['low_funniness'].value_counts().to_dict()}")
            print(f"llm_classification_no: {sdt_data['llm_classification_no'].value_counts().to_dict()}")
            print(f"true_label: {sdt_data['true_label'].value_counts().to_dict()}")

            # Check for single-class variables
            if sdt_data["true_label"].nunique() < 2:
                print(f"Error: true_label has only one class after dropna for {llm} in {dataset_name}: {sdt_data['true_label'].unique()}. Skipping.")
                continue

            if sdt_data["low_funniness"].nunique() < 2:
                print(f"Error: low_funniness has only one class for {llm} in {dataset_name}: {sdt_data['low_funniness'].unique()}. Skipping low funniness analysis.")
                low_funniness_skipped = True
            else:
                low_funniness_skipped = False

            if sdt_data["llm_classification_no"].nunique() < 2:
                print(f"Error: llm_classification_no has only one class for {llm} in {dataset_name}: {sdt_data['llm_classification_no'].unique()}. Skipping llm_classification_no analysis.")
                llm_classification_no_skipped = True
            else:
                llm_classification_no_skipped = False

            # Compute P(Funniness=low | Var=High)
            if not low_funniness_skipped:
                for predictor in predictors:
                    print(f"\nComputing P(Funniness=low | {predictor}=High) for {llm} in {dataset_name}")
                    try:
                        # Use median as threshold for "high"
                        threshold = sdt_data[predictor].median()
                        print(f"Threshold for {predictor}: {threshold}")
                        prob = compute_conditional_probabilities(
                            sdt_data, predictor, "low_funniness", threshold
                        )
                        low_funniness_results.append({
                            "llm": llm,
                            "dataset": dataset_name,
                            "predictor": predictor,
                            "prob_funniness_low_given_high": prob,
                            "valid_entries": len(sdt_data)
                        })
                        sdt_df = pd.DataFrame([low_funniness_results[-1]])
                        sdt_csv_path = os.path.join(low_funniness_output_folder, f"low_funniness_prob_{predictor}_{llm}_{dataset_name}.csv")
                        sdt_df.to_csv(sdt_csv_path, index=False)
                        print(f"Saved P(Funniness=low | {predictor}=High) results to {sdt_csv_path}")
                    except ValueError as e:
                        print(f"Error in P(Funniness=low | {predictor}=High) for {llm} in {dataset_name}: {e}")

            # Compute P(llm_humor_classification=no | Var=High)
            if not llm_classification_no_skipped:
                for predictor in predictors:
                    print(f"\nComputing P(llm_humor_classification=no | {predictor}=High) for {llm} in {dataset_name}")
                    try:
                        threshold = sdt_data[predictor].median()
                        print(f"Threshold for {predictor}: {threshold}")
                        prob = compute_conditional_probabilities(
                            sdt_data, predictor, "llm_classification_no", threshold
                        )
                        llm_classification_no_results.append({
                            "llm": llm,
                            "dataset": dataset_name,
                            "predictor": predictor,
                            "prob_no_given_high": prob,
                            "valid_entries": len(sdt_data)
                        })
                        sdt_df = pd.DataFrame([llm_classification_no_results[-1]])
                        sdt_csv_path = os.path.join(llm_classification_no_output_folder, f"llm_classification_no_prob_{predictor}_{llm}_{dataset_name}.csv")
                        sdt_df.to_csv(sdt_csv_path, index=False)
                        print(f"Saved P(llm_humor_classification=no | {predictor}=High) results to {sdt_csv_path}")
                    except ValueError as e:
                        print(f"Error in P(llm_humor_classification=no | {predictor}=High) for {llm} in {dataset_name}: {e}")

    # Plotting for Low Funniness
    low_funniness_df = pd.DataFrame(low_funniness_results)
    for dataset_name in datasets:
        if not low_funniness_df.empty:
            ds_df = low_funniness_df[low_funniness_df["dataset"] == dataset_name]
            if not ds_df.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                sns.barplot(data=ds_df, x="llm", y="prob_funniness_low_given_high", hue="predictor", ax=ax1, palette="viridis")
                ax1.set_title(f"P(Funniness=Low | Var=High) ({dataset_name})")
                ax1.set_ylabel("P(Funniness=Low | Var=High)")
                ax1.set_ylim(0, 1)
                ax1.legend(title="Predictor")
                # Second subplot is placeholder for consistency with previous setup
                ax2.set_visible(False)
                plt.tight_layout()
                plot_path = os.path.join(low_funniness_output_folder, f"low_funniness_prob_plot_{dataset_name}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved low funniness probability plot to {plot_path}")
            else:
                print(f"No low funniness probability data to plot for {dataset_name}.")
        else:
            print(f"No low funniness probability results available for plotting.")

    # Plotting for LLM Classification = "NO"
    llm_classification_no_df = pd.DataFrame(llm_classification_no_results)
    for dataset_name in datasets:
        if not llm_classification_no_df.empty:
            ds_df = llm_classification_no_df[llm_classification_no_df["dataset"] == dataset_name]
            if not ds_df.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                sns.barplot(data=ds_df, x="llm", y="prob_no_given_high", hue="predictor", ax=ax1, palette="viridis")
                ax1.set_title(f"P(LLM Classification=NO | Var=High) ({dataset_name})")
                ax1.set_ylabel("P(LLM Classification=NO | Var=High)")
                ax1.set_ylim(0, 1)
                ax1.legend(title="Predictor")
                # Second subplot is placeholder for consistency with previous setup
                ax2.set_visible(False)
                plt.tight_layout()
                plot_path = os.path.join(llm_classification_no_output_folder, f"llm_classification_no_prob_plot_{dataset_name}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved llm_classification = NO probability plot to {plot_path}")
            else:
                print(f"No llm_classification = NO probability data to plot for {dataset_name}.")
        else:
            print(f"No llm_classification = NO probability results available for plotting.")

import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configuration
llms = ['grok', 'deepseek', 'chatgpt', 'llama']
datasets = ['Reddit', 'ColBERT']
variables_all = ['funniness', 'offensiveness', 'originality', 'appropriateness', 'clarity', 'confidence']
variables_no_funniness = ['offensiveness', 'originality', 'appropriateness', 'clarity', 'confidence']
tier_2_folder = 'tier_2_outputs'
output_dir = 'blah'
os.makedirs(output_dir, exist_ok=True)

# Load data
data = {dataset: {llm: [] for llm in llms} for dataset in datasets}
for llm in llms:
    # Reddit files
    reddit_files = [f for f in os.listdir(tier_2_folder) if f.startswith(f'{llm}_') and f.endswith('_tier_2_quantitative.csv') and 'colbert' not in f]
    for file in reddit_files:
        try:
            df = pd.read_csv(os.path.join(tier_2_folder, file))
            data['Reddit'][llm].append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    # ColBERT file
    colbert_file = f'{llm}_colbert_tier_2_quantitative.csv'
    if os.path.exists(os.path.join(tier_2_folder, colbert_file)):
        try:
            df = pd.read_csv(os.path.join(tier_2_folder, colbert_file))
            data['ColBERT'][llm].append(df)
        except Exception as e:
            print(f"Error loading {colbert_file}: {e}")

# Combine Reddit data
for llm in llms:
    if data['Reddit'][llm]:
        data['Reddit'][llm] = pd.concat(data['Reddit'][llm], ignore_index=True)
    else:
        data['Reddit'][llm] = pd.DataFrame()
    if data['ColBERT'][llm]:
        data['ColBERT'][llm] = pd.concat(data['ColBERT'][llm], ignore_index=True)
    else:
        data['ColBERT'][llm] = pd.DataFrame()

# Compute correlations
pb_yes_results = {dataset: pd.DataFrame(index=llms, columns=variables_all) for dataset in datasets}
pb_no_results = {dataset: pd.DataFrame(index=llms, columns=variables_all) for dataset in datasets}
pb_high_results = {dataset: pd.DataFrame(index=llms, columns=variables_no_funniness) for dataset in datasets}
pb_low_results = {dataset: pd.DataFrame(index=llms, columns=variables_no_funniness) for dataset in datasets}

for dataset in datasets:
    for llm in llms:
        df = data[dataset][llm].copy()
        if df.empty:
            print(f"No data for {llm} in {dataset}. Skipping.")
            continue

        # Prepare binary variables
        df['Yes_No'] = df['llm_classification'].apply(lambda x: 1 if x == 'YES' else 0 if x == 'NO' else np.nan)
        df['No_Yes'] = df['llm_classification'].apply(lambda x: 1 if x == 'NO' else 0 if x == 'YES' else np.nan)
        df['High_F'] = df['funniness'].apply(lambda x: 1 if x > 7 else 0)
        df['Low_F'] = df['funniness'].apply(lambda x: 1 if x < 3 else 0)
        df = df.dropna(subset=['Yes_No', 'No_Yes', 'High_F', 'Low_F'] + variables_all)

        if len(df) < 10:
            print(f"Insufficient data for {llm} in {dataset} ({len(df)} rows). Skipping.")
            continue

        # Point-biserial correlations for Yes classification
        for var in variables_all:
            r_pb, _ = pointbiserialr(df['Yes_No'], df[var])
            pb_yes_results[dataset].loc[llm, var] = r_pb if not np.isnan(r_pb) else 0.0

        # Point-biserial correlations for No classification
        for var in variables_all:
            r_pb, _ = pointbiserialr(df['No_Yes'], df[var])
            pb_no_results[dataset].loc[llm, var] = r_pb if not np.isnan(r_pb) else 0.0

        # Point-biserial correlations for High Funniness
        for var in variables_no_funniness:
            r_pb, _ = pointbiserialr(df['High_F'], df[var])
            pb_high_results[dataset].loc[llm, var] = r_pb if not np.isnan(r_pb) else 0.0

        # Point-biserial correlations for Low Funniness
        for var in variables_no_funniness:
            r_pb, _ = pointbiserialr(df['Low_F'], df[var])
            pb_low_results[dataset].loc[llm, var] = r_pb if not np.isnan(r_pb) else 0.0

# Save correlation tables
for dataset in datasets:
    pb_yes_results[dataset].to_csv(os.path.join(output_dir, f'point_biserial_yes_{dataset.lower()}_table.csv'))
    pb_no_results[dataset].to_csv(os.path.join(output_dir, f'point_biserial_no_{dataset.lower()}_table.csv'))
    pb_high_results[dataset].to_csv(os.path.join(output_dir, f'point_biserial_high_funniness_{dataset.lower()}_table.csv'))
    pb_low_results[dataset].to_csv(os.path.join(output_dir, f'point_biserial_low_funniness_{dataset.lower()}_table.csv'))
    print(f"Saved correlation tables for {dataset}")

# Generate heatmaps
for dataset in datasets:
    # Point-biserial heatmap for Yes classification
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pb_yes_results[dataset].astype(float),
        annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f'
    )
    plt.title(f'Point-Biserial Correlations for Yes Humor Classification ({dataset})')
    plt.xlabel('Variables')
    plt.ylabel('LLMs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'point_biserial_yes_{dataset.lower()}.png'), dpi=300)
    plt.close()

    # Point-biserial heatmap for No classification
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pb_no_results[dataset].astype(float),
        annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f'
    )
    plt.title(f'Point-Biserial Correlations for No Humor Classification ({dataset})')
    plt.xlabel('Variables')
    plt.ylabel('LLMs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'point_biserial_no_{dataset.lower()}.png'), dpi=300)
    plt.close()

    # Point-biserial heatmap for High Funniness
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pb_high_results[dataset].astype(float),
        annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f'
    )
    plt.title(f'Point-Biserial Correlations for High Funniness (> 7) ({dataset})')
    plt.xlabel('Variables')
    plt.ylabel('LLMs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'point_biserial_high_funniness_{dataset.lower()}.png'), dpi=300)
    plt.close()

    # Point-biserial heatmap for Low Funniness
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pb_low_results[dataset].astype(float),
        annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f'
    )
    plt.title(f'Point-Biserial Correlations for Low Funniness (< 3) ({dataset})')
    plt.xlabel('Variables')
    plt.ylabel('LLMs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'point_biserial_low_funniness_{dataset.lower()}.png'), dpi=300)
    plt.close()

print(f"Correlation tables and heatmaps saved in {output_dir}")