import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import accuracy_score

# HUMOR_CATEGORIES dictionary
HUMOR_CATEGORIES = {
    "puns": ("I. Linguistic Humor", "Wordplay"),
    "homophonic": ("I. Linguistic Humor", "Wordplay"),
    "paraprosdokians": ("I. Linguistic Humor", "Wordplay"),
    "double_entendres": ("I. Linguistic Humor", "Semantic Shifts"),
    "malapropisms": ("I. Linguistic Humor", "Semantic Shifts"),
    "surreal": ("II. Contextual Humor", "Situational Absurdity"),
    "anti-jokes": ("II. Contextual Humor", "Situational Absurdity"),
    "historical_satire": ("II. Contextual Humor", "Cultural References"),
    "pop_culture_parody": ("II. Contextual Humor", "Cultural References"),
    "gallows_humor": ("III. NSFW Humor", "Morbid Humor"),
    "terminal_illness": ("III. NSFW Humor", "Morbid Humor"),
    "religious": ("III. NSFW Humor", "Taboo Topics"),
    "political_incorrectness": ("III. NSFW Humor", "Taboo Topics"),
    "absurdist_dread": ("III. NSFW Humor", "Existential Nihilism"),
    "cosmic_horror": ("III. NSFW Humor", "Existential Nihilism"),
    "underdog_humor": ("IV. Social Dynamics", "Power Reversals"),
    "authority_mockery": ("IV. Social Dynamics", "Power Reversals"),
    "misfortune": ("IV. Social Dynamics", "Schadenfreude"),
    "cringe_comedy": ("IV. Social Dynamics", "Schadenfreude"),
    "rule-of-three_violations": ("V. Technical/Structural", "Pattern Interrupts"),
    "misdirection": ("V. Technical/Structural", "Pattern Interrupts"),
    "emoji_absurdism": ("V. Technical/Structural", "Visual Humor"),
    "recursive_meta-humor": ("V. Technical/Structural", "Visual Humor"),
    "british_class_humor": ("VI. Regional/Subcultural", "Dialect Humor"),
    "southern_us": ("VI. Regional/Subcultural", "Dialect Humor"),
    "programming": ("VI. Regional/Subcultural", "Nerd Culture"),
    "science_puns": ("VI. Regional/Subcultural", "Nerd Culture"),
    "contextual_sarcasm": ("VII. Experimental", "AI-Specific Challenges"),
    "ethical_edge_cases": ("VII. Experimental", "AI-Specific Challenges"),
    "anachronism": ("VII. Experimental", "Temporal Humor"),
    "future_shock": ("VII. Experimental", "Temporal Humor"),
    "colbert": (None, None)
}

def ensure_directory_exists(directory):
    """
    Ensure the specified directory exists, creating it if necessary.
    Handles permissions errors gracefully.
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except PermissionError as e:
        print(f"Permission denied creating directory {directory}: {e}")
        raise
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        raise

def load_and_validate_data(tier_2_folder, llm, required_cols):
    """
    Load and validate Tier 2 data for a given LLM, filtering out empty subcategories.
    Returns a list of valid DataFrames with category mapping.
    """
    ensure_directory_exists(tier_2_folder)
    assessment_files = [f for f in os.listdir(tier_2_folder) if f.startswith(f"{llm}_") and f.endswith("_tier_2_quantitative.csv")]
    if not assessment_files:
        print(f"No Tier 2 assessment files found for {llm}. Skipping.")
        return []

    dataframes = []
    for file in assessment_files:
        subcategory = file.replace(f"{llm}_", "").replace("_tier_2_quantitative.csv", "")
        file_path = os.path.join(tier_2_folder, file)
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} entries for {llm} - {subcategory}")

            # Check required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Missing required columns in {file}: {missing_cols}. Skipping.")
                continue

            # Check for empty or all-NaN data
            df_valid = df.dropna(subset=required_cols, how="all")
            if df_valid.empty:
                print(f"No valid data in {file} after removing all-NaN rows. Skipping.")
                continue

            # Log NaN counts and unique values
            nan_counts = df[required_cols].isna().sum()
            print(f"NaN counts in {file}:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"  {col}: {count} NaN values")
            for col in required_cols:
                print(f"Unique values for {col} in {file}: {df[col].unique()}")

            # Map subcategory to category
            category = HUMOR_CATEGORIES.get(subcategory, (None, None))[0]
            if category is None:
                print(f"Skipping {subcategory} as it has no associated category.")
                continue

            df["category"] = category
            dataframes.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    return dataframes

def compute_mismatch_personal_funniness_all(tier_2_folder):
    """Compute average mismatch percentage between llm_classification and llm_personal_funny across all subcategories."""
    results = []
    llms = ["grok", "deepseek", "chatgpt", "llama"]
    all_data = []

    for llm in llms:
        dataframes = load_and_validate_data(tier_2_folder, llm, ["llm_classification", "llm_personal_funny"])
        if not dataframes:
            continue

        llm_data = pd.concat(dataframes, ignore_index=True)
        df_valid = llm_data[
            (llm_data["llm_classification"].isin(["YES", "NO"])) &
            (llm_data["llm_personal_funny"].isin(["YES", "NO"]))
        ].dropna(subset=["llm_classification", "llm_personal_funny"])

        if df_valid.empty:
            print(f"No valid data for mismatch calculation for {llm}.")
            continue

        mismatch = df_valid[
            ((df_valid["llm_classification"] == "YES") & (df_valid["llm_personal_funny"] == "NO")) |
            ((df_valid["llm_classification"] == "NO") & (df_valid["llm_personal_funny"] == "YES"))
        ]
        mismatch_percentage = (len(mismatch) / len(df_valid)) * 100 if len(df_valid) > 0 else np.nan

        results.append({
            "llm": llm,
            "metric": "mismatch_percentage",
            "value": mismatch_percentage,
            "valid_entries": len(df_valid)
        })

        print(f"Average mismatch percentage for {llm} (all subcategories): {mismatch_percentage:.2f}% with {len(df_valid)} valid entries")
        all_data.append(df_valid)

    # Compute average
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        df_valid = all_df[
            (all_df["llm_classification"].isin(["YES", "NO"])) &
            (all_df["llm_personal_funny"].isin(["YES", "NO"]))
        ]
        if not df_valid.empty:
            mismatch = df_valid[
                ((df_valid["llm_classification"] == "YES") & (df_valid["llm_personal_funny"] == "NO")) |
                ((df_valid["llm_classification"] == "NO") & (df_valid["llm_personal_funny"] == "YES"))
            ]
            avg_mismatch_percentage = (len(mismatch) / len(df_valid)) * 100 if len(df_valid) > 0 else np.nan

            results.append({
                "llm": "average",
                "metric": "mismatch_percentage",
                "value": avg_mismatch_percentage,
                "valid_entries": len(df_valid)
            })

            print(f"Average mismatch percentage (all subcategories): {avg_mismatch_percentage:.2f}% with {len(df_valid)} valid entries")

    return pd.DataFrame(results)

def compute_average_confidence_all(tier_2_folder):
    """Compute average confidence across all subcategories."""
    results = []
    llms = ["grok", "deepseek", "chatgpt", "llama"]
    all_data = []

    for llm in llms:
        dataframes = load_and_validate_data(tier_2_folder, llm, ["confidence"])
        if not dataframes:
            continue

        llm_data = pd.concat(dataframes, ignore_index=True)
        df_valid = llm_data.dropna(subset=["confidence"])
        if df_valid.empty:
            print(f"No valid confidence data for {llm}.")
            continue

        avg_confidence = df_valid["confidence"].mean()

        results.append({
            "llm": llm,
            "metric": "confidence",
            "value": avg_confidence,
            "valid_entries": len(df_valid)
        })

        print(f"Average confidence for {llm} (all subcategories): {avg_confidence:.4f} with {len(df_valid)} valid entries")
        all_data.append(df_valid)

    # Compute average
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        df_valid = all_df.dropna(subset=["confidence"])
        if not df_valid.empty:
            avg_confidence = df_valid["confidence"].mean()

            results.append({
                "llm": "average",
                "metric": "confidence",
                "value": avg_confidence,
                "valid_entries": len(df_valid)
            })

            print(f"Average confidence (all subcategories): {avg_confidence:.4f} with {len(df_valid)} valid entries")

    return pd.DataFrame(results)

def compute_average_accuracy_all(tier_2_folder):
    """Compute average accuracy across all subcategories."""
    results = []
    llms = ["grok", "deepseek", "chatgpt", "llama"]
    all_y_true = []
    all_y_pred = []

    for llm in llms:
        dataframes = load_and_validate_data(tier_2_folder, llm, ["true_label", "llm_classification"])
        if not dataframes:
            continue

        llm_data = pd.concat(dataframes, ignore_index=True)
        df_valid = llm_data.dropna(subset=["true_label", "llm_classification"])
        if df_valid.empty:
            print(f"No valid data for accuracy calculation for {llm}.")
            continue

        y_true = df_valid["true_label"].apply(lambda x: 1 if pd.notna(x) and x in [True, False] else np.nan)
        y_pred = df_valid["llm_classification"].apply(lambda x: 1 if x == "YES" else 0 if x == "NO" else np.nan)
        valid_idx = ~(y_true.isna() | y_pred.isna())
        y_true = y_true[valid_idx].astype(int)
        y_pred = y_pred[valid_idx].astype(int)

        if len(y_true) == 0:
            print(f"No valid predictions for accuracy calculation for {llm}.")
            continue

        accuracy = accuracy_score(y_true, y_pred)

        results.append({
            "llm": llm,
            "metric": "accuracy",
            "value": accuracy,
            "valid_entries": len(y_true)
        })

        print(f"Average accuracy for {llm} (all subcategories): {accuracy:.4f} with {len(y_true)} valid entries")
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    # Compute average
    if all_y_true:
        avg_accuracy = accuracy_score(all_y_true, all_y_pred)

        results.append({
            "llm": "average",
            "metric": "accuracy",
            "value": avg_accuracy,
            "valid_entries": len(all_y_true)
        })

        print(f"Average accuracy (all subcategories): {avg_accuracy:.4f} with {len(all_y_true)} valid entries")

    return pd.DataFrame(results)

def compute_mismatch_personal_funniness_by_category(tier_2_folder):
    """Compute average mismatch percentage between llm_classification and llm_personal_funny by category."""
    results = []
    llms = ["grok", "deepseek", "chatgpt", "llama"]
    all_data = []

    for llm in llms:
        dataframes = load_and_validate_data(tier_2_folder, llm, ["llm_classification", "llm_personal_funny"])
        if not dataframes:
            continue

        llm_data = pd.concat(dataframes, ignore_index=True)
        for category in llm_data["category"].unique():
            cat_df = llm_data[llm_data["category"] == category]
            df_valid = cat_df[
                (cat_df["llm_classification"].isin(["YES", "NO"])) &
                (cat_df["llm_personal_funny"].isin(["YES", "NO"]))
            ].dropna(subset=["llm_classification", "llm_personal_funny"])

            if df_valid.empty:
                print(f"No valid data for mismatch calculation for {llm} - {category}.")
                continue

            mismatch = df_valid[
                ((df_valid["llm_classification"] == "YES") & (df_valid["llm_personal_funny"] == "NO")) |
                ((df_valid["llm_classification"] == "NO") & (df_valid["llm_personal_funny"] == "YES"))
            ]
            mismatch_percentage = (len(mismatch) / len(df_valid)) * 100 if len(df_valid) > 0 else np.nan

            results.append({
                "llm": llm,
                "category": category,
                "metric": "mismatch_percentage",
                "value": mismatch_percentage,
                "valid_entries": len(df_valid)
            })

            print(f"Average mismatch percentage for {llm} - {category}: {mismatch_percentage:.2f}% with {len(df_valid)} valid entries")
            all_data.append(df_valid)

    # Compute average by category
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        for category in all_df["category"].unique():
            cat_df = all_df[all_df["category"] == category]
            df_valid = cat_df[
                (cat_df["llm_classification"].isin(["YES", "NO"])) &
                (cat_df["llm_personal_funny"].isin(["YES", "NO"]))
            ]
            if not df_valid.empty:
                mismatch = df_valid[
                    ((df_valid["llm_classification"] == "YES") & (df_valid["llm_personal_funny"] == "NO")) |
                    ((df_valid["llm_classification"] == "NO") & (df_valid["llm_personal_funny"] == "YES"))
                ]
                avg_mismatch_percentage = (len(mismatch) / len(df_valid)) * 100 if len(df_valid) > 0 else np.nan

                results.append({
                    "llm": "average",
                    "category": category,
                    "metric": "mismatch_percentage",
                    "value": avg_mismatch_percentage,
                    "valid_entries": len(df_valid)
                })

                print(f"Average mismatch percentage for {category}: {avg_mismatch_percentage:.2f}% with {len(df_valid)} valid entries")

    return pd.DataFrame(results)

def compute_average_confidence_by_category(tier_2_folder):
    """Compute average confidence by category."""
    results = []
    llms = ["grok", "deepseek", "chatgpt", "llama"]
    all_data = []

    for llm in llms:
        dataframes = load_and_validate_data(tier_2_folder, llm, ["confidence"])
        if not dataframes:
            continue

        llm_data = pd.concat(dataframes, ignore_index=True)
        for category in llm_data["category"].unique():
            cat_df = llm_data[llm_data["category"] == category]
            df_valid = cat_df.dropna(subset=["confidence"])
            if df_valid.empty:
                print(f"No valid confidence data for {llm} - {category}.")
                continue

            avg_confidence = df_valid["confidence"].mean()

            results.append({
                "llm": llm,
                "category": category,
                "metric": "confidence",
                "value": avg_confidence,
                "valid_entries": len(df_valid)
            })

            print(f"Average confidence for {llm} - {category}: {avg_confidence:.4f} with {len(df_valid)} valid entries")
            all_data.append(df_valid)

    # Compute average by category
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        for category in all_df["category"].unique():
            cat_df = all_df[all_df["category"] == category]
            df_valid = cat_df.dropna(subset=["confidence"])
            if not df_valid.empty:
                avg_confidence = df_valid["confidence"].mean()

                results.append({
                    "llm": "average",
                    "category": category,
                    "metric": "confidence",
                    "value": avg_confidence,
                    "valid_entries": len(df_valid)
                })

                print(f"Average confidence for {category}: {avg_confidence:.4f} with {len(df_valid)} valid entries")

    return pd.DataFrame(results)

def compute_average_accuracy_by_category(tier_2_folder):
    """Compute average accuracy by category."""
    results = []
    llms = ["grok", "deepseek", "chatgpt", "llama"]
    all_data = []

    for llm in llms:
        dataframes = load_and_validate_data(tier_2_folder, llm, ["true_label", "llm_classification"])
        if not dataframes:
            continue

        llm_data = pd.concat(dataframes, ignore_index=True)
        for category in llm_data["category"].unique():
            cat_df = llm_data[llm_data["category"] == category]
            df_valid = cat_df.dropna(subset=["true_label", "llm_classification"])
            if df_valid.empty:
                print(f"No valid data for accuracy calculation for {llm} - {category}.")
                continue

            y_true = df_valid["true_label"].apply(lambda x: 1 if pd.notna(x) and x in [True, False] else np.nan)
            y_pred = df_valid["llm_classification"].apply(lambda x: 1 if x == "YES" else 0 if x == "NO" else np.nan)
            valid_idx = ~(y_true.isna() | y_pred.isna())
            y_true = y_true[valid_idx].astype(int)
            y_pred = y_pred[valid_idx].astype(int)

            if len(y_true) == 0:
                print(f"No valid predictions for accuracy calculation for {llm} - {category}.")
                continue

            accuracy = accuracy_score(y_true, y_pred)

            results.append({
                "llm": llm,
                "category": category,
                "metric": "accuracy",
                "value": accuracy,
                "valid_entries": len(y_true)
            })

            print(f"Average accuracy for {llm} - {category}: {accuracy:.4f} with {len(y_true)} valid entries")
            all_data.append(df_valid.assign(y_true=y_true, y_pred=y_pred))

    # Compute average by category
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        for category in all_df["category"].unique():
            cat_df = all_df[all_df["category"] == category]
            if not cat_df.empty and "y_true" in cat_df and "y_pred" in cat_df:
                y_true = cat_df["y_true"].dropna()
                y_pred = cat_df["y_pred"].dropna()
                if len(y_true) == len(y_pred) and len(y_true) > 0:
                    avg_accuracy = accuracy_score(y_true, y_pred)

                    results.append({
                        "llm": "average",
                        "category": category,
                        "metric": "accuracy",
                        "value": avg_accuracy,
                        "valid_entries": len(y_true)
                    })

                    print(f"Average accuracy for {category}: {avg_accuracy:.4f} with {len(y_true)} valid entries")

    return pd.DataFrame(results)

def plot_metric(data, metric_name, output_folder, by_category=False):
    """
    Create a bar chart for the given metric, either across all subcategories or by category.
    """
    ensure_directory_exists(output_folder)
    plt.figure(figsize=(10 if not by_category else 14, 6))
    if by_category:
        sns.barplot(
            data=data,
            x="category",
            y="value",
            hue="llm",
            palette="viridis",
            order=[
                "I. Linguistic Humor",
                "II. Contextual Humor",
                "III. NSFW Humor",
                "IV. Social Dynamics",
                "V. Technical/Structural",
                "VI. Regional/Subcultural",
                "VII. Experimental"
            ]
        )
        plt.xticks(rotation=45, ha="right")
    else:
        sns.barplot(
            data=data,
            x="llm",
            y="value",
            hue="llm",
            palette="viridis",
            order=["grok", "deepseek", "chatgpt", "llama", "average"]
        )
    plt.xlabel("Category" if by_category else "LLM")
    plt.ylabel(f"{metric_name.capitalize()} ({'%' if metric_name == 'mismatch_percentage' else ''})")
    plt.title(f"Average {metric_name.capitalize()} {'by Category' if by_category else 'Across All Subcategories'}")
    plt.ylim(0, 100 if metric_name == "mismatch_percentage" else 1)
    plt.legend(title="LLM")
    plt.tight_layout()

    output_path = os.path.join(output_folder, f"{metric_name}_plot{'_by_category' if by_category else ''}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {metric_name} plot to {output_path}")

def main(tier_2_folder="tier_2_outputs"):
    """Main function to compute and plot all metrics."""
    output_folder = os.path.join(tier_2_folder, "metric_plots")
    ensure_directory_exists(output_folder)
    
    # Compute and save mismatch percentage (all subcategories)
    mismatch_all = compute_mismatch_personal_funniness_all(tier_2_folder)
    if not mismatch_all.empty:
        mismatch_all.to_csv(os.path.join(output_folder, "mismatch_percentage_all.csv"), index=False)
        print(f"Saved mismatch percentage (all subcategories) to {os.path.join(output_folder, 'mismatch_percentage_all.csv')}")
        plot_metric(mismatch_all, "mismatch_percentage", output_folder)

    # Compute and save average confidence (all subcategories)
    confidence_all = compute_average_confidence_all(tier_2_folder)
    if not confidence_all.empty:
        confidence_all.to_csv(os.path.join(output_folder, "confidence_all.csv"), index=False)
        print(f"Saved average confidence (all subcategories) to {os.path.join(output_folder, 'confidence_all.csv')}")
        plot_metric(confidence_all, "confidence", output_folder)

    # Compute and save average accuracy (all subcategories)
    accuracy_all = compute_average_accuracy_all(tier_2_folder)
    if not accuracy_all.empty:
        accuracy_all.to_csv(os.path.join(output_folder, "accuracy_all.csv"), index=False)
        print(f"Saved average accuracy (all subcategories) to {os.path.join(output_folder, 'accuracy_all.csv')}")
        plot_metric(accuracy_all, "accuracy", output_folder)

    # Compute and save mismatch percentage (by category)
    mismatch_by_category = compute_mismatch_personal_funniness_by_category(tier_2_folder)
    if not mismatch_by_category.empty:
        mismatch_by_category.to_csv(os.path.join(output_folder, "mismatch_percentage_by_category.csv"), index=False)
        print(f"Saved mismatch percentage (by category) to {os.path.join(output_folder, 'mismatch_percentage_by_category.csv')}")
        plot_metric(mismatch_by_category, "mismatch_percentage", output_folder, by_category=True)

    # Compute and save average confidence (by category)
    confidence_by_category = compute_average_confidence_by_category(tier_2_folder)
    if not confidence_by_category.empty:
        confidence_by_category.to_csv(os.path.join(output_folder, "confidence_by_category.csv"), index=False)
        print(f"Saved average confidence (by category) to {os.path.join(output_folder, 'confidence_by_category.csv')}")
        plot_metric(confidence_by_category, "confidence", output_folder, by_category=True)

    # Compute and save average accuracy (by category)
    accuracy_by_category = compute_average_accuracy_by_category(tier_2_folder)
    if not accuracy_by_category.empty:
        accuracy_by_category.to_csv(os.path.join(output_folder, "accuracy_by_category.csv"), index=False)
        print(f"Saved average accuracy (by category) to {os.path.join(output_folder, 'accuracy_by_category.csv')}")
        plot_metric(accuracy_by_category, "accuracy", output_folder, by_category=True)

def replot_confidence_metrics(input_folder="tier_2_outputs/metric_plots", output_folder="tier_2_outputs/metric_plots/revised_confidence_plots"):
    """
    Replot confidence metrics from confidence_all.csv and confidence_by_category.csv.
    Creates two bar charts: overall confidence and confidence by category.
    Scales y-axis dynamically based on data and saves plots in the output folder.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    def ensure_directory_exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            raise

    def load_and_validate_csv(file_path, required_cols):
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {file_path} with {len(df)} entries")
        except FileNotFoundError:
            print(f"Error: {file_path} does not exist.")
            return None
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns in {file_path}: {missing_cols}")
            return None
        df = df.dropna(subset=required_cols, how="all")
        if df.empty:
            print(f"No valid data in {file_path}.")
            return None
        return df

    ensure_directory_exists(output_folder)

    # Replot overall confidence
    df_all = load_and_validate_csv(os.path.join(input_folder, "confidence_all.csv"), ["llm", "metric", "value", "valid_entries"])
    if df_all is not None:
        df_all = df_all[(df_all["metric"] == "confidence") & (df_all["llm"].isin(["grok", "deepseek", "chatgpt", "llama", "average"]))].dropna(subset=["value"])
        if not df_all.empty:
            print("Confidence data (all subcategories):", df_all[["llm", "value", "valid_entries"]])
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_all, x="llm", y="value", hue="llm", palette="viridis", order=["grok", "deepseek", "chatgpt", "llama", "average"])
            plt.xlabel("LLM")
            plt.ylabel("Average Confidence")
            plt.title("Average Confidence Across All Subcategories")
            max_value = df_all["value"].max()
            plt.ylim(0, max_value * 1.1)  # Dynamic y-axis with 10% padding
            plt.legend(title="LLM")
            plt.tight_layout()
            output_path = os.path.join(output_folder, "confidence_all_plot.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved confidence plot (all subcategories) to {output_path}")

    # Replot confidence by category
    df_by_cat = load_and_validate_csv(os.path.join(input_folder, "confidence_by_category.csv"), ["llm", "category", "metric", "value", "valid_entries"])
    if df_by_cat is not None:
        df_by_cat = df_by_cat[(df_by_cat["metric"] == "confidence") & (df_by_cat["llm"].isin(["grok", "deepseek", "chatgpt", "llama", "average"]))].dropna(subset=["value", "category"])
        if not df_by_cat.empty:
            print("Confidence data (by category):", df_by_cat[["llm", "category", "value", "valid_entries"]])
            plt.figure(figsize=(14, 8))
            sns.barplot(
                data=df_by_cat,
                x="category",
                y="value",
                hue="llm",
                palette="viridis",
                order=[
                    "I. Linguistic Humor",
                    "II. Contextual Humor",
                    "III. NSFW Humor",
                    "IV. Social Dynamics",
                    "V. Technical/Structural",
                    "VI. Regional/Subcultural",
                    "VII. Experimental"
                ]
            )
            plt.xlabel("Category")
            plt.ylabel("Average Confidence")
            plt.title("Average Confidence by Category")
            max_value = df_by_cat["value"].max()
            plt.ylim(0, max_value * 1.1)  # Dynamic y-axis with 10% padding
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="LLM")
            plt.tight_layout()
            output_path = os.path.join(output_folder, "confidence_by_category_plot.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved confidence plot (by category) to {output_path}")
            
#replot_confidence_metrics()

#TIER 2 graphs:
def visualize_tier_2_metrics_by_llm(tier_2_folder="tier_2_outputs", output_folder="tier_2_outputs/tier_2_visualizations"):
    """Generate four bar charts, one per LLM, showing average funniness, offensiveness, originality, appropriateness, clarity by category and overall average."""
    # Create output directory at the start
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created/verified output directory: {output_folder}")
    except Exception as e:
        print(f"Error creating directory {output_folder}: {e}")
        raise

    # HUMOR_CATEGORIES dictionary
    HUMOR_CATEGORIES = {
        "puns": "I. Linguistic Humor", "homophonic": "I. Linguistic Humor", "paraprosdokians": "I. Linguistic Humor",
        "double_entendres": "I. Linguistic Humor", "malapropisms": "I. Linguistic Humor",
        "surreal": "II. Contextual Humor", "anti-jokes": "II. Contextual Humor",
        "historical_satire": "II. Contextual Humor", "pop_culture_parody": "II. Contextual Humor",
        "gallows_humor": "III. NSFW Humor", "terminal_illness": "III. NSFW Humor",
        "religious": "III. NSFW Humor", "political_incorrectness": "III. NSFW Humor",
        "absurdist_dread": "III. NSFW Humor", "cosmic_horror": "III. NSFW Humor",
        "underdog_humor": "IV. Social Dynamics", "authority_mockery": "IV. Social Dynamics",
        "misfortune": "IV. Social Dynamics", "cringe_comedy": "IV. Social Dynamics",
        "rule-of-three_violations": "V. Technical/Structural", "misdirection": "V. Technical/Structural",
        "emoji_absurdism": "V. Technical/Structural", "recursive_meta-humor": "V. Technical/Structural",
        "british_class_humor": "VI. Regional/Subcultural", "southern_us": "VI. Regional/Subcultural",
        "programming": "VI. Regional/Subcultural", "science_puns": "VI. Regional/Subcultural",
        "contextual_sarcasm": "VII. Experimental", "ethical_edge_cases": "VII. Experimental",
        "anachronism": "VII. Experimental", "future_shock": "VII. Experimental",
        "colbert": None
    }

    def load_and_validate_data(llm, required_cols):
        assessment_files = [f for f in os.listdir(tier_2_folder) if f.startswith(f"{llm}_") and f.endswith("_tier_2_quantitative.csv")]
        if not assessment_files:
            print(f"No Tier 2 files for {llm}. Skipping.")
            return []

        dataframes = []
        for file in assessment_files:
            subcategory = file.replace(f"{llm}_", "").replace("_tier_2_quantitative.csv", "")
            file_path = os.path.join(tier_2_folder, file)
            try:
                df = pd.read_csv(file_path)
                print(f"Loaded {len(df)} entries for {llm} - {subcategory}")
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"Missing columns in {file}: {missing_cols}. Skipping.")
                    continue
                df_valid = df.dropna(subset=required_cols, how="all")
                if df_valid.empty:
                    print(f"No valid data in {file}. Skipping.")
                    continue
                nan_counts = df_valid[required_cols].isna().sum()
                print(f"NaN counts in {file}:")
                for col, count in nan_counts.items():
                    if count > 0:
                        print(f"  {col}: {count} NaN values")
                category = HUMOR_CATEGORIES.get(subcategory)
                if category is None:
                    print(f"Skipping {subcategory} as it has no associated category.")
                    continue
                df_valid["category"] = category
                dataframes.append(df_valid)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        return dataframes

    def plot_metrics_for_llm(llm_data, llm_name, metrics):
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=llm_data,
            x="category",
            y="value",
            hue="metric",
            palette="viridis",
            order=[
                "I. Linguistic Humor",
                "II. Contextual Humor",
                "III. NSFW Humor",
                "IV. Social Dynamics",
                "V. Technical/Structural",
                "VI. Regional/Subcultural",
                "VII. Experimental",
                "Overall Average"
            ]
        )
        plt.xlabel("Category")
        plt.ylabel("Average Value")
        plt.title(f"Average Metrics for {llm_name}")
        max_value = llm_data["value"].max()
        plt.ylim(0, max_value * 1.1)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Metric")
        plt.tight_layout()
        output_path = os.path.join(output_folder, f"{llm_name.lower()}_metrics_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {llm_name} metrics plot to {output_path}")

    metrics = ["funniness", "offensiveness", "originality", "appropriateness", "clarity"]
    llms = ["grok", "deepseek", "chatgpt", "llama"]
    all_results = []

    for llm in llms:
        dataframes = load_and_validate_data(llm, metrics)
        if not dataframes:
            continue
        llm_data = pd.concat(dataframes, ignore_index=True)
        if "appropriateness" in llm_data.columns:
            llm_data["appropriateness"] = llm_data["appropriateness"] / 2.0

        llm_results = []
        for category in llm_data["category"].unique():
            cat_df = llm_data[llm_data["category"] == category]
            for metric in metrics:
                df_valid = cat_df.dropna(subset=[metric])
                if df_valid.empty:
                    print(f"No valid {metric} data for {llm} - {category}.")
                    continue
                avg_value = df_valid[metric].mean()
                llm_results.append({
                    "llm": llm,
                    "category": category,
                    "metric": metric,
                    "value": avg_value,
                    "valid_entries": len(df_valid)
                })
                print(f"Average {metric} for {llm} - {category}: {avg_value:.4f} with {len(df_valid)} valid entries")

        # Compute overall average per metric
        for metric in metrics:
            df_valid = llm_data.dropna(subset=[metric])
            if not df_valid.empty:
                avg_value = df_valid[metric].mean()
                llm_results.append({
                    "llm": llm,
                    "category": "Overall Average",
                    "metric": metric,
                    "value": avg_value,
                    "valid_entries": len(df_valid)
                })
                print(f"Overall average {metric} for {llm}: {avg_value:.4f} with {len(df_valid)} valid entries")

        llm_results_df = pd.DataFrame(llm_results)
        all_results.append(llm_results_df)
        if not llm_results_df.empty:
            plot_metrics_for_llm(llm_results_df, llm, metrics)

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df.to_csv(os.path.join(output_folder, "llm_category_metrics.csv"), index=False)
        print(f"Saved LLM category metrics to {os.path.join(output_folder, 'llm_category_metrics.csv')}")
        
#visualize_tier_2_metrics_by_llm()
