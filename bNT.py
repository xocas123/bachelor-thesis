import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, creating it if necessary."""
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        raise


SUBREDDIT_MAPPING = {
    "I. Linguistic Humor": {
        "Wordplay": {
            "Puns": ["puns", "dadjokes", "punpatrol", "punny", "badpuns", "cleanjokes", "punintended", "punsters", "puncentral"],
            "Homophonic": ["boneappletea", "wordavalanches", "homophones", "eggcorn", "mondegreens"],
            "Paraprosdokians": ["oneliners", "showerthoughts", "wordplay", "unexpectedtwist", "quipsters"]
        },
        "Semantic Shifts": {
            "Double Entendres": ["dirtyjokes", "wordplay", "standupcomedy", "innuendos", "naughtyhumor"],
            "Malapropisms": ["boneappletea", "malaphors", "engrish", "lostintranslation", "funnytypos"]
        }
    },
    "II. Contextual Humor": {
        "Situational Absurdity": {
            "Surreal": ["fifthworldproblems", "surrealhumor", "absurdism", "weirdhumor", "oddlyspecificmemes"],
            "Anti-jokes": ["antijokes", "dryhumor", "deadpan", "literalhumor", "notfunnybutfunny"]
        },
        "Cultural References": {
            "Historical Satire": ["historymemes", "historyjerk", "ancientmemes", "timemachinehumor", "medievalmemes"],
            "Pop Culture Parody": ["memes", "terriblefandommemes", "popcultureparody", "tvshowmemes", "moviehumor"]
        }
    },
    "III. NSFW Humor": {
        "Morbid Humor": {
            "Gallows Humor": ["darkjokes", "blackhumor", "edgydarkjokes", "morbidjokes", "deathhumor"],
            "Terminal Illness": ["toosoon", "morbidhumor", "darkmedicalhumor", "sickjokes"]
        },
        "Taboo Topics": {
            "Religious": ["religiousfruitcake", "exchristianmemes", "atheistmemes", "blasphemoushumor"],
            "Political Incorrectness": ["imgoingtohellforthis", "offensivememes", "edgymemes", "controversialhumor"]
        },
        "Existential Nihilism": {
            "Absurdist Dread": ["absurdism", "existentialmemes", "nihilistmemes", "meaninglesshumor"],
            "Cosmic Horror": ["lovecraftianhumor", "cthulhumemes", "weirdfictionhumor", "eldritchhumor"]
        }
    },
    "IV. Social Dynamics": {
        "Power Reversals": {
            "Underdog Humor": ["wholesomememes", "upliftinghumor", "feelgoodmemes", "kindhumor"],
            "Authority Mockery": ["boomershumor", "okboomer", "genzhumor", "antiworkmemes"]
        },
        "Schadenfreude": {
            "Misfortune": ["leopardsatemyface", "instantkarma", "justicepornmemes", "pettyrevengehumor"],
            "Cringe Comedy": ["cringetopia_Two", "sadcringe", "cringe", "awkwardhumor"]
        }
    },
    "V. Technical/Structural": {
        "Pattern Interrupts": {
            "Rule-of-three Violations": ["unexpected", "theydidthemath", "plotwistmemes", "surpriseending"],
            "Misdirection": ["nonononoyes", "yesyesyesno", "unexpectedoutcome", "twistyjokes"]
        },
        "Visual Humor": {
            "Emoji Absurdism": ["emojipasta", "deepfriedmemes", "emojimemes", "absurdemojis"],
            "Recursive Meta-Humor": ["recursion", "metamemes", "selfreferentialhumor", "memeception"]
        }
    },
    "VI. Regional/Subcultural": {
        "Dialect Humor": {
            "British Class Humor": ["britishhumour", "casualuk", "britishproblems", "ukcomedy", "britishmemes"],
            "Southern US": ["holdmybeer", "holdmyfries", "southernmemes", "redneckhumor"]
        },
        "Nerd Culture": {
            "Programming": ["programmerhumor", "codinghumor", "softwaregore", "techhumor"],
            "Science Puns": ["sciencememes", "physicsmemes", "chemistrymemes", "mathmemes"]
        }
    },
    "VII. Experimental": {
        "AI-Specific Challenges": {
            "Contextual Sarcasm": ["sarcasm", "fuckyouinparticular", "sarcasticmemes", "ironicmemes"],
            "Ethical Edge Cases": ["unethicallifeprotips", "badphilosophy", "morallygreyhumor", "ethicallydubious"]
        },
        "Temporal Humor": {
            "Anachronism": ["historymemes", "timetravelercaught", "anachronistichumor", "pastfutureclash"],
            "Future Shock": ["cyberpunkmemes", "transhumanismhumor", "scifihumor", "futuristmemes"]
        }
    }
}

def create_subreddit_to_category_mapping(subreddit_mapping):
    """Create a mapping from subsubcategory to category."""
    subsubcategory_to_category = {}
    for category, subcategories in subreddit_mapping.items():
        for subcategory, subsubcategories in subcategories.items():
            for subsubcategory in subsubcategories.keys():
                subsubcategory_cleaned = subsubcategory.lower().replace(" ", "_")
                subsubcategory_to_category[subsubcategory_cleaned] = category
    return subsubcategory_to_category

def correlation_matrix_analysis(tier_2_folder="tier_2_outputs", output_base_folder="tier_2_outputs"):
    
    output_folder = os.path.join(output_base_folder, "correlation_matrices")
    ensure_directory_exists(output_folder)

    llms = ["grok", "deepseek", "chatgpt", "llama"]
    attributes = ["offensiveness", "originality", "appropriateness", "clarity", "confidence", "score"]
    categories = [
        "I. Linguistic Humor",
        "II. Contextual Humor",
        "III. NSFW Humor",
        "IV. Social Dynamics",
        "V. Technical/Structural",
        "VI. Regional/Subcultural",
        "VII. Experimental",
        "ColBERT"
    ]

    subsubcategory_to_category = create_subreddit_to_category_mapping(SUBREDDIT_MAPPING)

    for llm in llms:
        print(f"\nProcessing dataset for {llm}")
        reddit_dataframes = []
        colbert_dataframes = []

        colbert_file = os.path.join(tier_2_folder, f"{llm}_colbert_tier_2_quantitative.csv")
        if os.path.exists(colbert_file):
            try:
                df = pd.read_csv(colbert_file)
                df["category"] = "ColBERT"
                colbert_dataframes.append(df)
            except Exception as e:
                print(f"Error processing {colbert_file}: {e}")

        for file in os.listdir(tier_2_folder):
            if file.startswith(f"{llm}_") and file.endswith("_tier_2_quantitative.csv") and "colbert" not in file:
                file_path = os.path.join(tier_2_folder, file)
                try:
                    df = pd.read_csv(file_path)
                    parts = file.split("_")
                    if len(parts) >= 3:
                        subsubcategory_parts = parts[1:-3]  
                        subsubcategory = "_".join(subsubcategory_parts).lower()  
                        category = subsubcategory_to_category.get(subsubcategory)
                        if category:
                            df["category"] = category
                            reddit_dataframes.append(df)
                        else:
                            print(f"Could not map subsubcategory {subsubcategory} to a category for file {file}. Skipping.")
                    else:
                        print(f"Could not parse subsubcategory from file name {file}. Skipping.")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

     
        all_dataframes = reddit_dataframes + colbert_dataframes
        if not all_dataframes:
            print(f"No data found for {llm}. Skipping.")
            continue

        data = pd.concat(all_dataframes, ignore_index=True)

        required_cols = attributes + ["category"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Missing columns for {llm}: {missing_cols}. Skipping.")
            continue
        data = data.copy()
        valid_attributes = []
        for var in attributes:
            try:
                data.loc[:, var] = pd.to_numeric(data[var], errors='coerce')
                nan_count = data[var].isna().sum()
                if nan_count > 0:
                    print(f"Warning: {var} contains {nan_count} NaN values. Filling with median.")
                    data.loc[:, var] = data[var].fillna(data[var].median())
                if data[var].isna().all():
                    print(f"Error: {var} contains only NaN after conversion. Skipping attribute.")
                    continue
                if data[var].std() == 0:
                    print(f"Warning: {var} has no variance (all values are identical). Skipping attribute.")
                    continue
                valid_attributes.append(var)
            except Exception as e:
                print(f"Error converting {var} to numeric: {e}. Skipping attribute.")

        if not valid_attributes:
            print(f"No valid attributes for {llm} after filtering. Skipping.")
            continue
        for category in categories:
            data.loc[:, f"category_{category}"] = (data["category"] == category).astype(int)
        correlation_matrix = pd.DataFrame(index=valid_attributes, columns=categories)
        for attr in valid_attributes:
            for category in categories:
                correlation = data[attr].corr(data[f"category_{category}"], method="pearson")
                correlation_matrix.loc[attr, category] = correlation if not pd.isna(correlation) else 0.0

        correlation_matrix = correlation_matrix.astype(float)

        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, fmt=".2f")
        plt.title(f"Correlation Matrix for {llm}: Attributes vs Categories")
        plt.xlabel("Categories")
        plt.ylabel("Attributes")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path = os.path.join(output_folder, f"correlation_matrix_{llm}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved correlation matrix plot for {llm} to {plot_path}")

if __name__ == "__main__":
    correlation_matrix_analysis()
    
    
def rf(OUTPUT_FOLDER):
    OUTPUT_FOLDER = "abc123"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

csv_path = os.path.join(OUTPUT_FOLDER, "attribute_stats.csv")
stats_df = pd.read_csv(csv_path)
print(f"Loaded {len(stats_df)} rows from {csv_path}")

stats_df['is_mismatch'] = stats_df['group'].apply(lambda x: 1 if x == 'mismatch' else 0)

features = ["funniness_mean", "offensiveness_mean", "originality_mean",
            "appropriateness_mean", "clarity_mean", "confidence_mean"]
X = stats_df[features]
y = stats_df['is_mismatch']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)
print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")

feature_importance = pd.DataFrame({
    'predictor': features,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values(by='importance', ascending=False)

feature_importance['llm'] = stats_df['llm'].iloc[0]
feature_importance['dataset'] = stats_df['dataset'].iloc[0]

feature_importance = feature_importance[['llm', 'dataset', 'predictor', 'importance']]

output_path = os.path.join(OUTPUT_FOLDER, "random_forest_results.csv")
feature_importance.to_csv(output_path, index=False)
print(f"Saved feature importance to {output_path}")

print("\nFeature Importance:")
print(feature_importance)