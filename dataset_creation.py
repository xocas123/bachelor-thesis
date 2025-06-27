import praw
import os
import pandas as pd
import time
import re
import nltk
import datasets
from datasets import load_dataset
import random
import backoff
from nltk.tokenize import sent_tokenize
import logging
from concurrent.futures import ThreadPoolExecutor

nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reddit API Configuration. Since I am putting a link to this on the github repo, i'm omitting my reddit credentials.
reddit = praw.Reddit(
    client_id="placeholder",
    client_secret="placeholder",
    user_agent="placeholder",
    username="placeholder",  
    password="placeholder",
)


logger.info(f"Read-only mode: {reddit.read_only}")
try:
    logger.info(f"Authenticated as: {reddit.user.me().name}")
    logger.info(f"NSFW access enabled: {reddit.user.me().over_18}")
    logger.info(reddit.subreddit("darkhumor").hot(limit=1).__next__().title)
    
except Exception as e:
    logger.error(f"Authentication failed: {e}")
    
"""
print(reddit.subreddit("dark_humor").hot(limit=1).__next__().title)
print(reddit.subreddit("blackhumor").hot(limit=1).__next__().title)

"""
"""
try:
    for subreddit in ["darkjokes", "blackhumor", "meanjokes"]:
        post = reddit.subreddit(subreddit).hot(limit=1).__next__()
        print(f"{subreddit}: {post.title}, NSFW: {post.over_18}")
except Exception as e:
    print(f"Error: {e}")

"""




output_dir = "humor_dataset_reddit"
os.makedirs(output_dir, exist_ok=True)

#mapping of the rubreddits. roman nu
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
        "Existential Nihlism": {
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

def scrape_subreddit(subreddit, limit=25000, min_score=2, min_length=20, max_length=500):
    """Scraping posts, upvotres, downvotes. combine title and body"""
    try:
        time.sleep(2)  #reddit scrape compliance, maybe 1 works
        sub = reddit.subreddit(subreddit)
        logger.info(f"Scraping {subreddit}")
        posts = []
        for submission in sub.hot(limit=limit):
            if (not submission.stickied and 
                submission.is_self and 
                submission.score >= min_score and
                not submission.url.endswith(('.jpg', '.png', '.gif', '.mp4', '.jpeg')) and #no  images, links, vids, etc
                not re.search(r'http[s]?://', submission.selftext)):
                text_content = f"{submission.title.strip()} {submission.selftext.strip().replace('\n', ' ').replace('\r', ' ')}"
                text_content = re.sub(r'\s+', ' ', text_content).strip()
                if min_length <= len(text_content) <= max_length:
                    posts.append({
                        "text": text_content,
                        "upvotes": submission.ups,
                        "downvotes": submission.downs,
                        "score": submission.score
                    })
        logger.info(f"Scraped {len(posts)} valid posts from {subreddit}")
        return posts
    except Exception as e:
        logger.error(f"Error scraping {subreddit}: {e}")
        return []

def generate_subcategory_dataset(subcat, subreddits, target_size=1000):
    data = []
    for subreddit in subreddits:
        posts = scrape_subreddit(subreddit)
        for post in posts:
            data.append(post) 
            if len(data) >= target_size:
                break
        if len(data) >= target_size:
            break
        time.sleep(1)
    
    df = pd.DataFrame(data)
    if not df.empty:
        filename = f"{output_dir}/{subcat.lower().replace(' ', '_')}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} entries to {filename}")
    else:
        logger.warning(f"No data collected for {subcat}")

def generate_dataset():
    """helper function""" 
    for category, subcats in SUBREDDIT_MAPPING.items():
        for subcat_group, subcat_dict in subcats.items():
            for subcat, subreddits in subcat_dict.items():
                logger.info(f"Generating dataset for {subcat}")
                generate_subcategory_dataset(subcat, subreddits)
                time.sleep(2)

def load_non_humorous_texts(sample_size, min_length=20, max_length=500):
    logger.info(f"Loading {sample_size} non-humorous texts")
    try:
        non_humor_texts = []
        dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        for entry in dataset:
            text = entry.get("text", "").split('.')[0].strip()
            text = re.sub(r'\s+', ' ', text).strip()
            if min_length <= len(text) <= max_length:
                non_humor_texts.append(text)
            if len(non_humor_texts) >= sample_size:
                break
        random.shuffle(non_humor_texts)
        return non_humor_texts[:sample_size]
    except Exception as e:
        logger.error(f"error non humores texts: {e}")
        return []

def augment_humor_datasets(folder_path):
    """Augment--> add non humor to  humor datastes."""
    humor_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and not f.endswith('_augmented.csv')]
    

    total_humor_rows = 0
    for file in humor_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        total_humor_rows += len(df)
    
    logger.info(f"Total humorous rows: {total_humor_rows}")
    non_humor_texts = load_non_humorous_texts(total_humor_rows)
    if not non_humor_texts:
        logger.error("failed. wiki not uploaded")
        return

    idx = 0
    for file in humor_files:
        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)
        df["is_humorous"] = True
        n = len(df)
        non_humor_sample = non_humor_texts[idx:idx + n]
        idx += n
        
        non_df = pd.DataFrame({
            "text": non_humor_sample,
            "upvotes": [None] * len(non_humor_sample),
            "downvotes": [None] * len(non_humor_sample),
            "score": [None] * len(non_humor_sample),
            "is_humorous": [False] * len(non_humor_sample)
        })
        full_df = pd.concat([df, non_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
        full_df.to_csv(path.replace('.csv', '_augmented.csv'), index=False)
        logger.info(f"Augmented: {file} (now {len(full_df)} rows)")



folder_path = "humor_dataset_reddit"
generate_dataset()
augment_humor_datasets(folder_path)