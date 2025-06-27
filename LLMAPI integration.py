import pandas as pd
import os
import requests
import json
import re
import time
from together import Together

#omitted api keys w/ placeholder 

def process_all_reddit_with_grok(folder_path="humor_dataset_reddit", num_batches=10, api_key="abc"):
    endpoint = "https://api.x.ai/v1/chat/completions"
    model = "grok-3-latest"
    batch_size = 3
   
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }


    output_folder = "llm_assessments"
    os.makedirs(output_folder, exist_ok=True)


    dataset_files = [f for f in os.listdir(folder_path) if f.endswith('_augmented.csv')]
    if not dataset_files:
        print(f"Not found in {folder_path}.")
        return

    #print(f" {len(dataset_files)} Reddit datasets: {dataset_files}")


    # Process each dataset
    for dataset_file in dataset_files:
        dataset_path = os.path.join(folder_path, dataset_file)
        #print(f"\nProcessing dataset: {dataset_file}")

       
        try:
            df = pd.read_csv(dataset_path)
            df = df.rename(columns={"is_humorous": "true_label"})
            subcategory = dataset_file.replace('_augmented.csv', '')
            #print(f"Loaded {len(df)} entries for {subcategory}")
        except Exception as e:
            #print(f"Error loading dataset {dataset_path}: {e}")
            continue

    
        results = []
        #print(f"Processing {len(df)} entries for {subcategory} with Grok...")

        for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
            batch = df.iloc[i:i+batch_size]
            statements = batch["text"].tolist()
            true_labels = batch["true_label"].tolist()
            upvotes = batch["upvotes"].tolist()
            downvotes = batch["downvotes"].tolist()
            scores = batch["score"].tolist()

            if len(statements) < batch_size:
                statements += ["Placeholder"] * (batch_size - len(statements))
                true_labels += [False] * (batch_size - len(true_labels))
                upvotes += [None] * (batch_size - len(upvotes))
                downvotes += [None] * (batch_size - len(downvotes))
                scores += [None] * (batch_size - len(scores))

            # prompting
            prompt = (
                "For the joke below, provide:\n"
                "1. Joke classification (YES/NO)\n"
                "2. Confidence score (1-10, where 1 is least confident and 10 is most confident)\n"
                "3. Your personal opinion if you find it funny (YES/NO)\n"
                "Format strictly as: YES, 10, YES\n\n"
                f"Joke: {statements[0]}"
            )

            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a humor classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 50
            }

            # Send request
            try:
                #print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
                response = requests.post(endpoint, headers=headers, json=payload)
                response_json = response.json()
                #print(f"stat Code: {response.status_code}")
                #print(f"resposta: {json.dumps(response_json, indent=2)}")

                # ainda falta errors por a falta de conectividade ou problemas co a API
                if response.status_code == 401:
                    raise Exception(
                        f"401 key error. "
                    )
                if response.status_code != 200 or "error" in response_json:
                    raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

                # Parse response
                if "choices" in response_json:
                    llm_answer = response_json["choices"][0]["message"]["content"].strip()
                    #print(f"LLM Output: {llm_answer}")

                    # output parsing
                    cleaned = llm_answer.replace("\n", "").replace(";", ",")
                    parts = [x.strip() for x in cleaned.split(",")]

                    if len(parts) >= 3:
                        classification = parts[0].upper() if parts[0].upper() in ["YES", "NO"] else None
                        try:
                            confidence = int(parts[1]) if 1 <= int(parts[1]) <= 10 else None
                        except:
                            confidence = None
                        personal_funny = parts[2].upper() if parts[2].upper() in ["YES", "NO"] else None
                    else:
                        print("Invalid response format from Grok.")
                        classification, confidence, personal_funny = None, None, None
                else:
                    #print("No choices in response.")
                    classification, confidence, personal_funny = None, None, None

            except Exception as e:
                #print(f"Error w/ batch {i//batch_size + 1}: {e}")
                classification, confidence, personal_funny = None, None, None

            
            for j in range(batch_size):
                results.append({
                    "joke_num": i + j + 1,
                    "text": statements[j],
                    "true_label": true_labels[j],
                    "llm_classification": classification,
                    "confidence": confidence,
                    "llm_personal_funny": personal_funny,
                    "upvotes": upvotes[j],
                    "downvotes": downvotes[j],
                    "score": scores[j]
                })

            time.sleep(1) #

        
        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_folder, f"grok_{subcategory}_assessments.csv")
        results_df.to_csv(output_file, index=False)
        #print(f"Saved {len(results_df)} entries ,{subcategory}, {output_file}")

def process_all_reddit_with_deepseek(folder_path="humor_dataset_reddit", num_batches=10, api_key="abc"):
    # api config/modlel
    endpoint = "https://api.deepseek.com/v1/chat/completions"
    model = "deepseek-chat"
    batch_size = 3

  
    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    
    output_folder = "llm_assessments"
    os.makedirs(output_folder, exist_ok=True)

    
    dataset_files = [f for f in os.listdir(folder_path) if f.endswith('_augmented.csv')]
    if not dataset_files:
        #print(f"No _augmented.csv files found in {folder_path}.")
        return

    #print(f" {len(dataset_files)} Reddit datasets: {dataset_files}")
    #print(f" API key (redacted): {api_key[:4]}...{api_key[-4:]}")

   
    for dataset_file in dataset_files:
        dataset_path = os.path.join(folder_path, dataset_file)
        #print(" dataset: {dataset_file}")

        
        try:
            df = pd.read_csv(dataset_path)
            df = df.rename(columns={"is_humorous": "true_label"})
            subcategory = dataset_file.replace('_augmented.csv', '')
            #print(f"Loaded {len(df)} entries. {subcategory}")
        except Exception as e:
            #print(f"Error w/ dataset {dataset_path}: {e}")
            continue

       
        results = []
        print(f"Processing {len(df)} entries for {subcategory} with DeepSeek...")

        for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
            batch = df.iloc[i:i+batch_size]
            statements = batch["text"].tolist()
            true_labels = batch["true_label"].tolist()
            upvotes = batch["upvotes"].tolist()
            downvotes = batch["downvotes"].tolist()
            scores = batch["score"].tolist()

            if len(statements) < batch_size:
                statements += ["Placeholder"] * (batch_size - len(statements))
                true_labels += [False] * (batch_size - len(true_labels))
                upvotes += [None] * (batch_size - len(upvotes))
                downvotes += [None] * (batch_size - len(downvotes))
                scores += [None] * (batch_size - len(scores))

            #prompt
            prompt = (
                "For the joke below, provide:\n"
                "1. Joke classification (YES/NO)\n"
                "2. Confidence score (1-10, where 1 is least confident and 10 is most confident)\n"
                "3. Your personal opinion if you find it funny (YES/NO)\n"
                "Format strictly as: YES, 10, YES\n\n"
                f"Joke: {statements[0]}"
            )

            #payload
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a humor classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "temperature": 0.3,  
                "max_tokens": 50
            }

            #request
            try:
                #print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
                response = requests.post(endpoint, headers=headers, json=payload)
                response_json = response.json()
                #print(f"stat code: {response.status_code}")
                #print(f"Response: {json.dumps(response_json, indent=2)}")

                # Check for errors
                if response.status_code == 401:
                    raise Exception(
                        f"Authentication failed (401): Invalid DeepSeek API key. "
                        f"Verify at DeepSeek's developer portal, regenerate if needed, and check permissions."
                    )
                if response.status_code != 200 or "error" in response_json:
                    raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

                # Parse response
                if "choices" in response_json:
                    llm_answer = response_json["choices"][0]["message"]["content"].strip()
                    #print(f"Output: {llm_answer}")

                    
                    cleaned = llm_answer.replace("\n", "").replace(";", ",")
                    parts = [x.strip() for x in cleaned.split(",")]

                    if len(parts) >= 3:
                        classification = parts[0].upper() if parts[0].upper() in ["YES", "NO"] else None
                        try:
                            confidence = int(parts[1]) if 1 <= int(parts[1]) <= 10 else None
                        except:
                            confidence = None
                        personal_funny = parts[2].upper() if parts[2].upper() in ["YES", "NO"] else None
                    else:
                        print("Invalid response format from DeepSeek.")
                        classification, confidence, personal_funny = None, None, None
                else:
                    print("No choices in response.")
                    classification, confidence, personal_funny = None, None, None

            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")
                classification, confidence, personal_funny = None, None, None

            #results
            for j in range(batch_size):
                results.append({
                    "joke_num": i + j + 1,
                    "text": statements[j],
                    "true_label": true_labels[j],
                    "llm_classification": classification,
                    "confidence": confidence,
                    "llm_personal_funny": personal_funny,
                    "upvotes": upvotes[j],
                    "downvotes": downvotes[j],
                    "score": scores[j]
                })

            time.sleep(1)  

        # Save results
        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_folder, f"deepseek_{subcategory}_assessments.csv")
        results_df.to_csv(output_file, index=False)
        #print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")
        
def process_all_reddit_with_chatgpt(folder_path="humor_dataset_reddit", num_batches=10, api_key="sk-proj--YIKbdOX0AfJfeYRaDWP4KAxZwyJIDChbmH6sxo4fzkHDfLyePoj3Umd8UfAAT3BlbkFJoY17U3jAayYQDabP1UmRRljLhp4IWzzcxdRHiZyAo2FOpYYOWY9NY7KSg3Q-mrluvFO2uNFZMA"):

    #config
    endpoint = "https://api.openai.com/v1/chat/completions"
    model = "gpt-3.5-turbo"
    batch_size = 3

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

 
    output_folder = "llm_assessments"
    os.makedirs(output_folder, exist_ok=True)


    dataset_files = [f for f in os.listdir(folder_path) if f.endswith('_augmented.csv')]
    if not dataset_files:
        print(f"No _augmented.csv files found in {folder_path}.")
        return

    print(f"Found {len(dataset_files)} Reddit datasets: {dataset_files}")
    print(f"Using ChatGPT API key (redacted): {api_key[:4]}...{api_key[-4:]}")

    #dataset process
    for dataset_file in dataset_files:
        dataset_path = os.path.join(folder_path, dataset_file)
        print(f"\nProcessing dataset: {dataset_file}")

        # Loading
        try:
            df = pd.read_csv(dataset_path)
            df = df.rename(columns={"is_humorous": "true_label"})
            subcategory = dataset_file.replace('_augmented.csv', '')
            print(f"Loaded {len(df)} entries for {subcategory}")
        except Exception as e:
            print(f"Error loading dataset {dataset_path}: {e}")
            continue

     
        results = []
        print(f"Processing {len(df)} entries for {subcategory} with ChatGPT...")

        for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
            batch = df.iloc[i:i+batch_size]
            statements = batch["text"].tolist()
            true_labels = batch["true_label"].tolist()
            upvotes = batch["upvotes"].tolist()
            downvotes = batch["downvotes"].tolist()
            scores = batch["score"].tolist()

            if len(statements) < batch_size:
                statements += ["Placeholder"] * (batch_size - len(statements))
                true_labels += [False] * (batch_size - len(true_labels))
                upvotes += [None] * (batch_size - len(upvotes))
                downvotes += [None] * (batch_size - len(downvotes))
                scores += [None] * (batch_size - len(scores))

            #prompt
            prompt = (
                "For the joke below, provide:\n"
                "1. Joke classification (YES/NO)\n"
                "2. Confidence score (1-10, where 1 is least confident and 10 is most confident)\n"
                "3. Your personal opinion if you find it funny (YES/NO)\n"
                "Format strictly as: YES, 10, YES\n\n"
                f"Joke: {statements[0]}"
            )

            #payload
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a humor classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "temperature": 0.3,  
                "max_tokens": 50
            }

            #request
            try:
                print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
                response = requests.post(endpoint, headers=headers, json=payload)
                response_json = response.json()
                print(f"Status Code: {response.status_code}")
                print(f"Response: {json.dumps(response_json, indent=2)}")

             
                if response.status_code == 401:
                    raise Exception(
                        f"(401): invalid key "

                    )
                if response.status_code != 200 or "error" in response_json:
                    raise Exception(f" error: Status {response.status_code}, Response: {json.dumps(response_json)}")

                #parsiong
                if "choices" in response_json:
                    llm_answer = response_json["choices"][0]["message"]["content"].strip()
                    print(f"LLM Output: {llm_answer}")

                    
                    cleaned = llm_answer.replace("\n", "").replace(";", ",")
                    parts = [x.strip() for x in cleaned.split(",")]

                    if len(parts) >= 3:
                        classification = parts[0].upper() if parts[0].upper() in ["YES", "NO"] else None
                        try:
                            confidence = int(parts[1]) if 1 <= int(parts[1]) <= 10 else None
                        except:
                            confidence = None
                        personal_funny = parts[2].upper() if parts[2].upper() in ["YES", "NO"] else None
                    else:
                        print("Invalid response format from ChatGPT.")
                        classification, confidence, personal_funny = None, None, None
                else:
                    print("No choices in response.")
                    classification, confidence, personal_funny = None, None, None

            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")
                classification, confidence, personal_funny = None, None, None

            #reuslts
            for j in range(batch_size):
                results.append({
                    "joke_num": i + j + 1,
                    "text": statements[j],
                    "true_label": true_labels[j],
                    "llm_classification": classification,
                    "confidence": confidence,
                    "llm_personal_funny": personal_funny,
                    "upvotes": upvotes[j],
                    "downvotes": downvotes[j],
                    "score": scores[j]
                })

            time.sleep(1)  

        
        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_folder, f"chatgpt_{subcategory}_assessments.csv")
        results_df.to_csv(output_file, index=False)
        print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")

def process_all_reddit_with_llama(folder_path="humor_dataset_reddit", num_batches=10, api_key="abc"):
    client = Together(api_key=api_key)
    model = "meta-llama/LLaMA-3.3-70B-Instruct-Turbo-Free"
    batch_size = 3

   
   
    output_folder = "llm_assessments"
    os.makedirs(output_folder, exist_ok=True)

 
    dataset_files = [f for f in os.listdir(folder_path) if f.endswith('_augmented.csv')]
    if not dataset_files:
        print(f"No _augmented.csv files found in {folder_path}.")
        return

    #print(f"Found {len(dataset_files)} Reddit datasets: {dataset_files}")
    #print(f"Using LLaMA API key (redacted): {api_key[:4]}...{api_key[-4:]}")

    #dataset  processing
    for dataset_file in dataset_files:
        dataset_path = os.path.join(folder_path, dataset_file)

        try:
            df = pd.read_csv(dataset_path)
            df = df.rename(columns={"is_humorous": "true_label"})
            subcategory = dataset_file.replace('_augmented.csv', '')
            #print(f"Loaded {len(df)} entries for {subcategory}")
        except Exception as e:
            #print(f"Error loading dataset {dataset_path}: {e}")
            continue


        results = []
        #print(f" {len(df)} entries for {subcategory}")

        for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
            batch = df.iloc[i:i+batch_size]
            statements = batch["text"].tolist()
            true_labels = batch["true_label"].tolist()
            upvotes = batch["upvotes"].tolist()
            downvotes = batch["downvotes"].tolist()
            scores = batch["score"].tolist()

            if len(statements) < batch_size:
                statements += ["Placeholder"] * (batch_size - len(statements))
                true_labels += [False] * (batch_size - len(true_labels))
                upvotes += [None] * (batch_size - len(upvotes))
                downvotes += [None] * (batch_size - len(downvotes))
                scores += [None] * (batch_size - len(scores))

            #prompt
            prompt = (
                "For the joke below, provide:\n"
                "1. Joke classification (YES/NO)\n"
                "2. Confidence score (1-10, where 1 is least confident and 10 is most confident)\n"
                "3. Your personal opinion if you find it funny (YES/NO)\n"
                "Format strictly as: YES, 10, YES\n\n"
                f"Joke: {statements[0]}"
            )

            #request
            try:
                #print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a humor classification assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False,
                    temperature=0.3,  
                    max_tokens=50
                )

                #parse
                if hasattr(response, 'choices') and response.choices:
                    llm_answer = response.choices[0].message.content.strip()
                    #print(f" Output: {llm_answer}")

                   
                    cleaned = llm_answer.replace("\n", "").replace(";", ",")
                    parts = [x.strip() for x in cleaned.split(",")]

                    if len(parts) >= 3:
                        classification = parts[0].upper() if parts[0].upper() in ["YES", "NO"] else None
                        try:
                            confidence = int(parts[1]) if 1 <= int(parts[1]) <= 10 else None
                        except:
                            confidence = None
                        personal_funny = parts[2].upper() if parts[2].upper() in ["YES", "NO"] else None
                    else:
                        print("Invalid response format from LLaMA.")
                        classification, confidence, personal_funny = None, None, None
                else:
                    #print("n/a.")
                    classification, confidence, personal_funny = None, None, None

            except Exception as e:
                #print(f"error in batch {i//batch_size + 1}: {e}")
                if "401" in str(e):
                    print(
                        f" (401):invalid key. "
                    
                    )
                classification, confidence, personal_funny = None, None, None

            #results
            for j in range(batch_size):
                results.append({
                    "joke_num": i + j + 1,
                    "text": statements[j],
                    "true_label": true_labels[j],
                    "llm_classification": classification,
                    "confidence": confidence,
                    "llm_personal_funny": personal_funny,
                    "upvotes": upvotes[j],
                    "downvotes": downvotes[j],
                    "score": scores[j]
                })

            time.sleep(2)  #shoundt be lower than 2, free version is very stingy.


        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_folder, f"llama_{subcategory}_assessments.csv")
        results_df.to_csv(output_file, index=False)
        #print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")

num_b = 20180/3

#process_all_reddit_with_grok(folder_path="humor_dataset_reddit", num_batches=6727)
#process_all_reddit_with_deepseek(folder_path="humor_dataset_reddit", num_batches=6727)
#process_all_reddit_with_chatgpt(folder_path="humor_dataset_reddit", num_batches=6727)
#process_all_reddit_with_llama(folder_path="humor_dataset_reddit", num_batches=6727)


#llama?

#colbert processing:

def process_colbert_with_grok(num_batches=10, api_key="abc"):

    endpoint = "https://api.x.ai/v1/chat/completions"
    model = "grok-3-latest"
    batch_size = 3

    

   
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    #load colbert from hf
    try:
        df = pd.read_csv("hf://datasets/CreativeLang/ColBERT_Humor_Detection/dataset.csv")
        df = df.rename(columns={"humor": "true_label"})
        df["upvotes"] = None
        df["downvotes"] = None
        df["score"] = None
        subcategory = "colbert"
        #print(f"Loaded {len(df)} entries. {subcategory}")
    except Exception as e:
        print(f"Error loading ColBERT dataset: {e}")
        return

    output_folder = "llm_assessments"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"grok_{subcategory}_assessments.csv")

 
    results = []
    #print(f API key (redacted): {api_key[:4]}...{api_key[-4:]}")
    #print(f"Processing {len(df)}, {subcategory} with Grok...")

    for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
        batch = df.iloc[i:i+batch_size]
        statements = batch["text"].tolist()
        true_labels = batch["true_label"].tolist()
        upvotes = batch["upvotes"].tolist()
        downvotes = batch["downvotes"].tolist()
        scores = batch["score"].tolist()

        if len(statements) < batch_size:
            statements += ["Placeholder"] * (batch_size - len(statements))
            true_labels += [False] * (batch_size - len(true_labels))
            upvotes += [None] * (batch_size - len(upvotes))
            downvotes += [None] * (batch_size - len(downvotes))
            scores += [None] * (batch_size - len(scores))

        #prompting
        prompt = (
            "For the joke below, provide:\n"
            "1. Joke classification (YES/NO)\n"
            "2. Confidence score (1-10, where 1 is least confident and 10 is most confident)\n"
            "3. Your personal opinion if you find it funny (YES/NO)\n"
            "Format strictly as: YES, 10, YES\n\n"
            f"Joke: {statements[0]}"
        )

        #payload
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a humor classification assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": 0,
            "max_tokens": 50
        }

        #request
        try:
            #print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
            response = requests.post(endpoint, headers=headers, json=payload)
            response_json = response.json()
            #print(f"Status Code: {response.status_code}")
            #print(f"Response: {json.dumps(response_json, indent=2)}")

            # Check for errors
            if response.status_code == 401:
                raise Exception(
                    f"authentication failed (401): invalid API key. "
                
                )
            if response.status_code != 200 or "error" in response_json:
                raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

            #parsing
            if "choices" in response_json:
                llm_answer = response_json["choices"][0]["message"]["content"].strip()
                cleaned = llm_answer.replace("\n", "").replace(";", ",")
                parts = [x.strip() for x in cleaned.split(",")]

                if len(parts) >= 3:
                    classification = parts[0].upper() if parts[0].upper() in ["YES", "NO"] else None
                    try:
                        confidence = int(parts[1]) if 1 <= int(parts[1]) <= 10 else None
                    except:
                        confidence = None
                    personal_funny = parts[2].upper() if parts[2].upper() in ["YES", "NO"] else None
                else:
                    print("Invalid response.")
                    classification, confidence, personal_funny = None, None, None
            else:
                print("No choices in response.")
                classification, confidence, personal_funny = None, None, None

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            classification, confidence, personal_funny = None, None, None

        #results
        for j in range(batch_size):
            results.append({
                "joke_num": i + j + 1,
                "text": statements[j],
                "true_label": true_labels[j],
                "llm_classification": classification,
                "confidence": confidence,
                "llm_personal_funny": personal_funny,
                "upvotes": upvotes[j],
                "downvotes": downvotes[j],
                "score": scores[j]
            })

        time.sleep(1) 

   
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    #print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")

def process_colbert_with_deepseek(num_batches=20000, api_key="abc"):
    endpoint = "https://api.deepseek.com/v1/chat/completions"
    model = "deepseek-chat"
    batch_size = 3

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    #hf colbert load in
    try:
        df = pd.read_csv("hf://datasets/CreativeLang/ColBERT_Humor_Detection/dataset.csv")
        df = df.rename(columns={"humor": "true_label"})
        df["upvotes"] = None
        df["downvotes"] = None
        df["score"] = None
        subcategory = "colbert"
        #print(f" {len(df)} entries for {subcategory}")
    except Exception as e:
        #print(f"error loading ColBERT: {e}")
        return

    
    output_folder = "llm_assessments"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"deepseek_{subcategory}_assessments.csv")

    #dataset processing
    results = []
    #print(f"Using DeepSeek API key (redacted): {api_key[:4]}...{api_key[-4:]}")
    #print(f"Processing {len(df)} entries for {subcategory} with DeepSeek...")

    for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
        batch = df.iloc[i:i+batch_size]
        statements = batch["text"].tolist()
        true_labels = batch["true_label"].tolist()
        upvotes = batch["upvotes"].tolist()
        downvotes = batch["downvotes"].tolist()
        scores = batch["score"].tolist()

        if len(statements) < batch_size:
            statements += ["Placeholder"] * (batch_size - len(statements))
            true_labels += [False] * (batch_size - len(true_labels))
            upvotes += [None] * (batch_size - len(upvotes))
            downvotes += [None] * (batch_size - len(downvotes))
            scores += [None] * (batch_size - len(scores))

        #prompting
        prompt = (
            "For the joke below, provide:\n"
            "1. Joke classification (YES/NO)\n"
            "2. Confidence score (1-10, where 1 is least confident and 10 is most confident)\n"
            "3. Your personal opinion if you find it funny (YES/NO)\n"
            "Format strictly as: YES, 10, YES\n\n"
            f"Joke: {statements[0]}"
        )

        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a humor classification assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 50
        }

        
        try:
            #print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
            response = requests.post(endpoint, headers=headers, json=payload)
            response_json = response.json()
            #print(f"Status Code: {response.status_code}")
            #print(f"Response: {json.dumps(response_json, indent=2)}")

            # Check for errors
            if response.status_code == 401:
                raise Exception(
                    f"Authentication failed (401): Invalid API key. "
                   
                )
            if response.status_code != 200 or "error" in response_json:
                raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

            #parsing
            if "choices" in response_json:
                llm_answer = response_json["choices"][0]["message"]["content"].strip()
                #print(f"LLM Output: {llm_answer}")

                # Parse the output (YES, 10, YES format)
                cleaned = llm_answer.replace("\n", "").replace(";", ",")
                parts = [x.strip() for x in cleaned.split(",")]

                if len(parts) >= 3:
                    classification = parts[0].upper() if parts[0].upper() in ["YES", "NO"] else None
                    try:
                        confidence = int(parts[1]) if 1 <= int(parts[1]) <= 10 else None
                    except:
                        confidence = None
                    personal_funny = parts[2].upper() if parts[2].upper() in ["YES", "NO"] else None
                else:
                    print("Invalid response format from DeepSeek.")
                    classification, confidence, personal_funny = None, None, None
            else:
                print("No choices in response.")
                classification, confidence, personal_funny = None, None, None

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            classification, confidence, personal_funny = None, None, None

        
        for j in range(batch_size):
            results.append({
                "joke_num": i + j + 1,
                "text": statements[j],
                "true_label": true_labels[j],
                "llm_classification": classification,
                "confidence": confidence,
                "llm_personal_funny": personal_funny,
                "upvotes": upvotes[j],
                "downvotes": downvotes[j],
                "score": scores[j]
            })

        #time.sleep(1)  #rate limits less strict

   
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    #print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")

def process_colbert_with_chatgpt(num_batches=10, api_key="abc"):
  
    endpoint = "https://api.openai.com/v1/chat/completions"
    model = "gpt-3.5-turbo"
    batch_size = 3

   
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    #hf loading colbert
    try:
        df = pd.read_csv("hf://datasets/CreativeLang/ColBERT_Humor_Detection/dataset.csv")
        df = df.rename(columns={"humor": "true_label"})
        df["upvotes"] = None
        df["downvotes"] = None
        df["score"] = None
        subcategory = "colbert"
        print(f"Loaded {len(df)} entries for {subcategory}")
    except Exception as e:
        #print(f"Error loading ColBERT: {e}")
        return


    output_folder = "llm_assessments"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"chatgpt_{subcategory}_assessments.csv")

  
    results = []
    #print(f"API key (): {api_key[:4]}...{api_key[-4:]}")
    #print(f"Processing {len(df)} entries for {subcategory} with ChatGPT...")

    for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
        batch = df.iloc[i:i+batch_size]
        statements = batch["text"].tolist()
        true_labels = batch["true_label"].tolist()
        upvotes = batch["upvotes"].tolist()
        downvotes = batch["downvotes"].tolist()
        scores = batch["score"].tolist()

        if len(statements) < batch_size:
            statements += ["Placeholder"] * (batch_size - len(statements))
            true_labels += [False] * (batch_size - len(true_labels))
            upvotes += [None] * (batch_size - len(upvotes))
            downvotes += [None] * (batch_size - len(downvotes))
            scores += [None] * (batch_size - len(scores))

        #prompting
        prompt = (
            "For the joke below, provide:\n"
            "1. Joke classification (YES/NO)\n"
            "2. Confidence score (1-10, where 1 is least confident and 10 is most confident)\n"
            "3. Your personal opinion if you find it funny (YES/NO)\n"
            "Format strictly as: YES, 10, YES\n\n"
            f"Joke: {statements[0]}"
        )

        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a humor classification assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 50
        }

        try:
            #print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
            response = requests.post(endpoint, headers=headers, json=payload)
            response_json = response.json()
            #print(f"Status Code: {response.status_code}")
            #print(f"Response: {json.dumps(response_json, indent=2)}")

            # Check for errors
            if response.status_code == 401:
                raise Exception(
                    f"Authentication failed (401): Invalid ChatGPT API key. "
                    f"Verify at OpenAI's developer portal, regenerate if needed, and check permissions."
                )
            if response.status_code != 200 or "error" in response_json:
                raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

            #parsing
            if "choices" in response_json:
                llm_answer = response_json["choices"][0]["message"]["content"].strip()
                #print(f"LLM Output: {llm_answer}")

               
                cleaned = llm_answer.replace("\n", "").replace(";", ",")
                parts = [x.strip() for x in cleaned.split(",")]

                if len(parts) >= 3:
                    classification = parts[0].upper() if parts[0].upper() in ["YES", "NO"] else None
                    try:
                        confidence = int(parts[1]) if 1 <= int(parts[1]) <= 10 else None
                    except:
                        confidence = None
                    personal_funny = parts[2].upper() if parts[2].upper() in ["YES", "NO"] else None
                else:
                    print("Invalid response format from ChatGPT.")
                    classification, confidence, personal_funny = None, None, None
            else:
                print("No choices in response.")
                classification, confidence, personal_funny = None, None, None

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            classification, confidence, personal_funny = None, None, None

    
        for j in range(batch_size):
            results.append({
                "joke_num": i + j + 1,
                "text": statements[j],
                "true_label": true_labels[j],
                "llm_classification": classification,
                "confidence": confidence,
                "llm_personal_funny": personal_funny,
                "upvotes": upvotes[j],
                "downvotes": downvotes[j],
                "score": scores[j]
            })

        time.sleep(1) 

  
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    #print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")

def process_colbert_with_llama(num_batches=10, api_key="abc"):

    client = Together(api_key=api_key)
    model = "meta-llama/LLaMA-3.3-70B-Instruct-Turbo-Free"
    batch_size = 3


    #hf colbert load in
    try:
        df = pd.read_csv("hf://datasets/CreativeLang/ColBERT_Humor_Detection/dataset.csv")
        df = df.rename(columns={"humor": "true_label"})
        df["upvotes"] = None
        df["downvotes"] = None
        df["score"] = None
        subcategory = "colbert"
        #print(f"Loaded {len(df)} entries for {subcategory}")
    except Exception as e:
        #print(f"Error loading ColBERT dataset: {e}")
        return

   
    output_folder = "llm_assessments"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"llama_{subcategory}_assessments.csv")

    #procesing
    results = []
    #print(f"Using LLaMA API key (redacted): {api_key[:4]}...{api_key[-4:]}")
    #print(f"Processing {len(df)} entries for {subcategory} with LLaMA...")

    for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
        batch = df.iloc[i:i+batch_size]
        statements = batch["text"].tolist()
        true_labels = batch["true_label"].tolist()
        upvotes = batch["upvotes"].tolist()
        downvotes = batch["downvotes"].tolist()
        scores = batch["score"].tolist()

        if len(statements) < batch_size:
            statements += ["Placeholder"] * (batch_size - len(statements))
            true_labels += [False] * (batch_size - len(true_labels))
            upvotes += [None] * (batch_size - len(upvotes))
            downvotes += [None] * (batch_size - len(downvotes))
            scores += [None] * (batch_size - len(scores))

        #prompting
        prompt = (
            "For the joke below, provide:\n"
            "1. Joke classification (YES/NO)\n"
            "2. Confidence score (1-10, where 1 is least confident and 10 is most confident)\n"
            "3. Your personal opinion if you find it funny (YES/NO)\n"
            "Format strictly as: YES, 10, YES\n\n"
            f"Joke: {statements[0]}"
        )

     
        try:
            #print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a humor classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.3,  
                max_tokens=50
            )

            #parsing
            if hasattr(response, 'choices') and response.choices:
                llm_answer = response.choices[0].message.content.strip()
                #print(f"LLM Output: {llm_answer}")

                
                cleaned = llm_answer.replace("\n", "").replace(";", ",")
                parts = [x.strip() for x in cleaned.split(",")]

                if len(parts) >= 3:
                    classification = parts[0].upper() if parts[0].upper() in ["YES", "NO"] else None
                    try:
                        confidence = int(parts[1]) if 1 <= int(parts[1]) <= 10 else None
                    except:
                        confidence = None
                    personal_funny = parts[2].upper() if parts[2].upper() in ["YES", "NO"] else None
                else:
                    print("Invalid response format from LLaMA.")
                    classification, confidence, personal_funny = None, None, None
            else:
                print("No valid choices in response.")
                classification, confidence, personal_funny = None, None, None

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            if "401" in str(e):
                print(
                    f"Authentication failed (401): Invalid LLaMA API key. "
                    f"Verify at Together AI's developer portal (https://api.together.ai/), regenerate if needed, and check permissions."
                )
            classification, confidence, personal_funny = None, None, None

      
        for j in range(batch_size):
            results.append({
                "joke_num": i + j + 1,
                "text": statements[j],
                "true_label": true_labels[j],
                "llm_classification": classification,
                "confidence": confidence,
                "llm_personal_funny": personal_funny,
                "upvotes": upvotes[j],
                "downvotes": downvotes[j],
                "score": scores[j]
            })

        #time.sleep(1)  

   
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    #print(f"Saved {len(results `_df)} entries for {subcategory} to {output_file}")
    
    
    """Process the ColBERT dataset with OLMo via Together API for humor classification."""
    # Together API configuration for OLMo
    endpoint = "https://api.together.ai/v1/chat/completions"
    model = "allenai/OLMo-7B"
    batch_size = 1

    

    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Load ColBERT dataset
    try:
        df = pd.read_csv("hf://datasets/CreativeLang/ColBERT_Humor_Detection/dataset.csv")
        df = df.rename(columns={"humor": "true_label"})
        df["upvotes"] = None
        df["downvotes"] = None
        df["score"] = None
        subcategory = "colbert"
        print(f"Loaded {len(df)} entries for {subcategory}")
    except Exception as e:
        print(f"Error loading ColBERT dataset: {e}")
        return

    # Prepare output folder
    output_folder = "llm_assessments"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"olmo_{subcategory}_assessments.csv")

    # Process dataset
    results = []
    print(f"Using OLMo API key (redacted): {api_key[:4]}...{api_key[-4:]}")
    print(f"Processing {len(df)} entries for {subcategory} with OLMo...")

    for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
        batch = df.iloc[i:i+batch_size]
        statements = batch["text"].tolist()
        true_labels = batch["true_label"].tolist()
        upvotes = batch["upvotes"].tolist()
        downvotes = batch["downvotes"].tolist()
        scores = batch["score"].tolist()

        if len(statements) < batch_size:
            statements += ["Placeholder"] * (batch_size - len(statements))
            true_labels += [False] * (batch_size - len(true_labels))
            upvotes += [None] * (batch_size - len(upvotes))
            downvotes += [None] * (batch_size - len(downvotes))
            scores += [None] * (batch_size - len(scores))

        # Create prompt
        prompt = (
            "For the joke below, provide:\n"
            "1. Joke classification (YES/NO)\n"
            "2. Confidence score (1-10, where 1 is least confident and 10 is most confident)\n"
            "3. Your personal opinion if you find it funny (YES/NO)\n"
            "Format strictly as: YES, 10, YES\n\n"
            f"Joke: {statements[0]}"
        )

        # Payload
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a humor classification assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 50
        }

        # Send request
        try:
            print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
            response = requests.post(endpoint, headers=headers, json=payload)
            response_json = response.json()
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response_json, indent=2)}")

            # Check for errors
            if response.status_code == 401:
                raise Exception(
                    f"Authentication failed (401): Invalid OLMo API key. "
                    f"Verify at Together AI's developer portal (https://api.together.ai/), regenerate if needed, and check permissions."
                )
            if response.status_code != 200 or "error" in response_json:
                raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

            # Parse response
            if "choices" in response_json:
                llm_answer = response_json["choices"][0]["message"]["content"].strip()
                print(f"LLM Output: {llm_answer}")

                # Parse the output (YES, 10, YES format)
                cleaned = llm_answer.replace("\n", "").replace(";", ",")
                parts = [x.strip() for x in cleaned.split(",")]

                if len(parts) >= 3:
                    classification = parts[0].upper() if parts[0].upper() in ["YES", "NO"] else None
                    try:
                        confidence = int(parts[1]) if 1 <= int(parts[1]) <= 10 else None
                    except:
                        confidence = None
                    personal_funny = parts[2].upper() if parts[2].upper() in ["YES", "NO"] else None
                else:
                    print("Invalid response format from OLMo.")
                    classification, confidence, personal_funny = None, None, None
            else:
                print("No choices in response.")
                classification, confidence, personal_funny = None, None, None

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            classification, confidence, personal_funny = None, None, None

        # Store results
        for j in range(batch_size):
            results.append({
                "joke_num": i + j + 1,
                "text": statements[j],
                "true_label": true_labels[j],
                "llm_classification": classification,
                "confidence": confidence,
                "llm_personal_funny": personal_funny,
                "upvotes": upvotes[j],
                "downvotes": downvotes[j],
                "score": scores[j]
            })

        time.sleep(2)  # Avoid rate limits

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")
#process_colbert_with_grok(num_batches=25000)
#process_colbert_with_deepseek(num_batches=20000)
#process_colbert_with_chatgpt(num_batches=25000)
process_colbert_with_llama(num_batches=20000)