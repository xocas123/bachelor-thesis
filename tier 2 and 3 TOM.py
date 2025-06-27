# tier_2_3_analysis.py

import random
import pandas as pd
import os
import requests
import json
import re
import time
from together import Together



INPUT_FOLDER = "llm_assessments"
OUTPUT_FOLDER = "tier_2_outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#api keys omitted for placholders

#deepseek
def process_all_reddit_with_deepseek_tier2(tier1_folder="llm_assessments", num_batches=6727, batch_size = 3, api_key="placeholder"):
    
    endpoint = "https://api.deepseek.com/v1/chat/completions"
    model = "deepseek-chat"
   

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    output_folder = "tier_2_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Find all Reddit Tier 1 assessment files
    assessment_files = [f for f in os.listdir(tier1_folder) if f.startswith('deepseek_') and f.endswith('_assessments.csv') and 'colbert' not in f]
    if not assessment_files:
        print(f"No DeepSeek assessment files found in {tier1_folder}.")
        return

    #print(f"Found {len(assessment_files)} Reddit assessment files: {assessment_files}")
    #print(f"Using DeepSeek API key (redacted): {api_key[:4]}...{api_key[-4:]}")
    for assessment_file in assessment_files:
        assessment_path = os.path.join(tier1_folder, assessment_file)
        print(f"\nProcessing assessment file: {assessment_file}")

     
        try:
            df = pd.read_csv(assessment_path)
            subcategory = assessment_file.replace('deepseek_', '').replace('_assessments.csv', '')
            print(f"Loaded {len(df)} entries for {subcategory}")
        except Exception as e:
            print(f"Error loading assessment file {assessment_path}: {e}")
            continue

       
        results = []
        #print(f"Processing {len(df)} entries for {subcategory} with DeepSeek for Tier 2 quantitative...")

        for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
            batch = df.iloc[i:i+batch_size]
            statements = batch["text"].tolist()
            true_labels = batch["true_label"].tolist()
            llm_classifications = batch["llm_classification"].tolist()
            confidences = batch["confidence"].tolist()
            llm_personal_funnies = batch["llm_personal_funny"].tolist()
            upvotes = batch["upvotes"].tolist()
            downvotes = batch["downvotes"].tolist()
            scores = batch["score"].tolist()

            if len(statements) < batch_size:
                statements += ["Placeholder"] * (batch_size - len(statements))
                true_labels += [False] * (batch_size - len(statements))
                llm_classifications += [None] * (batch_size - len(statements))
                confidences += [None] * (batch_size - len(statements))
                llm_personal_funnies += [None] * (batch_size - len(statements))
                upvotes += [None] * (batch_size - len(statements))
                downvotes += [None] * (batch_size - len(statements))
                scores += [None] * (batch_size - len(statements))

      
            prompt = (
                "For the text below, provide scores (1–10, where 1 is lowest and 10 is highest) for:\n"
                "1. Funniness (how humorous you find it)\n"
                "2. Offensiveness (how likely to offend)\n"
                "3. Originality (how novel or creative)\n"
                "4. Appropriateness (how suitable for general audiences)\n"
                "5. Clarity (how clear and understandable the humor is)\n"
                "Format strictly as: Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V\n\n"
                f"Text: {statements[0]}"
            )

    
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a humor analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 100
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
                        f"Authentication failed (401): Invalid DeepSeek API key. "
                        f"Verify at DeepSeek's developer portal, regenerate if needed, and check permissions."
                    )
                if response.status_code != 200 or "error" in response_json:
                    raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

                # Parse response
                if "choices" in response_json:
                    llm_answer = response_json["choices"][0]["message"]["content"].strip()
                    #print(f"DeepSeek Output: {llm_answer}")

                    # Parse the output (Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V)
                    parts = [p.strip() for p in llm_answer.replace("\n", "").split(",")]
                    if len(parts) >= 5 and all(":" in p for p in parts):
                        funniness = int(parts[0].split(":")[1].strip()) if parts[0].split(":")[1].strip().isdigit() else None
                        offensiveness = int(parts[1].split(":")[1].strip()) if parts[1].split(":")[1].strip().isdigit() else None
                        originality = int(parts[2].split(":")[1].strip()) if parts[2].split(":")[1].strip().isdigit() else None
                        appropriateness = int(parts[3].split(":")[1].strip()) if parts[3].split(":")[1].strip().isdigit() else None
                        clarity = int(parts[4].split(":")[1].strip()) if parts[4].split(":")[1].strip().isdigit() else None
                    else:
                        print(f"Invalid response format from DeepSeek: {llm_answer}")
                        funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None
                else:
                    print("No choices in response.")
                    funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")
                funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

            # Store results
            for j in range(batch_size):
                results.append({
                    "joke_num": i + j + 1,
                    "text": statements[j],
                    "true_label": true_labels[j],
                    "llm_classification": llm_classifications[j],
                    "confidence": confidences[j],
                    "llm_personal_funny": llm_personal_funnies[j],
                    "funniness": funniness,
                    "offensiveness": offensiveness,
                    "originality": originality,
                    "appropriateness": appropriateness,
                    "clarity": clarity,
                    "upvotes": upvotes[j],
                    "downvotes": downvotes[j],
                    "score": scores[j]
                })

            time.sleep(1)  

      
        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_folder, f"deepseek_{subcategory}_tier_2_quantitative.csv")
        results_df.to_csv(output_file, index=False)
        #print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")



  
    endpoint = "https://api.deepseek.com/v1/chat/completions"
    model = "deepseek-chat"
    batch_size = 3
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

 
    try:
        df = pd.read_csv("hf://datasets/CreativeLang/ColBERT_Humor_Detection/dataset.csv")
        df = df.rename(columns={"humor": "true_label"})
        df["upvotes"] = None
        df["downvotes"] = None
        df["score"] = None
        subcategory = "colbert"
        #print(f"Loaded {len(df)} entries for {subcategory}")
    except Exception as e:
        print(f"Error loading ColBERT dataset: {e}")
        return

    output_folder = "tier_2_outputs"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"deepseek_{subcategory}_tier_2_quantitative.csv")

   
    results = []
    print(f"Using DeepSeek API key (redacted): {api_key[:4]}...{api_key[-4:]}")
    print(f"Processing {len(df)} entries for {subcategory} with DeepSeek for Tier 2 quantitative...")

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

      
        prompt = (
            "For the text below, provide scores (1–10, where 1 is lowest and 10 is highest) for:\n"
            "1. Funniness (how humorous you find it)\n"
            "2. Offensiveness (how likely to offend)\n"
            "3. Originality (how novel or creative)\n"
            "4. Appropriateness (how suitable for general audiences)\n"
            "5. Clarity (how clear and understandable the humor is)\n"
            "Format strictly as: Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V\n\n"
            f"Text: {statements[0]}"
        )

    
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a humor analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 100
        }

   
        try:
            #print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
            response = requests.post(endpoint, headers=headers, json=payload)
            response_json = response.json()
            #print(f"Status Code: {response.status_code}")
            #print(f"Response: {json.dumps(response_json, indent=2)}")

           
            if response.status_code == 401:
                raise Exception(
                    f"(401). "
                )
            if response.status_code != 200 or "error" in response_json:
                raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

           
            if "choices" in response_json:
                llm_answer = response_json["choices"][0]["message"]["content"].strip()
                #print(f"DeepSeek Output: {llm_answer}")

                # Parse the output (Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V)
                parts = [p.strip() for p in llm_answer.replace("\n", "").split(",")]
                if len(parts) >= 5 and all(":" in p for p in parts):
                    funniness = int(parts[0].split(":")[1].strip()) if parts[0].split(":")[1].strip().isdigit() else None
                    offensiveness = int(parts[1].split(":")[1].strip()) if parts[1].split(":")[1].strip().isdigit() else None
                    originality = int(parts[2].split(":")[1].strip()) if parts[2].split(":")[1].strip().isdigit() else None
                    appropriateness = int(parts[3].split(":")[1].strip()) if parts[3].split(":")[1].strip().isdigit() else None
                    clarity = int(parts[4].split(":")[1].strip()) if parts[4].split(":")[1].strip().isdigit() else None
                else:
                    print("Invalid response format from DeepSeek.")
                    funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None
            else:
                print("No choices in response.")
                funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

        # Store results
        for j in range(batch_size):
            results.append({
                "joke_num": i + j + 1,
                "text": statements[j],
                "true_label": true_labels[j],
                "funniness": funniness,
                "offensiveness": offensiveness,
                "originality": originality,
                "appropriateness": appropriateness,
                "clarity": clarity,
                "upvotes": upvotes[j],
                "downvotes": downvotes[j],
                "score": scores[j]
            })

        time.sleep(2)  

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    #print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")
def process_colbert_with_deepseek_tier2(tier1_folder="llm_assessments", num_batches=20000,batch_size = 3,  api_key="placeholder"):
  
    endpoint = "https://api.deepseek.com/v1/chat/completions"
    model = "deepseek-chat"
  
  

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    
    assessment_file = os.path.join(tier1_folder, "deepseek_colbert_assessments.csv")
    try:
        df = pd.read_csv(assessment_file)
        subcategory = "colbert"
        #print(f"Loaded {len(df)} entries for {subcategory}")
    except Exception as e:
        print(f"Error loading ColBERT assessment file {assessment_file}: {e}")
        return

    output_folder = "tier_2_outputs"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"deepseek_{subcategory}_tier_2_quantitative.csv")

    # Process assessment data
    results = []
    #print(f"Using DeepSeek API key (redacted): {api_key[:4]}...{api_key[-4:]}")
    #print(f"Processing {len(df)} entries for {subcategory} with DeepSeek for Tier 2 quantitative...")

    for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
        batch = df.iloc[i:i+batch_size]
        statements = batch["text"].tolist()
        true_labels = batch["true_label"].tolist()
        llm_classifications = batch["llm_classification"].tolist()
        confidences = batch["confidence"].tolist()
        llm_personal_funnies = batch["llm_personal_funny"].tolist()
        upvotes = batch["upvotes"].tolist()
        downvotes = batch["downvotes"].tolist()
        scores = batch["score"].tolist()

        if len(statements) < batch_size:
            statements += ["Placeholder"] * (batch_size - len(statements))
            true_labels += [False] * (batch_size - len(statements))
            llm_classifications += [None] * (batch_size - len(statements))
            confidences += [None] * (batch_size - len(statements))
            llm_personal_funnies += [None] * (batch_size - len(statements))
            upvotes += [None] * (batch_size - len(statements))
            downvotes += [None] * (batch_size - len(statements))
            scores += [None] * (batch_size - len(statements))

      
        prompt = (
            "For the text below, provide scores (1–10, where 1 is lowest and 10 is highest) Strictly provide only scores and nothing else. for:\n"
            "1. Funniness (how humorous you find it)\n"
            "2. Offensiveness (how likely to offend)\n"
            "3. Originality (how novel or creative)\n"
            "4. Appropriateness (how suitable for general audiences)\n"
            "5. Clarity (how clear and understandable the humor is)\n"
            "Format strictly as and nothing more than: Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V\n\n"
            f"Text: {statements[0]}"
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a humor analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 100
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
                    f"Authentication failed (401): Invalid DeepSeek API key. "
                    f"Verify at DeepSeek's developer portal, regenerate if needed, and check permissions."
                )
            if response.status_code != 200 or "error" in response_json:
                raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

            # Parse response
            if "choices" in response_json:
                llm_answer = response_json["choices"][0]["message"]["content"].strip()
                #print(f"DeepSeek Output: {llm_answer}")

                # Parse the output (Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V)
                parts = [p.strip() for p in llm_answer.replace("\n", "").split(",")]
                if len(parts) >= 5 and all(":" in p for p in parts):
                    funniness = int(parts[0].split(":")[1].strip()) if parts[0].split(":")[1].strip().isdigit() else None
                    offensiveness = int(parts[1].split(":")[1].strip()) if parts[1].split(":")[1].strip().isdigit() else None
                    originality = int(parts[2].split(":")[1].strip()) if parts[2].split(":")[1].strip().isdigit() else None
                    appropriateness = int(parts[3].split(":")[1].strip()) if parts[3].split(":")[1].strip().isdigit() else None
                    clarity = int(parts[4].split(":")[1].strip()) if parts[4].split(":")[1].strip().isdigit() else None
                else:
                    print(f"Invalid response format from DeepSeek: {llm_answer}")
                    funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None
            else:
                print("No choices in response.")
                funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

        # Store results
        for j in range(batch_size):
            results.append({
                "joke_num": i + j + 1,
                "text": statements[j],
                "true_label": true_labels[j],
                "llm_classification": llm_classifications[j],
                "confidence": confidences[j],
                "llm_personal_funny": llm_personal_funnies[j],
                "funniness": funniness,
                "offensiveness": offensiveness,
                "originality": originality,
                "appropriateness": appropriateness,
                "clarity": clarity,
                "upvotes": upvotes[j],
                "downvotes": downvotes[j],
                "score": scores[j]
            })

        

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    #print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")
    
#process_all_reddit_with_deepseek_tier2()
#process_all_reddit_with_deepseek_tier2()
#process_colbert_with_deepseek_tier2()


#-investigate a little bit more
def process_all_reddit_with_grok_tier2(tier1_folder="llm_assessments", num_batches=6727, batch_size = 3, api_key="placeholder"):
   
    endpoint = "https://api.x.ai/v1/chat/completions"
    model = "grok-3-latest"
    batch_size = 1
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }


    output_folder = "tier_2_outputs"
    os.makedirs(output_folder, exist_ok=True)

  
    assessment_files = [f for f in os.listdir(tier1_folder) if f.startswith('grok_') and f.endswith('_assessments.csv') and 'colbert' not in f]
    if not assessment_files:
        print(f"No Grok assessment files found in {tier1_folder}.")
        return

    print(f"Found {len(assessment_files)} Reddit assessment files: {assessment_files}")
    #print(f"Using Grok API key (redacted): {api_key[:4]}...{api_key[-4:]}")


    for assessment_file in assessment_files:
        assessment_path = os.path.join(tier1_folder, assessment_file)
        print(f"\nProcessing assessment file: {assessment_file}")

        # Load Tier 1 assessment data
        try:
            df = pd.read_csv(assessment_path)
            subcategory = assessment_file.replace('grok_', '').replace('_assessments.csv', '')
            print(f"Loaded {len(df)} entries for {subcategory}")
        except Exception as e:
            print(f"Error loading assessment file {assessment_path}: {e}")
            continue
        results = []
        print(f"Processing {len(df)} entries for {subcategory} with Grok for Tier 2 quantitative...")

        for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
            batch = df.iloc[i:i+batch_size]
            statements = batch["text"].tolist()
            true_labels = batch["true_label"].tolist()
            llm_classifications = batch["llm_classification"].tolist()
            confidences = batch["confidence"].tolist()
            llm_personal_funnies = batch["llm_personal_funny"].tolist()
            upvotes = batch["upvotes"].tolist()
            downvotes = batch["downvotes"].tolist()
            scores = batch["score"].tolist()

            if len(statements) < batch_size:
                statements += ["Placeholder"] * (batch_size - len(statements))
                true_labels += [False] * (batch_size - len(statements))
                llm_classifications += [None] * (batch_size - len(statements))
                confidences += [None] * (batch_size - len(statements))
                llm_personal_funnies += [None] * (batch_size - len(statements))
                upvotes += [None] * (batch_size - len(statements))
                downvotes += [None] * (batch_size - len(statements))
                scores += [None] * (batch_size - len(statements))

            prompt = (
                "For the text below, provide scores (1–10, where 1 is lowest and 10 is highest) Strictly provide only scores and nothing else. for:\n"
                "1. Funniness (how humorous you find it)\n"
                "2. Offensiveness (how likely to offend)\n"
                "3. Originality (how novel or creative)\n"
                "4. Appropriateness (how suitable for general audiences)\n"
                "5. Clarity (how clear and understandable the humor is)\n"
                "Format strictly as and nothing more than: Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V\n\n"
                f"Text: {statements[0]}"
            )

            # Payload
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a humor analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 100
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
                        f"Authentication failed (401): Invalid Grok API key. "
                        f"Verify at https://api.x.ai/, regenerate if needed, and check permissions."
                    )
                if response.status_code != 200 or "error" in response_json:
                    raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

    
                if "choices" in response_json:
                    llm_answer = response_json["choices"][0]["message"]["content"].strip()
                    #print(f"Grok Output: {llm_answer}")

        
                    parts = [p.strip() for p in llm_answer.replace("\n", "").split(",")]
                    if len(parts) >= 5 and all(":" in p for p in parts):
                        funniness = int(parts[0].split(":")[1].strip()) if parts[0].split(":")[1].strip().isdigit() else None
                        offensiveness = int(parts[1].split(":")[1].strip()) if parts[1].split(":")[1].strip().isdigit() else None
                        originality = int(parts[2].split(":")[1].strip()) if parts[2].split(":")[1].strip().isdigit() else None
                        appropriateness = int(parts[3].split(":")[1].strip()) if parts[3].split(":")[1].strip().isdigit() else None
                        clarity = int(parts[4].split(":")[1].strip()) if parts[4].split(":")[1].strip().isdigit() else None
                    else:
                        print(f"Invalid response format from Grok: {llm_answer}")
                        funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None
                else:
                    print("No choices in response.")
                    funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")
                funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

            # Store results
            for j in range(batch_size):
                results.append({
                    "joke_num": i + j + 1,
                    "text": statements[j],
                    "true_label": true_labels[j],
                    "llm_classification": llm_classifications[j],
                    "confidence": confidences[j],
                    "llm_personal_funny": llm_personal_funnies[j],
                    "funniness": funniness,
                    "offensiveness": offensiveness,
                    "originality": originality,
                    "appropriateness": appropriateness,
                    "clarity": clarity,
                    "upvotes": upvotes[j],
                    "downvotes": downvotes[j],
                    "score": scores[j]
                })

            
            time.sleep(1)

     
        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_folder, f"grok_{subcategory}_tier_2_quantitative.csv")
        results_df.to_csv(output_file, index=False)
        print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")
def process_colbert_with_grok_tier2(tier1_folder="llm_assessments", num_batches=10, batch_size = 3, api_key="placeholder"):
 
  
    endpoint = "https://api.x.ai/v1/chat/completions"
    model = "grok-3-latest"


   
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

  
    assessment_file = os.path.join(tier1_folder, "grok_colbert_assessments.csv")
    try:
        df = pd.read_csv(assessment_file)
        subcategory = "colbert"
        print(f"Loaded {len(df)} entries for {subcategory}")
    except Exception as e:
        print(f"Error loading ColBERT assessment file {assessment_file}: {e}")
        return

    
    output_folder = "tier_2_outputs"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"grok_{subcategory}_tier_2_quantitative.csv")

   
    results = []
    print(f"Using Grok API key (redacted): {api_key[:4]}...{api_key[-4:]}")
    print(f"Processing {len(df)} entries for {subcategory} with Grok for Tier 2 quantitative...")

    for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
        batch = df.iloc[i:i+batch_size]
        statements = batch["text"].tolist()
        true_labels = batch["true_label"].tolist()
        llm_classifications = batch["llm_classification"].tolist()
        confidences = batch["confidence"].tolist()
        llm_personal_funnies = batch["llm_personal_funny"].tolist()
        upvotes = batch["upvotes"].tolist()
        downvotes = batch["downvotes"].tolist()
        scores = batch["score"].tolist()

        if len(statements) < batch_size:
            statements += ["Placeholder"] * (batch_size - len(statements))
            true_labels += [False] * (batch_size - len(statements))
            llm_classifications += [None] * (batch_size - len(statements))
            confidences += [None] * (batch_size - len(statements))
            llm_personal_funnies += [None] * (batch_size - len(statements))
            upvotes += [None] * (batch_size - len(statements))
            downvotes += [None] * (batch_size - len(statements))
            scores += [None] * (batch_size - len(statements))

       
        prompt = (
            "For the text below, provide scores (1–10, where 1 is lowest and 10 is highest) for:\n"
            "1. Funniness (how humorous you find it)\n"
            "2. Offensiveness (how likely to offend)\n"
            "3. Originality (how novel or creative)\n"
            "4. Appropriateness (how suitable for general audiences)\n"
            "5. Clarity (how clear and understandable the humor is)\n"
            "Format strictly as: Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V\n\n"
            f"Text: {statements[0]}"
        )

        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a humor analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 100
        }

     
        try:
            print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
            response = requests.post(endpoint, headers=headers, json=payload)
            response_json = response.json()
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response_json, indent=2)}")

     
            if response.status_code == 401:
                raise Exception(f"Authentication failed (401): Invalid Grok API key. ")
            if response.status_code != 200 or "error" in response_json:
                raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

         
            if "choices" in response_json:
                llm_answer = response_json["choices"][0]["message"]["content"].strip()
                print(f"Grok Output: {llm_answer}")

                
                parts = [p.strip() for p in llm_answer.replace("\n", "").split(",")]
                if len(parts) >= 5 and all(":" in p for p in parts):
                    funniness = int(parts[0].split(":")[1].strip()) if parts[0].split(":")[1].strip().isdigit() else None
                    offensiveness = int(parts[1].split(":")[1].strip()) if parts[1].split(":")[1].strip().isdigit() else None
                    originality = int(parts[2].split(":")[1].strip()) if parts[2].split(":")[1].strip().isdigit() else None
                    appropriateness = int(parts[3].split(":")[1].strip()) if parts[3].split(":")[1].strip().isdigit() else None
                    clarity = int(parts[4].split(":")[1].strip()) if parts[4].split(":")[1].strip().isdigit() else None
                else:
                    print(f"Invalid response format from Grok: {llm_answer}")
                    funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None
            else:
                print("No choices in response.")
                funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

        # Store results
        for j in range(batch_size):
            results.append({
                "joke_num": i + j + 1,
                "text": statements[j],
                "true_label": true_labels[j],
                "llm_classification": llm_classifications[j],
                "confidence": confidences[j],
                "llm_personal_funny": llm_personal_funnies[j],
                "funniness": funniness,
                "offensiveness": offensiveness,
                "originality": originality,
                "appropriateness": appropriateness,
                "clarity": clarity,
                "upvotes": upvotes[j],
                "downvotes": downvotes[j],
                "score": scores[j]
            })

        time.sleep(2)  
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")
   
#did grok only gallows
#process_all_reddit_with_grok_tier2()
#process_colbert_with_grok_tier2()

#ChatGPT
def process_all_reddit_with_chatgpt_tier2(tier1_folder="llm_assessments", num_batches=6727, api_key="placeholder"):
    endpoint = "https://api.openai.com/v1/chat/completions"
    model = "gpt-3.5-turbo"
    batch_size = 3
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    output_folder = "tier_2_outputs"
    os.makedirs(output_folder, exist_ok=True)

    assessment_files = [f for f in os.listdir(tier1_folder) if f.startswith('chatgpt_') and f.endswith('_assessments.csv') and 'colbert' not in f]
    if not assessment_files:
        print(f"No ChatGPT assessment files found in {tier1_folder}.")
        return

    #print(f"Found {len(assessment_files)} Reddit assessment files: {assessment_files}")
    #print(f"Using ChatGPT API key (redacted): {api_key[:4]}...{api_key[-4:]}")

    
    for assessment_file in assessment_files:
        assessment_path = os.path.join(tier1_folder, assessment_file)
        print(f"\nProcessing assessment file: {assessment_file}")

      
        try:
            df = pd.read_csv(assessment_path)
            subcategory = assessment_file.replace('chatgpt_', '').replace('_assessments.csv', '')
            #print(f"Loaded {len(df)} entries for {subcategory}")
        except Exception as e:
            #print(f"Error loading assessment file {assessment_path}: {e}")
            continue

       
        results = []
        #print(f"Processing {len(df)} entries for {subcategory} with ChatGPT for Tier 2 quantitative...")

        for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
            batch = df.iloc[i:i+batch_size]
            statements = batch["text"].tolist()
            true_labels = batch["true_label"].tolist()
            llm_classifications = batch["llm_classification"].tolist()
            confidences = batch["confidence"].tolist()
            llm_personal_funnies = batch["llm_personal_funny"].tolist()
            upvotes = batch["upvotes"].tolist()
            downvotes = batch["downvotes"].tolist()
            scores = batch["score"].tolist()

            if len(statements) < batch_size:
                statements += ["Placeholder"] * (batch_size - len(statements))
                true_labels += [False] * (batch_size - len(statements))
                llm_classifications += [None] * (batch_size - len(statements))
                confidences += [None] * (batch_size - len(statements))
                llm_personal_funnies += [None] * (batch_size - len(statements))
                upvotes += [None] * (batch_size - len(statements))
                downvotes += [None] * (batch_size - len(statements))
                scores += [None] * (batch_size - len(scores))

            
            prompt = (
                "For the text below, provide scores (1–10, where 1 is lowest and 10 is highest) for:\n"
                "1. Funniness (how humorous you find it)\n"
                "2. Offensiveness (how likely to offend)\n"
                "3. Originality (how novel or creative)\n"
                "4. Appropriateness (how suitable for general audiences)\n"
                "5. Clarity (how clear and understandable the humor is)\n"
                "Format strictly as: Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V\n\n"
                f"Text: {statements[0]}"
            )

          
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a humor analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 100
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
                        f"Verify at OpenAI's developer portal (https://platform.openai.com/), regenerate if needed, and check permissions."
                    )
                if response.status_code != 200 or "error" in response_json:
                    raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

               
                if "choices" in response_json:
                    llm_answer = response_json["choices"][0]["message"]["content"].strip()
                    #print(f"ChatGPT Output: {llm_answer}")

                    # Parse the output (Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V)
                    parts = [p.strip() for p in llm_answer.replace("\n", "").split(",")]
                    if len(parts) >= 5 and all(":" in p for p in parts):
                        funniness = int(parts[0].split(":")[1].strip()) if parts[0].split(":")[1].strip().isdigit() else None
                        offensiveness = int(parts[1].split(":")[1].strip()) if parts[1].split(":")[1].strip().isdigit() else None
                        originality = int(parts[2].split(":")[1].strip()) if parts[2].split(":")[1].strip().isdigit() else None
                        appropriateness = int(parts[3].split(":")[1].strip()) if parts[3].split(":")[1].strip().isdigit() else None
                        clarity = int(parts[4].split(":")[1].strip()) if parts[4].split(":")[1].strip().isdigit() else None
                    else:
                        print(f"Invalid response format from ChatGPT: {llm_answer}")
                        funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None
                else:
                    print("No choices in response.")
                    funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")
                funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

            # Store results
            for j in range(batch_size):
                results.append({
                    "joke_num": i + j + 1,
                    "text": statements[j],
                    "true_label": true_labels[j],
                    "llm_classification": llm_classifications[j],
                    "confidence": confidences[j],
                    "llm_personal_funny": llm_personal_funnies[j],
                    "funniness": funniness,
                    "offensiveness": offensiveness,
                    "originality": originality,
                    "appropriateness": appropriateness,
                    "clarity": clarity,
                    "upvotes": upvotes[j],
                    "downvotes": downvotes[j],
                    "score": scores[j]
                })

            time.sleep(1)  # Avoid rate limits

        # Save results for this category/subcategory
        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_folder, f"chatgpt_{subcategory}_tier_2_quantitative.csv")
        results_df.to_csv(output_file, index=False)
        #print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")
def process_colbert_with_chatgpt_tier2(tier1_folder="llm_assessments", num_batches=25000, api_key="placeholder"):
    
    endpoint = "https://api.openai.com/v1/chat/completions"
    model = "gpt-3.5-turbo"
    batch_size = 1

   

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

  
    assessment_file = os.path.join(tier1_folder, "chatgpt_colbert_assessments.csv")
    try:
        df = pd.read_csv(assessment_file)
        subcategory = "colbert"
        print(f"Loaded {len(df)} entries for {subcategory}")
    except Exception as e:
        print(f"Error loading ColBERT assessment file {assessment_file}: {e}")
        return

   
    output_folder = "tier_2_outputs"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"chatgpt_{subcategory}_tier_2_quantitative.csv")

   
    results = []
    #print(f"Using ChatGPT API key (redacted): {api_key[:4]}...{api_key[-4:]}")
    #print(f"Processing {len(df)} entries for {subcategory} with ChatGPT for Tier 2 quantitative...")

    for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
        batch = df.iloc[i:i+batch_size]
        statements = batch["text"].tolist()
        true_labels = batch["true_label"].tolist()
        llm_classifications = batch["llm_classification"].tolist()
        confidences = batch["confidence"].tolist()
        llm_personal_funnies = batch["llm_personal_funny"].tolist()
        upvotes = batch["upvotes"].tolist()
        downvotes = batch["downvotes"].tolist()
        scores = batch["score"].tolist()

        if len(statements) < batch_size:
            statements += ["Placeholder"] * (batch_size - len(statements))
            true_labels += [False] * (batch_size - len(statements))
            llm_classifications += [None] * (batch_size - len(statements))
            confidences += [None] * (batch_size - len(statements))
            llm_personal_funnies += [None] * (batch_size - len(statements))
            upvotes += [None] * (batch_size - len(statements))
            downvotes += [None] * (batch_size - len(statements))
            scores += [None] * (batch_size - len(scores))

        
        prompt = (
            "For the text below, provide scores (1–10, where 1 is lowest and 10 is highest) for:\n"
            "1. Funniness (how humorous you find it)\n"
            "2. Offensiveness (how likely to offend)\n"
            "3. Originality (how novel or creative)\n"
            "4. Appropriateness (how suitable for general audiences)\n"
            "5. Clarity (how clear and understandable the humor is)\n"
            "Format strictly as: Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V\n\n"
            f"Text: {statements[0]}"
        )

       
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a humor analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 100
        }

        
        try:
            #print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
            response = requests.post(endpoint, headers=headers, json=payload)
            response_json = response.json()
            #print(f"Status Code: {response.status_code}")
            #print(f"Response: {json.dumps(response_json, indent=2)}")

           
            if response.status_code == 401:
                raise Exception(
                    f"Authentication failed (401): Invalid ChatGPT API key. "
                    f"Verify at OpenAI's developer portal (https://platform.openai.com/), regenerate if needed, and check permissions."
                )
            if response.status_code != 200 or "error" in response_json:
                raise Exception(f"API error: Status {response.status_code}, Response: {json.dumps(response_json)}")

            if "choices" in response_json:
                llm_answer = response_json["choices"][0]["message"]["content"].strip()
                print(f"ChatGPT Output: {llm_answer}")

                
                parts = [p.strip() for p in llm_answer.replace("\n", "").split(",")]
                if len(parts) >= 5 and all(":" in p for p in parts):
                    funniness = int(parts[0].split(":")[1].strip()) if parts[0].split(":")[1].strip().isdigit() else None
                    offensiveness = int(parts[1].split(":")[1].strip()) if parts[1].split(":")[1].strip().isdigit() else None
                    originality = int(parts[2].split(":")[1].strip()) if parts[2].split(":")[1].strip().isdigit() else None
                    appropriateness = int(parts[3].split(":")[1].strip()) if parts[3].split(":")[1].strip().isdigit() else None
                    clarity = int(parts[4].split(":")[1].strip()) if parts[4].split(":")[1].strip().isdigit() else None
                else:
                    print(f"Invalid response format from ChatGPT: {llm_answer}")
                    funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None
            else:
                print("No choices in response.")
                funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

        # Store results
        for j in range(batch_size):
            results.append({
                "joke_num": i + j + 1,
                "text": statements[j],
                "true_label": true_labels[j],
                "llm_classification": llm_classifications[j],
                "confidence": confidences[j],
                "llm_personal_funny": llm_personal_funnies[j],
                "funniness": funniness,
                "offensiveness": offensiveness,
                "originality": originality,
                "appropriateness": appropriateness,
                "clarity": clarity,
                "upvotes": upvotes[j],
                "downvotes": downvotes[j],
                "score": scores[j]
            })

        #time.sleep(2)  #can push more

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")


#process_all_reddit_with_chatgpt_tier2()
#process_colbert_with_chatgpt_tier2()
#Llama2

def process_all_reddit_with_llama_tier2(tier1_folder="llm_assessments", num_batches=6727,  batch_size = 3, api_key="placeholder"):
   
    client = Together(api_key=api_key)
    model = "meta-llama/LLaMA-3.3-70B-Instruct-Turbo-Free"
    batch_size = 1

    

    output_folder = "tier_2_outputs"
    os.makedirs(output_folder, exist_ok=True)

  
    assessment_files = [f for f in os.listdir(tier1_folder) if f.startswith('llama_') and f.endswith('_assessments.csv') and 'colbert' not in f]
    if not assessment_files:
        print(f"No LLaMA assessment files found in {tier1_folder}.")
        return

    #print(f"Found {len(assessment_files)} Reddit assessment files: {assessment_files}")
    #print(f"Using LLaMA API key (redacted): {api_key[:4]}...{api_key[-4:]}")


    for assessment_file in assessment_files:
        assessment_path = os.path.join(tier1_folder, assessment_file)
        print(f"\nProcessing assessment file: {assessment_file}")

    
        try:
            df = pd.read_csv(assessment_path)
            subcategory = assessment_file.replace('llama_', '').replace('_assessments.csv', '')
            print(f"Loaded {len(df)} entries for {subcategory}")
        except Exception as e:
            print(f"Error loading assessment file {assessment_path}: {e}")
            continue

      
        results = []
        #print(f"Processing {len(df)} entries for {subcategory} with LLaMA for Tier 2 quantitative...")

        for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
            batch = df.iloc[i:i+batch_size]
            statements = batch["text"].tolist()
            true_labels = batch["true_label"].tolist()
            llm_classifications = batch["llm_classification"].tolist()
            confidences = batch["confidence"].tolist()
            llm_personal_funnies = batch["llm_personal_funny"].tolist()
            upvotes = batch["upvotes"].tolist()
            downvotes = batch["downvotes"].tolist()
            scores = batch["score"].tolist()

            if len(statements) < batch_size:
                statements += ["Placeholder"] * (batch_size - len(statements))
                true_labels += [False] * (batch_size - len(statements))
                llm_classifications += [None] * (batch_size - len(statements))
                confidences += [None] * (batch_size - len(statements))
                llm_personal_funnies += [None] * (batch_size - len(statements))
                upvotes += [None] * (batch_size - len(statements))
                downvotes += [None] * (batch_size - len(statements))
                scores += [None] * (batch_size - len(statements))

        
            prompt = (
                "For the text below, provide scores (1–10, where 1 is lowest and 10 is highest) for:\n"
                "1. Funniness (how humorous you find it)\n"
                "2. Offensiveness (how likely to offend)\n"
                "3. Originality (how novel or creative)\n"
                "4. Appropriateness (how suitable for general audiences)\n"
                "5. Clarity (how clear and understandable the humor is)\n"
                "Format strictly as: Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V\n\n"
                f"Text: {statements[0]}"
            )

            
            try:
                #print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a humor analysis assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False,
                    temperature=0.3,
                    max_tokens=100
                )

           
                if hasattr(response, 'choices') and response.choices:
                    llm_answer = response.choices[0].message.content.strip()
                    #print(f"LLaMA Output: {llm_answer}")

                   
                    parts = [p.strip() for p in llm_answer.replace("\n", "").split(",")]
                    if len(parts) >= 5 and all(":" in p for p in parts):
                        funniness = int(parts[0].split(":")[1].strip()) if parts[0].split(":")[1].strip().isdigit() else None
                        offensiveness = int(parts[1].split(":")[1].strip()) if parts[1].split(":")[1].strip().isdigit() else None
                        originality = int(parts[2].split(":")[1].strip()) if parts[2].split(":")[1].strip().isdigit() else None
                        appropriateness = int(parts[3].split(":")[1].strip()) if parts[3].split(":")[1].strip().isdigit() else None
                        clarity = int(parts[4].split(":")[1].strip()) if parts[4].split(":")[1].strip().isdigit() else None
                    else:
                        print(f"Invalid response format from LLaMA: {llm_answer}")
                        funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None
                else:
                    print("No valid choices in response.")
                    funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")
                if "401" in str(e):
                    print(
                        f"Authentication failed (401): Invalid LLaMA API key. "
                        f"Verify at Together AI's developer portal (https://api.together.ai/), regenerate if needed, and check permissions."
                    )
                funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

        
            for j in range(batch_size):
                results.append({
                    "joke_num": i + j + 1,
                    "text": statements[j],
                    "true_label": true_labels[j],
                    "llm_classification": llm_classifications[j],
                    "confidence": confidences[j],
                    "llm_personal_funny": llm_personal_funnies[j],
                    "funniness": funniness,
                    "offensiveness": offensiveness,
                    "originality": originality,
                    "appropriateness": appropriateness,
                    "clarity": clarity,
                    "upvotes": upvotes[j],
                    "downvotes": downvotes[j],
                    "score": scores[j]
                })

            time.sleep(1)  

      
        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_folder, f"llama_{subcategory}_tier_2_quantitative.csv")
        results_df.to_csv(output_file, index=False)
        #print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")
def process_colbert_with_llama_tier2(tier1_folder="llm_assessments", num_batches=20000,batch_size = 3, api_key="placeholder"):
    
    client = Together(api_key=api_key)
    model = "meta-llama/LLaMA-3.3-70B-Instruct-Turbo-Free"
    

    
    assessment_file = os.path.join(tier1_folder, "llama_colbert_assessments.csv")
    try:
        df = pd.read_csv(assessment_file)
        subcategory = "colbert"
        #print(f"Loaded {len(df)} entries for {subcategory}")
    except Exception as e:
        print(f"Error loading ColBERT assessment file {assessment_file}: {e}")
        return

   
    output_folder = "tier_2_outputs"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"llama_{subcategory}_tier_2_quantitative.csv")


    results = []
    #print(f"Using LLaMA API key (redacted): {api_key[:4]}...{api_key[-4:]}")
    #print(f"Processing {len(df)} entries for {subcategory} with LLaMA for Tier 2 quantitative...")

    for i in range(0, min(num_batches * batch_size, len(df)), batch_size):
        batch = df.iloc[i:i+batch_size]
        statements = batch["text"].tolist()
        true_labels = batch["true_label"].tolist()
        llm_classifications = batch["llm_classification"].tolist()
        confidences = batch["confidence"].tolist()
        llm_personal_funnies = batch["llm_personal_funny"].tolist()
        upvotes = batch["upvotes"].tolist()
        downvotes = batch["downvotes"].tolist()
        scores = batch["score"].tolist()

        if len(statements) < batch_size:
            statements += ["Placeholder"] * (batch_size - len(statements))
            true_labels += [False] * (batch_size - len(statements))
            llm_classifications += [None] * (batch_size - len(statements))
            confidences += [None] * (batch_size - len(statements))
            llm_personal_funnies += [None] * (batch_size - len(statements))
            upvotes += [None] * (batch_size - len(statements))
            downvotes += [None] * (batch_size - len(statements))
            scores += [None] * (batch_size - len(statements))

        
        prompt = (
            "For the text below, provide scores (1–10, where 1 is lowest and 10 is highest) for:\n"
            "1. Funniness (how humorous you find it)\n"
            "2. Offensiveness (how likely to offend)\n"
            "3. Originality (how novel or creative)\n"
            "4. Appropriateness (how suitable for general audiences)\n"
            "5. Clarity (how clear and understandable the humor is)\n"
            "Format strictly as: Funniness: X, Offensiveness: Y, Originality: Z, Appropriateness: W, Clarity: V\n\n"
            f"Text: {statements[0]}"
        )

    
        try:
            #print(f"Processing batch {i//batch_size + 1}/{min(num_batches, len(df))} (text: {statements[0][:50]}...)")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a humor analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.3,
                max_tokens=100
            )

         
            if hasattr(response, 'choices') and response.choices:
                llm_answer = response.choices[0].message.content.strip()
                #print(f"LLaMA Output: {llm_answer}")

                
                parts = [p.strip() for p in llm_answer.replace("\n", "").split(",")]
                if len(parts) >= 5 and all(":" in p for p in parts):
                    funniness = int(parts[0].split(":")[1].strip()) if parts[0].split(":")[1].strip().isdigit() else None
                    offensiveness = int(parts[1].split(":")[1].strip()) if parts[1].split(":")[1].strip().isdigit() else None
                    originality = int(parts[2].split(":")[1].strip()) if parts[2].split(":")[1].strip().isdigit() else None
                    appropriateness = int(parts[3].split(":")[1].strip()) if parts[3].split(":")[1].strip().isdigit() else None
                    clarity = int(parts[4].split(":")[1].strip()) if parts[4].split(":")[1].strip().isdigit() else None
                else:
                    print(f"Invalid response format from LLaMA: {llm_answer}")
                    funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None
            else:
                print("No valid choices in response.")
                funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            if "401" in str(e):
                print(
                    f"Authentication failed (401): Invalid LLaMA API key. "
                    f"Verify at Together AI's developer portal (https://api.together.ai/), regenerate if needed, and check permissions."
                )
            funniness, offensiveness, originality, appropriateness, clarity = None, None, None, None, None

        # Store results
        for j in range(batch_size):
            results.append({
                "joke_num": i + j + 1,
                "text": statements[j],
                "true_label": true_labels[j],
                "llm_classification": llm_classifications[j],
                "confidence": confidences[j],
                "llm_personal_funny": llm_personal_funnies[j],
                "funniness": funniness,
                "offensiveness": offensiveness,
                "originality": originality,
                "appropriateness": appropriateness,
                "clarity": clarity,
                "upvotes": upvotes[j],
                "downvotes": downvotes[j],
                "score": scores[j]
            })

        time.sleep(1)  # Avoid rate limits


    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    #print(f"Saved {len(results_df)} entries for {subcategory} to {output_file}")

#process_all_reddit_with_llama_tier2()
process_colbert_with_llama_tier2()


#qualitative samples:
#TODO: fix qualitative samples sectionq
TIER_2_FOLDER = "tier_2_outputs"
OUTPUT_FOLDER = "tier_2_qualitative_samples"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
    "colbert": (None, None)  # ColBERT dataset has no category/subcategory
}

def collect_qualitative_explanation_samples(samples_per_type=5, target_total=40):
   
    llms = ["deepseek", "grok", "chatgpt", "llama", "olmo"]
    all_samples = {}

    for llm in llms:
       
        tier2_files = [f for f in os.listdir(TIER_2_FOLDER) if f.startswith(llm) and f.endswith("_tier_2_quantitative.csv")]
        if not tier2_files:
            print(f"No Tier 2 quantitative files found for {llm} in {TIER_2_FOLDER}.")
            continue

        
        tier2_dfs = []
        for file in tier2_files:
            try:
                df = pd.read_csv(os.path.join(TIER_2_FOLDER, file))
        
                subcategory = file.replace(f"{llm}_", "").replace("_tier_2_quantitative.csv", "")
                df["category"] = HUMOR_CATEGORIES.get(subcategory, (None, None))[0]
                df["subcategory"] = HUMOR_CATEGORIES.get(subcategory, (None, None))[1]
                tier2_dfs.append(df)
            except Exception as e:
                print(f"Error loading Tier 2 file {file}: {e}")
        if not tier2_dfs:
            print(f"No valid Tier 2 data loaded for {llm}.")
            continue
        combined_df = pd.concat(tier2_dfs, ignore_index=True)

      
        tp = combined_df[(combined_df["true_label"] == True) & (combined_df["llm_classification"] == "YES")]  # True Positive
        tn = combined_df[(combined_df["true_label"] == False) & (combined_df["llm_classification"] == "NO")]  # True Negative
        fp = combined_df[(combined_df["true_label"] == False) & (combined_df["llm_classification"] == "YES")]  # False Positive
        fn = combined_df[(combined_df["true_label"] == True) & (combined_df["llm_classification"] == "NO")]   # False Negative
        high_conf = combined_df[combined_df["confidence"] >= 7]  # High Confidence
        low_conf = combined_df[combined_df["confidence"] <= 3]   # Low Confidence
        funny_not_personal = combined_df[(combined_df["llm_classification"] == "YES") & (combined_df["llm_personal_funny"] == "NO")]  # Funny but not personally funny
        not_funny_personal = combined_df[(combined_df["llm_classification"] == "NO") & (combined_df["llm_personal_funny"] == "YES")]  # Not funny but personally funny

        
        categories = {
            "TruePositive": tp,
            "TrueNegative": tn,
            "FalsePositive": fp,
            "FalseNegative": fn,
            "HighConfidence": high_conf,
            "LowConfidence": low_conf,
            "FunnyNotPersonal": funny_not_personal,
            "NotFunnyPersonal": not_funny_personal
        }
        samples = []

       
        samples_per_category = max(1, target_total // len(categories))  # Aim for equal distribution
        remaining_samples = target_total

       
        for cat_name, cat_df in categories.items():
            if not cat_df.empty:
                n_samples = min(samples_per_type, len(cat_df), remaining_samples)
                if n_samples > 0:
                    grouped = cat_df.groupby(['category', 'subcategory'])
                    sampled = grouped.apply(lambda x: x.sample(min(len(x), max(1, n_samples // len(grouped)), n_samples), random_state=42)).reset_index(drop=True)
                    sampled["qualitative_category"] = cat_name
                    samples.append(sampled)
                    remaining_samples -= len(sampled)

        if samples:
            #deduplicate samples
            combined_samples = pd.concat(samples).drop_duplicates(subset=["text"]).sample(frac=1, random_state=42).reset_index(drop=True)
            all_samples[llm] = combined_samples.head(target_total)

            print(f"Collected {len(all_samples[llm])} samples for {llm} "
                  f"(TP: {len(tp)}, TN: {len(tn)}, FP: {len(fp)}, FN: {len(fn)}, "
                  f"HighConf: {len(high_conf)}, LowConf: {len(low_conf)}, "
                  f"FunnyNotPersonal: {len(funny_not_personal)}, NotFunnyPersonal: {len(not_funny_personal)})")
            print(f"Category coverage: {all_samples[llm]['category'].nunique()} categories, {all_samples[llm]['subcategory'].nunique()} subcategories")

            output_file = os.path.join(OUTPUT_FOLDER, f"{llm}_tier_2_qualitative_samples.csv")
            all_samples[llm].to_csv(output_file, index=False)
            print(f"Saved {len(all_samples[llm])} samples for {llm} to {output_file}")
        else:
            print(f"No valid samples collected for {llm}.")

    return all_samples

# Run the function
#samples = collect_qualitative_explanation_samples(samples_per_type=5, target_total=40)

#feed Samples through LLMS --> did this manually.
#tbd-feed samples through LLMs for qualitative analysis.

























