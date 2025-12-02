import json
import csv
import os

MODEL = "BPRMF"
DATASET = "MovieLens"
FEATURES = [
    "Drama",
    "Comedy",
    "Action",
    "Thriller",
    "Horror",  
    "Romance",
    "Sci-Fi",
    "Adventure"
]
OUTPUT_FILENAME = f"{MODEL}_{DATASET}.csv"

def process_files_and_generate_csv():
    feature_ranks = {}
    all_latent_ids = set()

    print("\n--- Reading JSON files and calculating ranks ---")
    for feature in FEATURES:
        if feature == "Sci-Fi":
            feature_key = "Sci_Fi"
        else:
            feature_key = feature
        filename = f"{MODEL}/{DATASET}/{feature_key}_regulation_scores.json"
        
        if not os.path.exists(filename):
            print(f"Warning: File not found for feature '{feature}': {filename}. Skipping.")
            continue
            
        print(f"Processing file: {filename}")
        
        feature_ranks[feature] = {}
        
        with open(filename, 'r') as f:
            data = json.load(f)
            
            for rank, item in enumerate(data, 1):
                latent_id = item["latent_id"]
                feature_ranks[feature][latent_id] = rank
                all_latent_ids.add(latent_id)
   
    if not all_latent_ids:
        print("\nError: No latent IDs found in any of the JSON files. Cannot generate CSV.")
        return

    print("\n--- Generating CSV file ---")
    sorted_latent_ids = sorted(list(all_latent_ids))

    header = ["latent_id"] + FEATURES
    
    with open(OUTPUT_FILENAME, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        for latent_id in sorted_latent_ids:
            row = [latent_id]
            for feature in FEATURES:
                rank = feature_ranks.get(feature, {}).get(latent_id, 'NaN')
                row.append(rank)

            writer.writerow(row)
            
    print(f"\nSuccessfully created '{OUTPUT_FILENAME}' with {len(sorted_latent_ids)} data rows.")

if __name__ == "__main__":
    process_files_and_generate_csv()