# RecFTT
RecFTT: Flexible Targeted Tuning of Recommendation Models

We use the [ReChorus](https://github.com/THUwangcy/ReChorus) framework as our code base and implement the SAE module upon it.

### Workflow
The experimental workflow consists of four main stages:
Stage 1: Pre-train the Base Recommendation Model.
Stage 2: Pre-train the SAE Module. Train the SAE module to learn a sparse, interpretable latent space.
Stage 3: Latent Analysis and Selection. Systematically probes each latent feature to quantify its causal effect on different recommendation attributes. For each latent, we perform independent interventions by multiplying its activation value by different scale factors (e.g., 10.0 for amplification and 0.1 for suppression). By evaluating the resulting change in the proportion of target items in the recommendation list, we calculate a contribution score for each latent, measuring its relevance and impact on the target attribute.
Stage 4: Fine-tune for the Target Task. Load all pre-trained models, freeze the base model, and then fine-tune the SAE using the latents selected in Stage 3 to achieve the desired control objective.

### Command
```bash
cd src


# STAGE 1: Pre-train the Base Recommendation Model
# This step generates the base model weights that will be frozen in the next stages.
python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1

# STAGE 2: Train the RecSAE module
python main_sae.py  --epoch 50 --sae_lr 1e-4 --sae_batch_size 4 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1

# STAGE 3: Latent Analysis and Selection
./run_ablation_loop.sh
cd ablation_analysis_results
python analyze_latent_gro.py # For Grocery dataset. You may need to manually change the file paths inside the analysis script to match the model and dataset you are analyzing.
# Or: python analyze_latent_ml.py # For MovieLens dataset
python choose_latent.py sasrec_analyze/vegetarian_regulation_scores.json sasrec_analyze/controller_vegetarian.json -x 100 # Select top 100 positive and top 100 negative latents
cd ..

# STAGE 4: Fine-tune the RecSAE module
# This is the core of our method. We load the pre-trained SASRec and the pre-trained SAE, freeze SASRec, and fine-tune the SAE module
# Example: Fine-tuning to boost "gluten_free" recommendations
python main_sae.py --epoch 50 --sae_batch_size 4 --sae_lr 5e-05 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --path ../data/ --test_all 1 --sae_train 1 --gpu 1 --is_tuning 1 --tuning_dims vegetarian --eval_dims low_price,gluten_free,children_friendly,vegetarian,caffeine_free,no_sugar --loss_lambda 5.0 --model_path ../model/SASRec_SAE/SASRec_SAE__Grocery_and_Gourmet_Food__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr=5e-05__sae_batch_size=4__sae_k=8__sae_scale_size=16.pt --tuning_lr 1e-4 --main_metric NDCG@5

```

### Hyper-parameters
Key hyper-parameters for the RecFTT stage can be found and adjusted in main_sae.py and are passed via the command line:
- --sae_lr: SAE learning rate
- --sae_k: SAE k value
- --sae_scale_size: SAE scale size
- --tuning_lr: Fine-tuning learning rate
- --loss_lambda: The weight of the target tuning loss (L_tuning) in the composite loss function
- --tuning_dim: Specifies the target dimension for the tuning loss
