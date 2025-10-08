for (( i=0; i<=1023; i+=1 )); do
  for offset in 0; do
    latent=$(( i + offset ))
    # 防止最后一批超过 1023
    if (( latent > 1023 )); then
      continue
    fi

    # 并行启动
	python main_sae.py --epoch 50 --sae_batch_size 4 --sae_lr 5e-05 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --path ../data/ --test_all 1 --sae_train 0 --gpu 4 --ablation_latent $latent --ablation_scale 10.0 &
  python main_sae.py --epoch 50 --sae_batch_size 4 --sae_lr 5e-05 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --path ../data/ --test_all 1 --sae_train 0 --gpu 5 --ablation_latent $latent --ablation_scale 0.1 &
  done

  # 等待本批次的 3 个进程全部结束
  wait
done