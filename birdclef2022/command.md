python3 training/run_expriment.py --wandb --gpus="0," --batch_size=24  --num_workers=4

python training\run_expriment.py --wandb --gpus="0," --batch_size=64 --num_workers=4 --model_class="ResNetBird" --max_epochs=20 --embedding_size=2048