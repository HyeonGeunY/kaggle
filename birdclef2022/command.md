python3 training/run_expriment.py --wandb --gpus="0," --batch_size=24  --num_workers=4

python training\run_experiment.py --wandb --gpus="0," --batch_size=64 --num_workers=4 --model_class="ResNetBird" --max_epochs=40 --embedding_size=1024 --data_class="BirdClef2022_v3" --oversampling


# evaluation
python birdclef\birdclef_model.py "C:\Users\ftmlab\Documents\hyoon\project_new\kaggle\birdclef2022\input\birdclef-2022\test_soundscapes"