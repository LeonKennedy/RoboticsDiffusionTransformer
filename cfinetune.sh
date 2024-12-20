#export CFLAGS="-I/usr/include"
#export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
#export CUTLASS_PATH="/path/to/cutlass"

export WANDB_PROJECT="robotics_diffusion_transformer"


python main.py \
    --max_train_steps=200000 \
#    --checkpointing_period=1000 \
#    --sample_period=500 \
    --dataloader_num_workers=4 \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=wandb
    --precomp_lang_embed

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
