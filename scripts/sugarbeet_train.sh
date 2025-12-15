python3 scripts/sugarbeet_train_ddp.py --model lightmanet --use_rgbnir

sleep 300

python3 scripts/sugarbeet_train_ddp.py --model unet --batch_size 16 --use_rgbnir