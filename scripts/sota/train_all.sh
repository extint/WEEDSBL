# python3 scripts/sota/train_weedsgalore.py --model deeplabsv3+ --use_rgbnir
# python3 scripts/sota/train_weedsgalore.py --model pspnet --use_rgbnir
# python3 scripts/sota/train_weedsgalore.py --model lightsegnet --use_rgbnir
# python3 scripts/sota/train_weedsgalore.py --model unet++ --use_rgbnir
# python3 scripts/sota/train_weedsgalore.py --model unet3+ --base_ch 16 --use_rgbnir
# python3 scripts/sota/sugarbeet_train.py --model unet --use_rgbnir 
# python3 scripts/sota/sugarbeet_train.py --model deeplabsv3+ --use_rgbnir --base_ch 16

# mini_dual_enocder 4ch to unet 3ch - vedant
python3 -m sota.distill_train_dual_enc_mini_to_unet --teacher_ckpt /home/vjti-comp/WEEDSBL/scripts/dual_encoder/runs/dual_encoder_mini_20260412_130647/checkpoints/best_model.pth 
python3 finetune.py --model deeplabv3_mobilenet --use_rgbnir --no_pretrained --epochs 50