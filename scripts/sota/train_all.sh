# python3 scripts/sota/train_weedsgalore.py --model deeplabsv3+ --use_rgbnir
# python3 scripts/sota/train_weedsgalore.py --model pspnet --use_rgbnir
# python3 scripts/sota/train_weedsgalore.py --model lightsegnet --use_rgbnir
# python3 scripts/sota/train_weedsgalore.py --model unet++ --use_rgbnir
# python3 scripts/sota/train_weedsgalore.py --model unet3+ --base_ch 16 --use_rgbnir
python3 scripts/sota/sugarbeet_train.py --model unet --use_rgbnir 
python3 scripts/sota/sugarbeet_train.py --model deeplabsv3+ --use_rgbnir --base_ch 16