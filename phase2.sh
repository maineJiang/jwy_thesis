data_dir=datasets/sub-imagenet-200  # igin imgs
bd_data_dir=datasets/sub-imagenet-200-bd/inject_a/      # imgs having trigger
resume_path=ckpt/bd/res18_bd_ratio_0.1_inject_a_centerloss/imagenet_checkpoint.pth.tar
model=res18
bd_ratio=0.1
train_batch=128
bd_label=0
bd_char=badnet #attack method


python train.py \
--net=$model \
--train_batch=$train_batch \
--workers=4 \
--epochs=25 \
--schedule 13 17 \
--bd_label=0 \
--bd_ratio=$bd_ratio \
--data_dir=$data_dir \
--bd_data_dir=$bd_data_dir \
--resume=$resume_path \
--freeze=true\
--checkpoint=ckpt/bd/${model}_bd_ratio_${bd_ratio}_inject_${bd_char}_phase2 \
--manualSeed=0