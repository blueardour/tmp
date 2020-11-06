
bit=2

python ../ldn-quantization/tools.py --keyword update,raw --old pretrained/fcos/R_18_1x-Full_SyncBN_dorefa_clip_${bit}bit_no_lp_pretrained/model_final.pth --new pretrained/fcos/R_18_1x-Full_SyncBN_dorefa_clip_${bit}bit_no_lp_pretrained/import_jing.pth --mf weights/det-resnet18/lsq-2bit-from_jing.txt-sort --mt weights/det-resnet18/lsq-2bit-to_FixFPN.txt-sort 

python ../ldn-quantization/tools.py --keyword reshape --old pretrained/fcos/R_18_1x-Full_SyncBN_dorefa_clip_${bit}bit_no_lp_pretrained/import_jing.pth --mf weights/det-resnet18/reshape.txt --new pretrained/fcos/R_18_1x-Full_SyncBN_dorefa_clip_${bit}bit_no_lp_pretrained/reshape.pth

