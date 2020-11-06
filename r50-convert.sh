
cd ../AdelaiDet/

bit=2

python ../ldn-quantization/tools.py --keyword update,raw --old pretrained/fcos/R_50_1x-FPN_SyncBN-FCOS-SyncBN_no_lp_pretrained_${bit}bit/model_final.pth --new pretrained/fcos/R_50_1x-FPN_SyncBN-FCOS-SyncBN_no_lp_pretrained_${bit}bit/import_jing.pth --mf ../tmp/lsq-2bit_to_reshape.txt-sort --mt ../tmp/lsq-2bit_from_jing.txt-sort 

python ../ldn-quantization/tools.py --keyword reshape --old pretrained/fcos/R_50_1x-FPN_SyncBN-FCOS-SyncBN_no_lp_pretrained_${bit}bit/import_jing.pth --new pretrained/fcos/R_50_1x-FPN_SyncBN-FCOS-SyncBN_no_lp_pretrained_${bit}bit/reshape.pth --mf weights/det-resnet18/reshape.txt 

