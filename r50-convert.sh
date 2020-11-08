
cd ../AdelaiDet/

bit=2

python ../ldn-quantization/tools.py --keyword update,raw --old pretrained/fcos/R_50_1x-FPN_SyncBN-FCOS-SyncBN_no_lp_pretrained_${bit}bit/model_final.pth --new pretrained/fcos/R_50_1x-FPN_SyncBN-FCOS-SyncBN_no_lp_pretrained_${bit}bit/import_jing.pth --mf ../tmp/fcos-r50-from.txt-sort --mt ../tmp/fcos-r50-to.txt-sort

python ../ldn-quantization/tools.py --keyword reshape --old pretrained/fcos/R_50_1x-FPN_SyncBN-FCOS-SyncBN_no_lp_pretrained_${bit}bit/import_jing.pth --new pretrained/fcos/R_50_1x-FPN_SyncBN-FCOS-SyncBN_no_lp_pretrained_${bit}bit/reshape.pth --mf weights/det-resnet18/reshape.txt 

cd -
