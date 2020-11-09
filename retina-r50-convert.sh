
cd ../detectron2

bit=4

python ../ldn-quantization/tools.py --keyword update,raw --old pretrained/retinanet/retinanet_R_50_1x_Full_SyncBN_dorefa_clip_${bit}bit_no_lp_pretrained/model_final.pth --new pretrained/retinanet/retinanet_R_50_1x_Full_SyncBN_dorefa_clip_${bit}bit_no_lp_pretrained/error-fpn.pth --mf ../tmp/retina-r50_from.txt-sort --mt ../tmp/retina-r50_to.txt-sort 

python ../ldn-quantization/tools.py --keyword reshape --mf weights/det-resnet18/reshape.txt --old pretrained/retinanet/retinanet_R_50_1x_Full_SyncBN_dorefa_clip_${bit}bit_no_lp_pretrained/error-fpn.pth --new pretrained/retinanet/retinanet_R_50_1x_Full_SyncBN_dorefa_clip_${bit}bit_no_lp_pretrained/error-reshape.pth

cd - 

