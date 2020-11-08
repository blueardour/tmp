
cd ../detectron2

bit=3

python ../ldn-quantization/tools.py --keyword update,raw --old pretrained/retinanet/retinanet_R_18_1x_Full_BN_dorefa_clip_${bit}bit_no_lp_pretrained/model_final.pth --new pretrained/retinanet/retinanet_R_18_1x_Full_BN_dorefa_clip_${bit}bit_no_lp_pretrained/error-fpn.pth --mf ../tmp/retina-r18-from_jing.txt-sort --mt ../tmp/retina-r18-to.txt-sort 

python ../ldn-quantization/tools.py --keyword reshape --mf weights/det-resnet18/reshape.txt --old pretrained/retinanet/retinanet_R_18_1x_Full_BN_dorefa_clip_${bit}bit_no_lp_pretrained/error-fpn.pth --new pretrained/retinanet/retinanet_R_18_1x_Full_BN_dorefa_clip_${bit}bit_no_lp_pretrained/error-reshape.pth

cd - 

