
cd ../detectron2

bit=4

python tools/train_net.py --eval-only --config-file configs/COCO-Detection/retinanet_R_50_FPN_1x-Full-SyncBN-FixFPN-FixPoint-lsq-M${bit}F8L8.yaml \
  MODEL.QUANTIZATION.policy configs/COCO-Detection/policy_retina-r50-test.txt MODEL.RESNETS.NORM BN MODEL.FPN.NORM BN-ReLU MODEL.RETINANET.NORM BN \
  MODEL.WEIGHTS output/coco-detection/retinanet_R_50_FPN_1x-Full_SyncBN-FixFPN-FixPoint-lsq-M4F8L8/model_0060499.pth

cd -

