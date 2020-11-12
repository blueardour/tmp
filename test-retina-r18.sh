
cd ../detectron2

bit=2

python tools/train_net.py --eval-only --config-file configs/COCO-Detection/retinanet_R_18_FPN_1x-Full-SyncBN-FixFPN-FixPoint-lsq-M${bit}F8L8.yaml \
  MODEL.QUANTIZATION.policy configs/COCO-Detection/policy_retina-r18-test.txt MODEL.RESNETS.NORM BN MODEL.FPN.NORM BN-ReLU MODEL.RETINANET.NORM BN \
  MODEL.WEIGHTS FCOS-R18-M${bit}F8L8/model_final.pth

cd -

