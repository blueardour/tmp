
cd ../AdelaiDet

bit=2

#python tools/train_net.py --eval-only --config-file configs/FCOS-Detection/R_18_1x-Full-SyncBN-FixFPN-FixPoint-lsq-M${bit}F8L8.yaml \
#  MODEL.QUANTIZATION.policy configs/FCOS-Detection/policy-fcos-r18-test.txt MODEL.RESNETS.NORM BN MODEL.FPN.NORM BN-ReLU MODEL.FCOS.NORM BN \
#  MODEL.WEIGHTS FCOS-R18-M${bit}F8L8/model_final.pth

python tools/train_net.py --eval-only --config-file configs/FCOS-Detection/R_18_1x-FPN-SyncBN-FixFPN-lsq-M2F8L8.yaml \
  MODEL.QUANTIZATION.policy configs/FCOS-Detection/policy-fcos-r18-test.txt MODEL.RESNETS.NORM BN MODEL.FPN.NORM BN-ReLU \
  MODEL.WEIGHTS
  #MODEL.WEIGHTS FCOS-R18-M${bit}F8L8/model_final.pth

cd -

