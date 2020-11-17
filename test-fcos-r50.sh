
cd ../AdelaiDet

bit=2

#python tools/train_net.py --eval-only --config-file configs/FCOS-Detection/R_50_1x-Full-SyncBN-FixFPN-FixPoint-lsq-M${bit}F8L8.yaml \
#  MODEL.QUANTIZATION.policy configs/FCOS-Detection/policy-fcos-r50-test.txt MODEL.RESNETS.NORM BN MODEL.FPN.NORM BN-ReLU MODEL.FCOS.NORM BN \
#  MODEL.WEIGHTS output/fcos/R_50_1x-Full_SyncBN-FixFPN-FixPoint-lsq-M${bit}F8L8-RD2/model_final.pth
#  #MODEL.WEIGHTS FCOS-R50-M${bit}F8L8/model_final.pth

python tools/train_net.py --eval-only --config-file configs/FCOS-Detection/R_50_1x-FPN-SyncBN-FixFPN-lsq-M2F8L8.yaml \
  MODEL.QUANTIZATION.policy configs/FCOS-Detection/policy-fcos-r50-test.txt MODEL.RESNETS.NORM BN MODEL.FPN.NORM BN-ReLU \
  MODEL.WEIGHTS output/fcos/R_50_1x-FPN_SyncBN-FixFPN-lsq-M2F8L8/model_final.pth

cd -

