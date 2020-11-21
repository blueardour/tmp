
cd ../AdelaiDet

bit=4

python tools/train_net.py --eval-only --config-file configs/FCOS-Detection/R_34_1x-Full-SyncBN-FixFPN-FixPoint-lsq-M${bit}F8L8.yaml \
  MODEL.QUANTIZATION.policy configs/FCOS-Detection/policy-fcos-r34-test.txt MODEL.RESNETS.NORM BN MODEL.FPN.NORM BN-ReLU MODEL.FCOS.NORM BN \
  MODEL.WEIGHTS output/fcos/R_34_1x-Full_SyncBN-FixFPN-FixPoint-lsq-M${bit}F8L8-RD2/model_final.pth
  #MODEL.WEIGHTS FCOS-R18-M${bit}F8L8/model_final.pth

cd -

