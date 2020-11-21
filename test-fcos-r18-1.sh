
cd ../AdelaiDet

python tools/train_net.py --eval-only --config-file configs/FCOS-Detection/R_18_1x-Full-SyncBN-FixFPN-Shared-lsq-M2F8L8.yaml \
  MODEL.QUANTIZATION.policy configs/FCOS-Detection/policy-fcos-r18-test_1.txt MODEL.RESNETS.NORM BN MODEL.FPN.NORM BN-ReLU MODEL.FCOS.NORM BN-shared \
  MODEL.WEIGHTS output/fcos/R_18_1x-Full_SyncBN-FixFPN-Shared-lsq-M2F8L8/model_final.pth

cd -

