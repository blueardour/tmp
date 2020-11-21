
cd ../AdelaiDet/

python ../ldn-quantization/tools.py --keyword update,raw --old tmp/r50-gn-2bit/model_final.pth --new tmp/r50-gn-2bit/import_jing.pth --mf ../tmp/fcos-r50-gn_from.txt-sort --mt ../tmp/fcos-r50-gn_to.txt-sort

python ../ldn-quantization/tools.py --keyword reshape --old tmp/r50-gn-2bit/import_jing.pth --new tmp/r50-gn-2bit/reshape.pth --mf weights/det-resnet18/reshape.txt 

cd -
