# 1. 测试 Crane+ on MVTec AD
#python test.py \
#  --dataset mvtec \
#  --model_name trained_on_visa_cranep \
#  --devices 2 \
#  --epoch 5 \
#  --dino_model dinov2 \
#  --features_list 24 \
#  --visualize False

# 2. 测试 Crane+ on VisA
python test.py \
  --dataset visa \
  --model_name trained_on_visa_cranep \
  --devices 3 \
  --epoch 5 \
  --dino_model dinov2 \
  --features_list 24 \
  --visualize False