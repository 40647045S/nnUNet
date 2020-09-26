1.安裝套件
  
git clone https://github.com/40647045S/nnUNet.git  
cd nnUNet  
pip install -e .  
  
2.準備資料  
  
python file_processor.py 訓練資料夾 測試資料夾  
  
export nnUNet_raw_data_base=${PWD}/nnUNet/nnUNet_raw  
export nnUNet_preprocessed=${PWD}/nnUNet/nnUNet_preprocessed  
export RESULTS_FOLDER=${PWD}/nnUNet/nnUNet_trained_models  
  
nnUNet_plan_and_preprocess -t 101 --verify_dataset_integrity  
  
3.進行訓練  
  
CUDA_VISIBLE_DEVICES=X nnUNet_train 3d_lowres my_trainer Task101_BrainTS 0  
CUDA_VISIBLE_DEVICES=X nnUNet_train 3d_lowres my_trainer Task101_BrainTS 1  
CUDA_VISIBLE_DEVICES=X nnUNet_train 3d_lowres my_trainer Task101_BrainTS 2  
CUDA_VISIBLE_DEVICES=X nnUNet_train 3d_lowres my_trainer Task101_BrainTS 3  
（CUDA_VISIBLE_DEVICES=X中的X是想使用的gpu編號，可以開多個terminal然後用不同的GPU跑就可以平行加速了）  
  
4.進行預測  
  
nnUNet_predict -i ./test_images  -o ./result -t Task101_BrainTS -m 3d_lowres -tr my_trainer -f 0 1 2 3 --save_npz -chk model_final_checkpoint  
  
cd result  
zip result.zip *.nii.gz  
就會出現最終結果result.zip  
