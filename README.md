
Defectnet: Multi-Class Fault Detection on Highly-Imbalanced Datasets
Paper: https://arxiv.org/abs/1904.00863

# For training the model:
LOGS_DIR="model/"  
PATCH_DIR="Patches400/Train/"  
python DefectNet.py --batch_size=24 --patch_size=400 --max_itr=100000 --logs_dir="${LOGS_DIR}" --patch_dir="${PATCH_DIR}"
# Using multiple GPUs on AWS
python DefectNet_multi_gpu.py --batch_size=15 --patch_size=400 --max_itr=10000 --logs_dir="${LOGS_DIR}" --patch_dir="${PATCH_DIR}" --num_gpus=4

# For prediction process:
LOGS_DIR="model/"  
DATA_DIR="blade_images/"  
RESULT_DIR="results/"  
python data_full_size.py --data_dir="${DATA_DIR}"  
python DefectNet.py --mode=evaluate --logs_dir="${LOGS_DIR}" --result_dir="${RESULT_DIR}"  
python read.py --data_dir="${DATA_DIR}" --result_dir="${RESULT_DIR}"    
# or if the model was trained by multiple gpus
python data_full_size.py --data_dir="${DATA_DIR}"  
python DefectNet_multi_gpu.py --mode=evaluate --logs_dir="${LOGS_DIR}" --result_dir="${RESULT_DIR}"   
python read.py --data_dir="${DATA_DIR}" --result_dir="${RESULT_DIR}"
