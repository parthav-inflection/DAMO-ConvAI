eval_path='./dev_data/dev_20240627/dev.json'
dev_path='./output/'
db_root_path='./dev_data/dev_20240627/dev_databases/'
use_knowledge='True'
not_use_knowledge='False'
mode='dev' # choose dev or dev
cot='True'
no_cot='False'

YOUR_API_KEY=''

engine1='code-davinci-002'
engine2='text-davinci-003'
engine3='inf-2-0-32b-sql'
max_workers=5  # Number of parallel workers - adjust based on your API rate limits

# data_output_path='./exp_result/gpt_output/'
# data_kg_output_path='./exp_result/gpt_output_kg/'

data_output_path='./exp_result/turbo_output/'
data_kg_output_path='./exp_result/qwen3_output_kg/'


echo 'generate GPT3.5 batch with knowledge'
python3 -u ./src/gpt_request.py --db_root_path ${db_root_path} --api_key ${YOUR_API_KEY} --mode ${mode} \
--engine ${engine3} --eval_path ${eval_path} --data_output_path ${data_kg_output_path} --use_knowledge ${use_knowledge} \
--chain_of_thought ${cot} --max_workers ${max_workers}

# echo 'generate GPT3.5 batch without knowledge'
# python3 -u ./src/gpt_request.py --db_root_path ${db_root_path} --api_key ${YOUR_API_KEY} --mode ${mode} \
# --engine ${engine3} --eval_path ${eval_path} --data_output_path ${data_output_path} --use_knowledge ${not_use_knowledge} \
# --chain_of_thought ${no_cot} --max_workers ${max_workers}
