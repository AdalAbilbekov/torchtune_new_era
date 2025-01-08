from datasets import load_dataset, concatenate_datasets, Features, Value
from huggingface_hub import HfApi, login, Repository
import json
import os
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__=="__main__":
    login(token = "_", add_to_git_credential = False) # dont change

    # define general repository details
    repo_id = "_" # dont change

    # initialise HfApi
    api = HfApi()

    all_data_files_1 = ['2024.09.17_instruct_kk_nlp-bundle_179196016.json',
                    '2024.09.17_instruct_kk_nlp-bundle_409586136.json',
                    '2024.09.17_instruct_kk_nlp-bundle_3383561.json',
                    '2024.09.17_instruct_kk_synthetic_1455707.json',
                    '2024.09.13_instruct_kk_ideology_291510.json',
                    '2024.09.12_instruct_kk_synthetic_17783761.json',
                    '2024.09.09_instruct_en_benchmarks-mmlu_29346593.json',
                    '2024.09.09_instruct_kk_benchmarks-mmlu_17725926.json',
                    '2024.09.09_instruct_ru_benchmarks-mmlu_2230688.json',
                    '2024.09.09_instruct_tr_benchmarks-mmlu_1865332.json',
                    '2024.09.09_instruct_en_benchmarks-hellaswag_6412871.json',
                    '2024.09.09_instruct_ot_theses_3971100.json',
                    '2024.09.06_instruct_tr_benchmarks-arc_123082.json',
                    '2024.09.06_instruct_ru_benchmarks-arc_128104.json',
                    '2024.09.06_instruct_kk_benchmarks-arc_119687.json',
                    '2024.09.06_instruct_en_benchmarks-arc_154489.json',
                    '2024.09.06_instruct_en_nu-faq_9346.json',
                    '2024.09.06_instruct_kk_nu-faq_7002.json',
                    '2024.09.06_instruct_ru_nu-faq_7620.json',
                    '2024.09.06_instruct_tr_benchmarks-gsm8k_632933.json',
                    '2024.09.06_instruct_ru_benchmarks-gsm8k_686872.json',
                    '2024.09.06_instruct_kk_benchmarks-gsm8k_628249.json',
                    '2024.09.06_instruct_en_benchmarks-gsm8k_770413.json',
                    '2024.09.06_instruct_en_benchmarks-winogrande_1325808.json',
                    '2024.09.06_instruct_kk_benchmarks-winogrande_1012779.json',
                    '2024.09.06_instruct_ru_benchmarks-winogrande_320886.json',
                    '2024.09.06_instruct_tr_benchmarks-winogrande_277850.json',
                    '2024.08.26_instruct_ot_guardrailing4lang0_62083998.json',
                    '2024.08.26_instruct_ot_langident167culturax0_45402708.json',
                    '2024.08.26_instruct_pr_sqlcreatecontext0_2729725.json',
                    '2024.08.28_instruct_pr_llama2sqlandcodedataset0_70003.json',
                    '2024.08.28_instruct_pr_llama2sqlandcodedataset1_4743568.json',
                    '2024.08.28_instruct_pr_pythoncodeinstructions18kalpaca0_3240404.json',
                    '2024.08.29_instruct_pr_synthetictexttosql0_485957.json',
                    '2024.08.29_instruct_pr_synthetictexttosql1_14214200.json',
                    '2024.09.17_instruct_kk_nlp-bundle_1157262.json',
                    '2024.09.09_instruct_kk_benchmarks-hellaswag_5134445.json',
                    '2024.09.09_instruct_ru_benchmarks-hellaswag_711624.json',
                    '2024.09.09_instruct_tr_benchmarks-hellaswag_632135.json']
    all_data_files_2 = ['2024.09.23_instruct_en_QnA-MBZUAI-LaMini-instruction-train_126137683.json',
                    '2024.09.23_instruct_en_QnA-Open-Orca-OpenOrca-train_1059806689.json',
                    '2024.09.23_instruct_en_QnA-ignmilton-ign-clean-instruct-dataset-500k-train_165792862.json',
                    '2024.09.23_instruct_en_QnA-nvidia-HelpSteer-train_15891094.json',
                    '2024.09.23_instruct_en_QnA-nvidia-HelpSteer-validation_817544.json',
                    '2024.09.23_instruct_en_Summarization-billsum-ca-test_1602724.json',
                    '2024.09.23_instruct_en_Summarization-billsum-test_2743818.json',
                    '2024.09.23_instruct_en_Summarization-billsum-train_15931656.json',
                    '2024.09.23_instruct_en_Summarization-gigaword-test_62647.json',
                    '2024.09.23_instruct_en_Summarization-gigaword-train_125271398.json',
                    '2024.09.23_instruct_en_Summarization-gigaword-validation_5996636.json',
                    '2024.09.23_instruct_en_Summarization-knkarthick-dialogsum-test_228082.json',
                    '2024.09.23_instruct_en_Summarization-knkarthick-dialogsum-train_1917754.json',
                    '2024.09.23_instruct_en_Summarization-knkarthick-dialogsum-validation_74760.json',
                    '2024.09.23_instruct_kk_QnA-MBZUAI-LaMini-instruction-train.json_98559875.json',
                    '2024.09.23_instruct_kk_QnA-ignmilton-ign-clean-instruct-dataset-500k-train.json_143251526.json',
                    '2024.09.23_instruct_kk_QnA-nvidia-HelpSteer-train.json_12192564.json',
                    '2024.09.23_instruct_kk_QnA-nvidia-HelpSteer-validation.json_626944.json',
                    '2024.09.23_instruct_kk_Summarization-billsum-ca-test.json_1249997.json',
                    '2024.09.23_instruct_kk_Summarization-billsum-test.json_2250638.json',
                    '2024.09.23_instruct_kk_Summarization-billsum-train.json_13101432.json',
                    '2024.09.23_instruct_kk_Summarization-gigaword-test.json_53250.json',
                    '2024.09.23_instruct_kk_Summarization-gigaword-train.json_106578348.json',
                    '2024.09.23_instruct_kk_Summarization-gigaword-validation.json_5143251.json',
                    '2024.09.23_instruct_kk_Summarization-knkarthick-dialogsum-test.json_195908.json',
                    '2024.09.23_instruct_kk_Summarization-knkarthick-dialogsum-train.json_1651473.json',
                    '2024.09.23_instruct_kk_Summarization-knkarthick-dialogsum-validation.json_64177.json',
                    '2024.09.23_instruct_kk_QnA-Open-Orca-OpenOrca-train_808125526.json',]
    all_data_files_3 = ['2024.09.23_instruct_en_QnA-IlyaGusev-gpt-roleplay-realm-en_2268990.json',
                    '2024.09.23_instruct_en_QnA-MBZUAI-Bactrian-X-train_7367678.json',
                    '2024.09.23_instruct_en_QnA-hakurei-open-instruct-v1-train_18806951.json',
                    '2024.09.23_instruct_en_Summarization-ccdv-WCEP-10-test_1020747.json',
                    '2024.09.23_instruct_en_Summarization-ccdv-WCEP-10-train_8130948.json',
                    '2024.09.23_instruct_en_Summarization-ccdv-WCEP-10-validation_1030088.json',
                    '2024.09.23_instruct_en_Summarization-cwebis-tldr-17-train_796539166.json',
                    '2024.09.23_instruct_ru_QnA-IlyaGusev-ru-turbo-alpaca-train_7103421.json',
                    '2024.09.23_instruct_ru_QnA-IlyaGusev-ru-turbo-saiga-train-IlyaGusev-ru-sharegpt-cleaned-train_244115.json',
                    '2024.09.23_instruct_kk_QnA-IlyaGusev-gpt-roleplay-realm-en.json_1745400.json',
                    '2024.09.23_instruct_kk_QnA-IlyaGusev-ru-turbo-alpaca-train.json_5634986.json',
                    '2024.09.23_instruct_kk_QnA-IlyaGusev-ru-turbo-saiga-train-IlyaGusev-ru-sharegpt-cleaned-train.json_223410.json',
                    '2024.09.23_instruct_kk_QnA-MBZUAI-Bactrian-X-train.json_5862167.json',
                    '2024.09.23_instruct_kk_Summarization-ccdv-WCEP-10-test.json_806955.json',
                    '2024.09.23_instruct_kk_Summarization-ccdv-WCEP-10-train.json_6417518.json',
                    '2024.09.23_instruct_kk_Summarization-ccdv-WCEP-10-validation.json_812622.json',
                    '2024.09.23_instruct_kk_Summarization-cwebis-tldr-17-train.json_597640852.json',
                    '2024.09.23_instruct_kk_QnA-hakurei-open-instruct-v1-train_14638854.json']
    all_data_files_4 = ['2024.10.09_instruct_ru_QnA-IlyaGusev-gpt-roleplay-realm-en.json_1908955.json',
                    '2024.10.09_instruct_ru_QnA-MBZUAI-Bactrian-X-train.json_6436776.json',
                    '2024.10.09_instruct_ru_QnA-MBZUAI-LaMini-instruction-train.json_108813349.json',
                    '2024.10.09_instruct_ru_QnA-hakurei-open-instruct-v1-train_16425504.json',
                    '2024.10.09_instruct_ru_QnA-hotpot-qa-train.json_34590850.json',
                    '2024.10.09_instruct_ru_QnA-hotpot-qa-validation.json_3108330.json',
                    '2024.10.09_instruct_ru_QnA-ignmilton-ign-clean-instruct-dataset-500k-train.json_155490156.json',
                    '2024.10.09_instruct_ru_QnA-nvidia-HelpSteer-train.json_14000455.json',
                    '2024.10.09_instruct_ru_QnA-nvidia-HelpSteer-validation.json_719702.json',
                    '2024.10.09_instruct_ru_QnA-rajpurkar-squad-train.json_3655425.json',
                    '2024.10.09_instruct_ru_QnA-rajpurkar-squad-validation.json_214937.json',
                    '2024.10.09_instruct_ru_Summarization-GEM-wiki-lingua-test-tr.json_266597.json',
                    '2024.10.09_instruct_ru_Summarization-GEM-wiki-lingua-test.json_9218459.json',
                    '2024.10.09_instruct_ru_Summarization-GEM-wiki-lingua-train-tr.json_985131.json',
                    '2024.10.09_instruct_ru_Summarization-GEM-wiki-lingua-train.json_32110610.json',
                    '2024.10.09_instruct_ru_Summarization-GEM-wiki-lingua-validation-tr.json_141261.json',
                    '2024.10.09_instruct_ru_Summarization-GEM-wiki-lingua-validation.json_4489451.json',
                    '2024.10.09_instruct_ru_Summarization-billsum-ca-test.json_1372923.json',
                    '2024.10.09_instruct_ru_Summarization-billsum-test.json_2502525.json',
                    '2024.10.09_instruct_ru_Summarization-billsum-train.json_14564632.json',
                    '2024.10.09_instruct_ru_Summarization-ccdv-WCEP-10-test.json_918634.json',
                    '2024.10.09_instruct_ru_Summarization-ccdv-WCEP-10-train.json_7314195.json',
                    '2024.10.09_instruct_ru_Summarization-ccdv-WCEP-10-validation.json_925643.json',
                    '2024.10.09_instruct_ru_Summarization-csebuetnlp-xlsum-test.json_4779570.json',
                    '2024.10.09_instruct_ru_Summarization-csebuetnlp-xlsum-train.json_127250329.json',
                    '2024.10.09_instruct_ru_Summarization-csebuetnlp-xlsum-validation.json_4781938.json',
                    '2024.10.09_instruct_ru_Summarization-cwebis-tldr-17-train.json_719449603.json',
                    '2024.10.09_instruct_ru_Summarization-gigaword-test.json_60615.json',
                    '2024.10.09_instruct_ru_Summarization-gigaword-train.json_121792981.json',
                    '2024.10.09_instruct_ru_Summarization-gigaword-validation.json_5806889.json',
                    '2024.10.09_instruct_ru_Summarization-knkarthick-dialogsum-test.json_200114.json',
                    '2024.10.09_instruct_ru_Summarization-knkarthick-dialogsum-train.json_1686812.json',
                    '2024.10.09_instruct_ru_Summarization-knkarthick-dialogsum-validation.json_65873.json',
                    '2024.10.09_instruct_ru_Summarization-multi-news-test.json_5510201.json',
                    '2024.10.09_instruct_ru_Summarization-multi-news-train.json_44285838.json',
                    '2024.10.09_instruct_ru_Summarization-multi-news-validation.json_5512118.json',
                    '2024.10.09_instruct_ru_ideology.json_306119.json',
                    '2024.10.09_instruct_ru_synthetic.json_1496512.json',
                    '2024.10.09_instruct_ru_synthetic2.json_11568353.json',
                    '2024.10.09_instruct_kk_ucinlp_drop_train_0-kk_13522894.json',
                    '2024.10.09_instruct_kk_ucinlp_drop_validation_0-kk_1399055.json',
                    '2024.10.09_instruct_ru_ucinlp_drop_train_0-ru_15565085.json',
                    '2024.10.09_instruct_ru_ucinlp_drop_validation_0-ru_1601094.json',
                    'cab_top_200.json',
                    '2024.10.09_instruct_ru_QnA-Open-Orca-OpenOrca-train_910608715.json']
    all_data_files_5 = ['2024.10.23_instruct_en_benchmark-mmlu_mcq_28770300.json',
                    '2024.10.23_instruct_kk_benchmark-mmlu_mcq_17333163.json',
                    '2024.10.23_instruct_ru_benchmark-mmlu_mcq_2184352.json',
                    '2024.10.23_instruct_tr_benchmark-mmlu_mcq_1823191.json',
                    '2024.10.29_instruct_ru_benchmark-mmlu_mcq_all_23601662.json',
                    '2024.10.29_instruct_tr_benchmark-mmlu_mcq_all_20117290.json']
    
    all_data = [all_data_files_1, all_data_files_2, all_data_files_3, all_data_files_4, all_data_files_5]
    all_data_name = ["critical_instruct_1", "critical_instruct_2", "critical_instruct_3", "critical_instruct_4", "experiments_critical_bench_mmlu_mcq"]

    main_list = api.list_repo_files(repo_id = repo_id, repo_type = "dataset")

    main_path = "/raid/adal_abilbekov/workspace/torchtune_new_era/datasets/"

    for name, all_data_files in zip(all_data_name, all_data):

        filtered_main = [item for item in main_list if any(sub in item for sub in all_data_files)]
        print(filtered_main)

        path = f"{main_path}{name}"

        if not os.path.exists(path):
            os.makedirs(path)

        for file in filtered_main:
            file_to_save = file.split('/')[-1].replace(".json.gz", ".json")
            dataset = load_dataset(repo_id, data_files=file, split="train")
            try:
                dataset_final = dataset.select_columns(['instruction', 'input', 'output'])
                path_to_save = f"{path}/{file_to_save}"
                dataset_final.to_json(path_to_save, lines=True, force_ascii=False)
            except Exception as e:
                print("*"*100)
                print(file_to_save)
                print(dataset.column_names)
                print("*"*100)