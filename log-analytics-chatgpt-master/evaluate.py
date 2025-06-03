from evaluation.evaluator import evaluate

out_dir = "outputs/few_shot_4"

datasets = ['OpenStack', 'BGL', 'Linux', 'Android', 'Apache', 'Proxifier']

if __name__ == '__main__':
    for dataset in datasets:
        evaluate(f"dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv",
                 f"{out_dir}/{dataset}_2k.log_structured_adjusted.csv")
