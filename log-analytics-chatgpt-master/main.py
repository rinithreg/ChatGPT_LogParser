import os
from chat import ChatGPT, config
from utils import get_log_messages
import pandas as pd
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dataset.data_loader import load_train_data

datasets = ['OpenStack', 'BGL', 'Linux', 'Android', 'Apache', 'Proxifier']                                              # List of datasets
MSG_LEN = 1                                                                                                             # Length of log messages to be processed at once


def zero_shot_benchmark(model, prompt_template, dataset, out_dir="."):
    chat = ChatGPT(model=model, prompt=prompt_template)                                                                 # Initializing ChatGPT function with specified model and prompt template
    _, test = get_log_messages("./", dataset, 0)                                                                        # Getting log messages for testing
    log_chunks = []
    for i in tqdm(range(len(test) // MSG_LEN)):                                                                         # Splitting log messages into chunks for processing
        log_chunks.append(test[i * MSG_LEN: (i + 1) * MSG_LEN])
    with ThreadPoolExecutor(max_workers=8) as executor:                                                                 # Multi-threaded execution to get response for each log chunk
        templates = list(
            tqdm(executor.map(lambda chunk: chat.get_response(chunk, request_type=MSG_LEN == 1), log_chunks),           # Generating templates using the ChatGPT model for each log chunk
                 total=len(log_chunks)))
        print("Completed!")

    os.makedirs("logs", exist_ok=True)                                                                                  # Creating a directory to store generated templates
    with open(f"logs/{dataset}_{out_dir}.log", mode="w") as f:                                                          # Writing generated templates into a log file
        [f.write(x[1] + "\n =================== \n") for x in templates]
    templates = [x[0] for x in templates]                                                                               # Processing the templates
    if MSG_LEN > 1:
        templates = sum(templates, [])
    unique_templates = Counter(templates).items()                                                                       # Counting occurences of each template
    logs_df = pd.read_csv(f"dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv")                               # Reading the corrected ground truth log files to associate generated templates with log entries
    logs_df.EventTemplate = pd.Series(templates)
    temp_df = pd.DataFrame(unique_templates, columns=['EventTemplate', 'Occurrences'])                                  # Creating a DataFrame to store unique templates and their occurrence counts
    os.makedirs(f"outputs/{out_dir}", exist_ok=True)                                                                    # Creating a directory to store output files
    logs_df.to_csv(f"outputs/{out_dir}/{dataset}_2k.log_structured.csv")                                                # Saving log entries with the generated log templates
    temp_df.to_csv(f"outputs/{out_dir}/{dataset}_2k.log_templates.csv")                                                 # Saving the unique log templates and their occurrence counts


def few_shot_benchmark(model, demo, prompt_template, demo_format, demo_inst, dataset, out_dir="."):
    chat = ChatGPT(model=model, prompt=prompt_template, demo_format=demo_format, demo_instruct=demo_inst)               # Initializing ChatGPT function with specified model, prompt template and instructions for training examples
    _, test = get_log_messages("./", dataset, 0)                                                                        # Getting log messages for testing
    log_chunks = []
    for i in tqdm(range(len(test) // MSG_LEN)):                                                                         # Splitting log messages into chunks for processing
        log_chunks.append(test[i * MSG_LEN: (i + 1) * MSG_LEN])
    with ThreadPoolExecutor(max_workers=2) as executor:                                                                 # Multi-threaded execution to get response for each log chunk
        templates = list(
            tqdm(executor.map(lambda chunk: chat.get_response(chunk, demos=demo), log_chunks), total=len(log_chunks)))  # Generating templates using the ChatGPT model for each log chunk
        print("Completed!")
    os.makedirs("logs", exist_ok=True)                                                                                  # Creating a directory to store generated templates
    with open(f"logs/{dataset}_{out_dir}.log", mode="w") as f:                                                          # Writing generated templates into a log file
        [f.write(x[1] + "\n =================== \n") for x in templates]
    templates = [x[0] for x in templates]                                                                               # Processing the templates
    unique_templates = Counter(templates).items()                                                                       # Counting occurences of each template
    logs_df = pd.read_csv(f"dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv")                               # Reading the corrected ground truth log files to associate generated templates with log entries
    logs_df.EventTemplate = pd.Series(templates)
    temp_df = pd.DataFrame(unique_templates, columns=['EventTemplate', 'Occurrences'])                                  # Creating a DataFrame to store unique templates and their occurrence counts
    os.makedirs(f"outputs/{out_dir}", exist_ok=True)                                                                    # Creating a directory to store output files
    logs_df.to_csv(f"outputs/{out_dir}/{dataset}_2k.log_structured.csv")                                                # Saving log entries with the generated log templates
    temp_df.to_csv(f"outputs/{out_dir}/{dataset}_2k.log_templates.csv")                                                 # Saving the unique log templates and their occurrence counts


if __name__ == '__main__':
    for dname in datasets:                                                                                              # Iterates through each dataset in the datasets list
        """ zero-shot benchmark                                                                                         
        """                                                                                                             # Performing zero-shot benchmark for current dataset
        prompt = config['ZERO_SHOT_PROMPT']                                                                             # Selecting the prompt configuration
        print(prompt['prompt'], "-" * 5, prompt['desc'])                                                                # Printing prompt information
        print(f"============== {dname} ==============")
        zero_shot_benchmark(config['MODEL'], prompt['prompt'], dname, f"{prompt['id']}")                                # Executing the zero_shot_benchmark function for specific parameters

        """ few-shot benchmark
        """                                                                                                             # Performing few-shot benchmark for current dataset
        prompt = config['FEW_SHOT_PROMPT']
        print(prompt['prompt'])
        for shot in [1, 2, 4]:                                                                                          # Iterates through different shot scenarios
            print(f"************ {shot} shot ************")
            print(f"============== {dname} ==============")
            demos = load_train_data(r_dir="./dataset", dataset=dname, shot=shot)                                                                             # Loading the training examples for current few shot scenario
            few_shot_benchmark(config['MODEL'], demos, prompt['prompt'], prompt['demo_format'], prompt['demo_instruct'], dname, f"{prompt['id']}_{shot}")    # Executing the few_shot_benchmark function for specific parameters
