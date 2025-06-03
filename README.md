# AIOps-LogParser
This repository is a part of COEN691 - Artificial Intelligence for Software Systems Operations course instructed by Professor Wahab Hamou-Lhadj, PhD.

In this study, we replicated a preliminary evaluation of ChatGPT for log parsing. To verify the effectiveness of ChatGPT as a log parsing tool, we are going to compare the results from ChatGPT with two other state-of-the-art log parsing tools, AEL and Drain.


## ChatGPT
Follow these steps to run the benchmark for ChatGPT:
+ Navigate to log-analytics-chatgpt-master
+ Set the Open AI API Key at /chat/__init__.py (OPEN_AI_KEY)
+ Run the script python main.py to generate log templates with ChatGPT
+ Set the output directory at /outputs/post_process.py and run the script cd outputs && python post_process.py to apply common post-process rules for log parsing.
+ Set the output directory at evaluate.py and run the script python evaluate.py


## AEL


Run the following scripts to execute the benchmark for AEL:
+ Navigate to logparser-main/logparser-main/logparser/AEL and run

```
python benchmark.py
```

## Drain

Run the following scripts to execute the benchmark for Drain:
+ Navigate to logparser-main/logparser-main/logparser/Drain and run

```
python benchmark.py
```

## Datasets
We use 6 representative log datasets from a wide range of systems for the evaluation, including distributed systems (OpenStack), supercomputers (BGL), operating systems (Linux), mobile systems (Android), server applications (Apache), and standalone software (Proxifier). Each dataset contains 2,000 manually labelled log messages The dataset originated from LogPAI. We use a corrected version that was available in the original study.

