# Bias Association Discovery Framework (BADF)

---

## Overview

The **BADF** is a comprehensive framework designed for extracting phrases from open-ended generated contexts from LLMs to discover representational harms.

---

## Story Generations

`results_responses.zip` and `results_responses_patch2.zip` contain all generated stories in csv format. Can be found in **[GoogleDrive](https://drive.google.com/drive/folders/1W2auHBgi0H8zBPsf09V7VFICGTGmkQro?usp=sharing)**

---
Below is the instructions you can run our framework to explore representational harms. Codes are in 📂 src/ and 📂 patchscope/
## Simple Instructions
Detailed instructions for running the framework are in `run_commands.txt`

1. Use `1_bias_discover.py` in 📂 src/ and `_Open-Ended_Patchscores.py` in 📂 patchscope/ to do all variations of story generations;
2. Run `2a_res_summary.py` for initial comprehensive phrase extraction;
3. Run `2b_res_summary_refine.py` for post-hoc review to remove hallucinated/bad phrases;
4. Run `3a_res_combine.py` to split combined phrases into finer-grained phrases;
5. Run `3b_res_combine_check.py` to do the second post-hoc review to check the forgotten/failed-split combined phrases from previous step and decompose them;
6. Preprocess raw data using `_3.5_data_preprocess_.ipynb`;
7. Run `4a_res_calc.py` to select significant phrases;
8. Preprocess raw data from step 7 via `_4a_final_raw_data.ipynb`;
9. Run `4b_res_filter.py` to filter exclusive phrases for each demographic identity.

---

## Results
`FINAL_RES_ALL` contains results of extracted phrases.


[//]: # (## Citation)

[//]: # ()
[//]: # (If you use DBB in your work, please cite:)

[//]: # ()
[//]: # (```bibtex)