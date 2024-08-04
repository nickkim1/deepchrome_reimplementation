# deepchrome_reimplementation
Reimplementation of deepchrome model from the following paper: https://academic.oup.com/bioinformatics/article/32/17/i639/2450757

The following scripts were created: 
1. deepchrome_doall.py: runs everything from training to evaluation on DeepChrome model, generates AUC and loss scores, and saves into a specified directory. 
2. ml_doall.py: runs all the machine learning baselines from the original paper (random forest, svm) + an additional XGBoost model for 
comparison to the core deepchrome model
3. featmap.py: generates combinatorial interaction heatmaps based on an input-optimized base model. 
4. optim.py: optimizes a base model based on inputs, following the procedure as specified within the paper 

Subfolder descriptions: 
1. models_ckpt: excluded - stores model checkpoints. Not too large to store but still not pushed for convenience sake
2. old_scripts: included - just stores old versions of scripts you see in the main repo 
3. toy_data: stores toy data for the E001 and E002 cell types. The real data was excluded for pushes up to the main repo
4. toy_results: some of the toy results, generated from a variety of different smaller-scale versions of the real data
5. real_results: final, compiled, results: namely 1. AUC metrics across all cell types, 2. peak AUC - as seen for E123 per the paper, 3. replicated feature map heatmaps as seen for the cell types in the paper. Use this as the benchmark for model performance. Note a lot of these images are blurry because they were generated on Oscar (hpc cluster), screenshotted into lab notebook, then afterwards screenshotted into this repository for temporary viewing.