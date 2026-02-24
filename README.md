<img width="600" alt="kan_plot" src="https://github.com/KindXiaoming/pykan/assets/23551623/a2d2d225-b4d2-4c1e-823e-bc45c7ea96f9">

# Kolmogorov-Arnold Networks (KANs)

This is the github repo for the paper ["KAN: Kolmogorov-Arnold Networks"](https://arxiv.org/abs/2404.19756) and ["KAN 2.0: Kolmogorov-Arnold Networks Meet Science"](https://arxiv.org/abs/2408.10205). You may want to quickstart with [hellokan](https://github.com/KindXiaoming/pykan/blob/master/hellokan.ipynb), try more examples in [tutorials](https://github.com/KindXiaoming/pykan/tree/master/tutorials), or read the documentation [here](https://kindxiaoming.github.io/pykan/).

Kolmogorov-Arnold Networks (KANs) are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations just like MLPs: MLPs are based on the universal approximation theorem, while KANs are based on Kolmogorov-Arnold representation theorem. KANs and MLPs are dual: KANs have activation functions on edges, while MLPs have activation functions on nodes. This simple change makes KANs better (sometimes much better!) than MLPs in terms of both model **accuracy** and **interpretability**. A quick intro of KANs [here](https://kindxiaoming.github.io/pykan/intro.html).

## Installation

Pykan can be installed via PyPI or directly from GitHub.

**Pre-requisites:**

```
Python 3.9.7 or higher
pip
```

**For developers**

```
git clone https://github.com/KindXiaoming/pykan.git
cd pykan
pip install -e .
```

**Installation via github**

```
pip install git+https://github.com/KindXiaoming/pykan.git
```

**Installation via PyPI:**

```
pip install pykan
```

Requirements

```python
# python==3.9.7
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
setuptools==65.5.0
sympy==1.11.1
torch==2.2.2
tqdm==4.66.2
pandas==2.0.1
seaborn
pyyaml
```

After activating the virtual environment, you can install specific package requirements as follows:

```python
pip install -r requirements.txt
```

**Optional: Conda Environment Setup**
For those who prefer using Conda:

```
conda create --name pykan-env python=3.9.7
conda activate pykan-env
pip install git+https://github.com/KindXiaoming/pykan.git  # For GitHub installation
# or
pip install pykan  # For PyPI installation
```

## Efficiency mode

For many machine-learning users, when (1) you need to write the training loop yourself (instead of using `model.fit()`); (2) you never use the symbolic branch, it is important to call `model.speed()` before training! Otherwise, the symbolic branch is on, which is super slow because the symbolic computations are not parallelized!

## Computation requirements

Examples in [tutorials](tutorials) are runnable on a single CPU typically less than 10 minutes. All examples in the paper are runnable on a single CPU in less than one day. Training KANs for PDE is the most expensive and may take hours to days on a single CPU. We use CPUs to train our models because we carried out parameter sweeps (both for MLPs and KANs) to obtain Pareto Frontiers. There are thousands of small models which is why we use CPUs rather than GPUs. Admittedly, our problem scales are smaller than typical machine learning tasks but are typical for science-related tasks. In case the scale of your task is large, it is advisable to use GPUs.

## Documentation

The documentation can be found [here](https://kindxiaoming.github.io/pykan/).

## Tutorials

**Quickstart**

Get started with [hellokan.ipynb](./hellokan.ipynb) notebook.

**More demos**

More Notebook tutorials can be found in [tutorials](tutorials).

## Advice on hyperparameter tuning

Many intuition about MLPs and other networks may not directly transfer to KANs. So how can I tune the hyperparameters effectively? Here is my general advice based on my experience playing with the problems reported in the paper. Since these problems are relatively small-scale and science-oriented, it is likely that my advice is not suitable to your case. But I want to at least share my experience such that users can have better clues where to start and what to expect from tuning hyperparameters.

- Start from a simple setup (small KAN shape, small grid size, small data, no reguralization `lamb=0`). This is very different from MLP literature, where people by default use widths of order `O(10^2)` or higher. For example, if you have a task with 5 inputs and 1 outputs, I would try something as simple as `KAN(width=[5,1,1], grid=3, k=3)`. If it doesn't work, I would gradually first increase width. If that still doesn't work, I would consider increasing depth. You don't need to be this extreme, if you have better understanding about the complexity of your task.

- Once an acceptable performance is achieved, you could then try refining your KAN (more accurate or more interpretable).

- If you care about accuracy, try grid extention technique. An example is [here](https://kindxiaoming.github.io/pykan/Examples/Example_1_function_fitting.html). But watch out for overfitting, see below.

- If you care about interpretability, try sparsifying the network with, e.g., `model.train(lamb=0.01)`. It would also be advisable to try increasing lamb gradually. After training with sparsification, plot it, if you see some neurons that are obvious useless, you may call `pruned_model = model.prune()` to get the pruned model. You can then further train (either to encourage accuracy or encouarge sparsity), or do symbolic regression.

- I also want to emphasize that accuracy and interpretability (and also parameter efficiency) are not necessarily contradictory, e.g., Figure 2.3 in [our paper](https://arxiv.org/pdf/2404.19756). They can be positively correlated in some cases but in other cases may dispaly some tradeoff. So it would be good not to be greedy and aim for one goal at a time. However, if you have a strong reason why you believe pruning (interpretability) can also help accuracy, you may want to plan ahead, such that even if your end goal is accuracy, you want to push interpretability first.

- Once you get a quite good result, try increasing data size and have a final run, which should give you even better results!

Disclaimer: Try the simplest thing first is the mindset of physicists, which could be personal/biased but I find this mindset quite effective and make things well-controlled for me. Also, The reason why I tend to choose a small dataset at first is to get faster feedback in the debugging stage (my initial implementation is slow, after all!). The hidden assumption is that a small dataset behaves qualitatively similar to a large dataset, which is not necessarily true in general, but usually true in small-scale problems that I have tried. To know if your data is sufficient, see the next paragraph.

Another thing that would be good to keep in mind is that please constantly checking if your model is in underfitting or overfitting regime. If there is a large gap between train/test losses, you probably want to increase data or reduce model (`grid` is more important than `width`, so first try decreasing `grid`, then `width`). This is also the reason why I'd love to start from simple models to make sure that the model is first in underfitting regime and then gradually expands to the "Goldilocks zone".

## Citation

```python
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```

## Contact

If you have any questions, please contact zmliu@mit.edu

## Author's note

I would like to thank everyone who's interested in KANs. When I designed KANs and wrote codes, I have math & physics examples (which are quite small scale!) in mind, so did not consider much optimization in efficiency or reusability. It's so honored to receive this unwarranted attention, which is way beyond my expectation. So I accept any criticism from people complaning about the efficiency and resuability of the codes, my apology. My only hope is that you find `model.plot()` fun to play with :).

For users who are interested in scientific discoveries and scientific computing (the orginal users intended for), I'm happy to hear your applications and collaborate. This repo will continue remaining mostly for this purpose, probably without signifiant updates for efficiency. In fact, there are already implmentations like [efficientkan](https://github.com/Blealtan/efficient-kan) or [fouierkan](https://github.com/GistNoesis/FourierKAN/) that look promising for improving efficiency.

For users who are machine learning focus, I have to be honest that KANs are likely not a simple plug-in that can be used out-of-the box (yet). Hyperparameters need tuning, and more tricks special to your applications should be introduced. For example, [GraphKAN](https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks) suggests that KANs should better be used in latent space (need embedding and unembedding linear layers after inputs and before outputs). [KANRL](https://github.com/riiswa/kanrl) suggests that some trainable parameters should better be fixed in reinforcement learning to increase training stability.

The most common question I've been asked lately is whether KANs will be next-gen LLMs. I don't have good intuition about this. KANs are designed for applications where one cares about high accuracy and/or interpretability. We do care about LLM interpretability for sure, but interpretability can mean wildly different things for LLM and for science. Do we care about high accuracy for LLMs? I don't know, scaling laws seem to imply so, but probably not too high precision. Also, accuracy can also mean different things for LLM and for science. This subtlety makes it hard to directly transfer conclusions in our paper to LLMs, or machine learning tasks in general. However, I would be very happy if you have enjoyed the high-level idea (learnable activation functions on edges, or interacting with AI for scientific discoveries), which is not necessariy _the future_, but can hopefully inspire and impact _many possible futures_. As a physicist, the message I want to convey is less of "KANs are great", but more of "try thinking of current architectures critically and seeking fundamentally different alternatives that can do fun and/or useful stuff".

I would like to welcome people to be critical of KANs, but also to be critical of critiques as well. Practice is the only criterion for testing understanding (实践是检验真理的唯一标准). We don't know many things beforehand until they are really tried and shown to be succeeding or failing. As much as I'm willing to see success mode of KANs, I'm equally curious about failure modes of KANs, to better understand the boundaries. KANs and MLPs cannot replace each other (as far as I can tell); they each have advantages in some settings and limitations in others. I would be intrigued by a theoretical framework that encompasses both and could even suggest new alternatives (physicists love unified theories, sorry :).

# Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery

This is the official code repository of ["Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery"](https://github.com/omron-sinicx/srsd-benchmark#citation).
This work revisits datasets and evaluation criteria for Symbolic Regression (SR), specifically focused on its potential
for scientific discovery. Focused on a set of formulas used in the existing datasets based on
Feynman Lectures on Physics, we recreate 120 datasets to discuss the performance of symbolic regression for
scientific discovery (SRSD). For each of the 120 SRSD datasets, we carefully review the properties of the formula and
its variables to design reasonably realistic sampling ranges of values so that our new SRSD datasets can be used for
evaluating the potential of SRSD such as whether or not an SR method can (re)discover physical laws from such datasets.
We also create another 120 datasets that contain dummy variables to examine whether SR methods can choose necessary
variables only. Besides, we propose to use normalized edit distances (NED) between a predicted equation and the true
equation trees for addressing a critical issue that existing SR metrics are either binary or errors between the target
values and an SR model’s predicted values for a given input. We conduct experiments on our new SRSD datasets using six
SR methods. The experimental results show that we provide a more realistic performance evaluation, and our user study
shows that the NED correlates with human judges significantly more than an existing SR metric.

[![YouTube](https://img.youtube.com/vi/MmeOXuUUAW0/0.jpg)](https://www.youtube.com/watch?v=MmeOXuUUAW0)

## Setup

We used pipenv for a Python virtual environment.

```shell
pipenv install --python 3.8
mkdir -p resource/datasets
```

You can also use conda or local Python environments.
Use `requirements.txt` if you use either conda or local Python environments.
e.g.,

```shell
# Activate your conda environment if you use conda, and run the following commands
pip install pip --upgrade
pip install -r requirements.txt
```

## Download or re-generate SRSD datasets

Our SRSD datasets are publicly available at Hugging Face Dataset repositories:

- [Easy set](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy)
- [Medium set](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_medium)
- [Hard set](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_hard)

We also created another 120 SRSD datasets by introducing dummy variables to the 120 SRSD datasets.

- [Easy set w/ dummy variables](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy_dummy)
- [Medium set w/ dummy variables](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_medium_dummy)
- [Hard set w/ dummy variables](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_hard_dummy)

Download and store the datasets at `./resource/datasets/`

If you want to re-generate the SRSD datasets,

```shell
pipenv run python dataset_generator.py --config configs/datasets/feynman/easy_set.yaml
pipenv run python dataset_generator.py --config configs/datasets/feynman/medium_set.yaml
pipenv run python dataset_generator.py --config configs/datasets/feynman/hard_set.yaml
```

Note that the resulting datasets may not exactly match those we published as the values are resampled from the same distributions.

Also, run the following command for merging all the sets

```shell
cp ./resource/datasets/srsd-feynman_easy/ ./resource/datasets/srsd-feynman_all/ -r
cp ./resource/datasets/srsd-feynman_medium/ ./resource/datasets/srsd-feynman_all/ -r
cp ./resource/datasets/srsd-feynman_hard/ ./resource/datasets/srsd-feynman_all/ -r
```

### Introduce dummy variables

To re-introduce dummy variables (columns) to the datasets, run the following commands:

```shell
pipenv run python dummy_column_mixer.py --input ./resource/datasets/srsd/easy_set/ --output ./resource/datasets/srsd/easy_set_dummy/
pipenv run python dummy_column_mixer.py --input ./resource/datasets/srsd/medium_set/ --output ./resource/datasets/srsd/medium_set_dummy/
pipenv run python dummy_column_mixer.py --input ./resource/datasets/srsd/hard_set/ --output ./resource/datasets/srsd/hard_set_dummy/
```

### Convert SRSD datasets for DSR

```shell
for srsd_category in easy medium hard; do
  for split_name in train val test; do
    pipenv run python dataset_converter.py --src ./resource/datasets/srsd-feynman_${srsd_category}/${split_name}/ \
      --dst ./resource/datasets/srsd-feynman_${srsd_category}_csv/${split_name}/ --dst_ext .csv
  done
  cp ./resource/datasets/srsd-feynman_${srsd_category}/true_eq/ ./resource/datasets/srsd-feynman_${srsd_category}_csv/true_eq/ -r
done
```

## Run existing symbolic regression baselines / your own model

Follow the instruction in [external/README.md](./external)

## Analyze equations

### SRSD-Feynman datasets

Check equation properties (e.g., number of variables and those used in sympy equation)

```shell
pipenv run python eq_analyzer.py --name feynman -simple_check
```

Visualize equation trees

```shell
pipenv run python eq_analyzer.py --name feynman -visualize --output ./eq_trees/
```

## Select the best model per dataset for DSO and AI Feynman

Due to their original implementations, DSO and AI Feynman are difficult to work with optuna for hyperparameter tuning.
If there are multiple trained models with different seeds and/or hyperparameters, select the best model per dataset
based on relative error on validation split like other methods in this repository.  
(If your model uses input variable index starting from 1 like DSO, you can still use the expressions by adding `-dec_idx` to the following command.)
e.g., DSO

```shell
pipenv run python model_selector.py --est ./dso_results/est_eq* \
    --val ~/dataset/symbolic_regression/srsd-feynman_all/val/ \
    --output ./results/dso_models/ \
    -dec_idx
```

## Compute edit distance between estimated and ground-truth equations

Both the estimated and ground-truth equations need to be "pickled" as sympy equations e.g., [sympy.sympify(eq_str)](https://github.com/omron-sinicx/srsd-benchmark/blob/main/external/gplearn/gp_runner.py#L83-L85) and [pickle the object](https://github.com/omron-sinicx/srsd-benchmark/blob/main/external/gplearn/gp_runner.py#L77-L80)  
Mathematical operations available in [sympy](https://www.sympy.org/en/index.html) are supported,
and input variables should use sympy.Symbol(f'x{index}'), where `index` is integer and starts from 0.  
(If your model uses input variable index starting from 1 like DSO, you can still use the expressions by adding `-dec_idx` to the following command.)

1 by 1

```shell
pipenv run python eq_comparator.py \
    --est external/gplearn/gplearn_w_optuna/feynman-i.12.1-est_eq.pkl \
    --gt ./resource/datasets/srsd-feynman_easy/true_eq/feynman-i.12.1.pkl
```

Batch process

```shell
pipenv run python eq_comparator.py \
    --method_name gplearn \
    --est external/gplearn/gplearn_w_optuna/ \
    --est_delim .txt-est_eq \
    --gt ./resource/datasets/srsd-feynman_easy/true_eq/ \
    --gt_delim .pkl \
    --eq_table feynman_eq.tsv \
    --dist_table feynman_dist.tsv \
    -normalize
```

Add `-dec_idx` for DSO's estimated equations to decrement variable indices since DSR's variable indices start from 1 instead of 0.

## Citation

[[OpenReview](https://openreview.net/forum?id=qrUdrXsiXX)] [[Video](https://www.youtube.com/watch?v=MmeOXuUUAW0)] [[Preprint](https://arxiv.org/abs/2206.10540)]

```bibtex
@article{matsubara2024rethinking,
  title={Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery},
  author={Matsubara, Yoshitomo and Chiba, Naoya and Igarashi, Ryo and Ushiku, Yoshitaka},
  journal={Journal of Data-centric Machine Learning Research},
  year={2024},
  url={https://openreview.net/forum?id=qrUdrXsiXX}
}
```
