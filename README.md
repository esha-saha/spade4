# ♠️ SPADE4: Sparsity and Delay Embedding based Forecasting of Epidemics
Implementation of Sparsity and Delay Embedding based Forecasting of Epidemics (SPADE4)

## Problem Statement
Given data { $y(t_k)$ } $_{k=1}^m$ corresponding to the solution of a multidimensional dynamical system based on epidemic models (daily active cases or cumulative cases), we aim to predict $y(t)$ over a short-term forecasting window.

## Method

Given $m$ measurements of the obervable $y$, we first build the input-output pairs $\{(\mathbf{h_k},y'(t_k))\}_{k=p}^m$ where $y'(t_k)$ is obtained using finite difference methods. SPADE4 uses random features with delay embedding to forecast epidemic over short term windows. Motivated by Takens' theorem, we assume that the rate of change in the observable $y(t)$ is a function of its time delayed mapping i.e.,

$y'(t_k) = f(y(t_k), y(t_{k-1}),...,y(t_{k-(p-1)})) = f(\mathbf{h_k})$ ,

where $p$ is the embedding dimension. We wish to learn the function $f:\mathbb{R}^p\rightarrow\mathbb{R}$ of the form

$f(\mathbf{h_k}) \approx \sum\limits_{j=1}^N c_j \phi(\langle \mathbf{h_k},\boldsymbol{\omega}_j\rangle)$,

where $\mathbf{h_k}, \boldsymbol{\omega}_j\in\mathbb{R}^{p}$, $\boldsymbol{\omega}_j$ are the random weights, $\phi$ is a nonlinear activation function and $\mathbf{c} =[c_1,... c_N]\in\mathbb{R}^N$ is the trainable coefficient vector learnt from the minimization problem 

$\text{argmin}_{\mathbf{c}} ||\mathbf{A}\mathbf{c} - \mathbf{z}||_2^2 + \lambda ||\mathbf{c}||_1$,

where $\mathbf{A} = (\phi(\langle \mathbf{h}_k,\boldsymbol{\omega}_j\rangle))\in\mathbb{R}^{ (m-p+1)\times N}$

Finally use Euler method to learn { $y(t_k)$ } $_{k=m+1}^T$ by 

$y(t_{m+i}) = y(t_{m+i-1}) +  (t_{m+i} - t_{m+i-1})f(\mathbf{h_{m+i-1}}),\quad i=1,...,T$

and $T$ denotes the forecasting window ( $T = 7$ or $T=14$ ).
   
 
## Contact and citation

Email esaha@uwaterloo.ca if you have any questions, comments or suggestions. Please cite the associated paper if you found the code useful in any way:

      @misc{https://doi.org/10.48550/arxiv.2211.08277,
      doi = {10.48550/ARXIV.2211.08277},
      url = {https://arxiv.org/abs/2211.08277},
      author = {Saha, Esha and Ho, Lam Si Tung and Tran, Giang},
      keywords = {Machine Learning (cs.LG), Physics and Society (physics.soc-ph), Populations and Evolution (q-bio.PE), FOS: Computer and information        sciences, FOS: Computer and information sciences, FOS: Physical sciences, FOS: Physical sciences, FOS: Biological sciences, FOS: Biological sciences},
      title = {SPADE4: Sparsity and Delay Embedding based Forecasting of Epidemics},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
      }
