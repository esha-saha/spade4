# SPADE4
Implementation of Sparsity and Delay Embedding based Forecasting of Epidemics (SPADE4)

## Problem Statement
Given data { $y(t_k)$ } $_{k=1}^m$ corresponding to the solution of a multidimensional dynamical system based on epidemic models (daily active cases or cumulative cases), we aim to predict $y(t)$ over a short-term forecasting window.

## Method

SPADE4 uses random features with delay embedding to forecast epidemic over short term windows. Motivated by Takens' theorem, we assume that the rate of change in the observable $y(t)$ is a function of its time delayed mapping i.e.,

    $y'(t_k) = f(y(t_k), y(t_{k-1}),...,y(t_{k-(p-1)})) = f(\mathbf{h}_k)$,
    
where $p$ is the embedding dimension. We wish to learn the function $f:\mathbb{R}^p\rightarrow\mathbb{R}$ of the form

$f(\mathbf{h}_k) \approx \sum\limits_{j=1}^N c_j \phi(\langle \mathbf{h},\boldsymbol{\omega}_j\rangle)$,

 where $\mathbf{h},\,\boldsymbol{\omega}_j\in\mathbb{R}^{p}$, $\boldsymbol{\omega}_j$ are the random weights, $\phi$ is a nonlinear activation function and $\mathbf{c} =[c_1,... c_N]\in\R^N$ is the trainable coefficient vector
