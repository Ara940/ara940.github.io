---
layout: distill
title:  "DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting"
date:   2023-12-05
authors: 
    - name: Salva Rühling Cachay
      url: https://salvarc.github.io/
      affiliations:
        name: UC San Diego
bibliography: blogs.bib
paper_url: https://arxiv.org/abs/2306.01984
code_url: https://github.com/Rose-STL-Lab/dyffusion
description: We introduce a novel diffusion model-based framework, DYffusion, for large-scale probabilistic forecasting.
    We propose to couple the diffusion steps with the physical timesteps of the data, 
    leading to temporal forward and reverse processes that we represent through a 
    stochastic interpolator and a deterministic forecaster network, respectively.
    These design choices effectively address the challenges of generating stable, accurate and probabilistic rollout forecasts.
comments: true
hidden: false

---

<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div><a href="#introduction"> Introduction </a></div>
    <div><a href="#limitations-of-previous-work"> Limitations of Previous Work </a></div>
    <div><a href="#our-key-idea"> Our Key Idea </a></div>
  </nav>
</d-contents>

[//]: # (    <ul>)

[//]: # (      <li><a href="#controllable-generation-for-inverse-problem-solving">Controllable generation for inverse problem solving</a></li>)

[//]: # (    </ul>)


<div class='l-body' align="center">
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/diagram.gif">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
DYffusion forecasts a sequence of $h$ snapshots $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_h$ 
given the initial conditions $\mathbf{x}_0$ similarly to how standard diffusion models are used to sample from a distribution.</figcaption>
</div>


### Introduction

Obtaining _accurate and reliable probabilistic forecasts_ is an important component of policy formulation,
risk management, resource optimization, and strategic planning with a wide range of applications from
climate simulations and fluid dynamics to financial markets and epidemiology.
Often, accurate _long-range_ probabilistic forecasts are particularly  challenging to obtain. When they exist, physics-based methods typically hinge on computationally expensive
numerical simulations. In contrast, data-driven methods are much more efficient and have started to have real-world impact
in fields such as [global weather forecasting](https://www.ecmwf.int/en/about/media-centre/news/2023/how-ai-models-are-transforming-weather-forecasting-showcase-data).

Generative modeling, and especially diffusion models, have shown great success in other fields such as 
natural image generation, and video synthesis.
Diffusion models iteratively transform data between an initial distribution
and the target distribution over multiple diffusion steps<d-cite key="sohldickstein2015deepunsupervised, ho2020ddpm, karras2022edm"></d-cite>.
The standard approach (e.g. see [this excellent blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/))
corrupts the data with increasing levels of Gaussian noise in the forward process,
and trains a neural network to denoise the data in the reverse process. 
Due to the need to generate data from noise over several sequential steps, diffusion models are expensive to train and, especially, to sample from.
Recent works such as Cold Diffusion <d-cite key="bansal2022cold"></d-cite>, by which our work was especially inspired, have proposed to use alternative data corruption processes like blurring. 

### Limitations of Previous Work

Common approaches for large-scale spatiotemporal problems tend to be _deterministic_ and _autoregressive_.
As such, they are often unable to capture the inherent uncertainty in the data, produce unphysical predictions,
and are prone to error accumulation for long-range forecasts. 
It is natural to ask how we can efficiently leverage diffusion models for large-scale spatiotemporal problems.
Given that diffusion models have been primarily designed for static data, we also ask how we can explicitly
incorporate the temporality of the data into the diffusion model.


### Our Key Idea

We introduce a solution for these issues by designing a temporal diffusion model, DYffusion.
Following the “generalized diffusion model” framework <d-cite key="bansal2022cold"></d-cite>, we
replace the forward and reverse processes of standard diffusion models
with dynamics-informed interpolation and forecasting, respectively.
This leads to a scalable generalized diffusion model for probabilistic forecasting that is naturally trained to forecast multiple timesteps.

[//]: # (Side-by-side images)
[//]: # (<div class="row l-body">)

[//]: # (	<div class="col-sm">)

[//]: # (	  <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/noise-diagram-gaussian.jpg">)

[//]: # (   <figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> Graphical model for a standard diffusion model.</figcaption>)

[//]: # (	</div>)

[//]: # (	<div class="col-sm">)

[//]: # (  <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/noise-diagram-dyffusion.jpg">)

[//]: # (   <figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px">Graphical model for DYffusion. </figcaption>)

[//]: # (  </div>)

[//]: # (</div>)

[//]: # (Images one below the other)

### Notation & Background

**Problem setup:**
We study the problem of probabilistic spatiotemporal forecasting using a dataset consisting of
a time series of snapshots $$ \mathbf{x}_t \in \mathcal{X}$$.
Here, $$\mathcal{X}$$ represents the space in which the data lies, which 
may consist of spatial dimensions (e.g., latitude, longitude, atmospheric height) and a channel dimension (e.g., velocities, temperature, humidity).
We focus on the task of forecasting a sequence of $$h$$ snapshots from a single initial condition. 
That is, we aim to train a model to learn $$P(\mathbf{x}_{t+1:t+h} \,|\, \mathbf{x}_t)$$ .
Note that during evaluation, we may evaluate the model on a larger horizon $$H>h$$ by running the model autoregressively.

**Standard diffusion models:**
Given a data sample $$\mathbf{s}^{(0)}$$, a standard diffusion model is defined through a _forward diffusion process_ 
in which small amounts of Gaussian noise are added to the sample in $$N$$ steps, producing a sequence of noisy samples 
$$\mathbf{s}^{(1)}, \ldots, \mathbf{s}^{(N)}$$. 
The step sizes are controlled by a variance schedule $$\{\beta_n \in (0, 1)\}_{n=1}^N$$ such that 
the samples are corrupted with increasing levels of noise for $$n\rightarrow N$$ and $$\mathbf{s}^{(N)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$.

$$
\begin{equation}
q(\mathbf{s}^{(n)} \vert \mathbf{s}^{(n-1)}) = \mathcal{N}(\mathbf{s}^{(n); \sqrt{1 - \beta_n} \mathbf{s}^{(n-1)}, \beta_n\mathbf{I}) \quad
q(\mathbf{s}^{(1:N)} \vert \mathbf{s}^{(0)}) = \prod^N_{n=1} q(\mathbf{s}^{(n)} \vert \mathbf{s}^{(n-1)})
\end{equation}
$$

<div class='l-body'>
<img class="img-fluid" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/noise-diagram-gaussian.png">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> Graphical model for a standard diffusion model.</figcaption>
</div>

Adopting the notation from <d-cite key="bansal2022cold"></d-cite>, for generalized diffusion models, we can also consider
a forward process operator, $$D$$, that outputs the corrupted samples $$\mathbf{s}^{(n)} = D(\mathbf{s}^{(0)}, n)$$ 
for increasing degrees of corruption $$n\in\{1,\dots, N\}$$.


<div class='l-body'>
<img class="img-fluid" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/noise-diagram-dyffusion.png">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px">Graphical model for DYffusion. </figcaption>
</div>


DYffusion is the first diffusion model that relies on task-informed forward and reverse processes.
All other existing diffusion models, albeit more general, use data corruption-based processes. 
As a result, our work provides a new perspective on designing a capable diffusion model, and may lead to a whole family of task-informed diffusion models.

For more details, please check out our [NeurIPS 2023 paper](https://arxiv.org/abs/2306.01984),
and our [code on GitHub](https://github.com/Rose-STL-Lab/dyffusion).

$$
\begin{equation}
\mathbb{E}_{t \in \mathcal{U}(0, T)}\mathbb{E}_{p_t(\mathbf{x})}[\lambda(t) \| \nabla_\mathbf{x} \log p_t(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x}, t) \|_2^2],
\end{equation}
$$