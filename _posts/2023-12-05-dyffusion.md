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
    <ul>
      <li><a href="#limitations-of-previous-work"> Limitations of Previous Work </a></li>
      <li><a href="#our-key-idea"> Our Key Idea </a></li>
    </ul>
    <div><a href="#notation--background"> Notation & Background </a></div>
    <ul>
      <li><a href="#problem-setup"> Problem setup </a></li>
      <li><a href="#standard-diffusion-models"> Standard diffusion models </a></li>
    </ul>
    <div><a href="#dyffusion-dynamics-informed-diffusion-model"> DYffusion</a></div>
  </nav>
</d-contents>


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
Often, accurate _long-range_ probabilistic forecasts are particularly  challenging to obtain. 
When they exist, physics-based methods typically hinge on computationally expensive
numerical simulations <d-cite key="bauer2015thequiet"></d-cite>.
In contrast, data-driven methods are much more efficient and have started to have real-world impact
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

#### Limitations of Previous Work

Common approaches for large-scale spatiotemporal problems tend to be _deterministic_ and _autoregressive_.
As such, they are often unable to capture the inherent uncertainty in the data, produce unphysical predictions,
and are prone to error accumulation for long-range forecasts. 
It is natural to ask how we can efficiently leverage diffusion models for large-scale spatiotemporal problems.
Given that diffusion models have been primarily designed for static data, we also ask how we can explicitly
incorporate the temporality of the data into the diffusion model.


#### Our Key Idea

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

#### Problem setup
We study the problem of probabilistic spatiotemporal forecasting using a dataset consisting of
a time series of snapshots $$ \mathbf{x}_t \in \mathcal{X}$$.
Here, $$\mathcal{X}$$ represents the space in which the data lies, which 
may consist of spatial dimensions (e.g., latitude, longitude, atmospheric height) and a channel dimension (e.g., velocities, temperature, humidity).
We focus on the task of forecasting a sequence of $$h$$ snapshots from a single initial condition. 
That is, we aim to train a model to learn $$P(\mathbf{x}_{t+1:t+h} \,|\, \mathbf{x}_t)$$ .
Note that during evaluation, we may evaluate the model on a larger horizon $$H>h$$ by running the model autoregressively.

#### Standard diffusion models
Here, we adapt the 
<a src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process">common notation for diffusion models</a> 
to use a superscript $$n$$ for the diffusion states $$\mathbf{s}^{(n)}$$, 
to clearly distinguish them from the time steps of the data, $$\mathbf{x}_t$$.
Given a data sample $$\mathbf{s}^{(0)}$$, a standard diffusion model is defined through a _forward diffusion process_ 
in which small amounts of Gaussian noise are added to the sample in $$N$$ steps, producing a sequence of noisy samples 
$$\mathbf{s}^{(1)}, \ldots, \mathbf{s}^{(N)}$$. 
The step sizes are controlled by a variance schedule $$\{\beta_n \in (0, 1)\}_{n=1}^N$$ such that 
the samples are corrupted with increasing levels of noise for $$n\rightarrow N$$ and $$\mathbf{s}^{(N)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$.

$$
\begin{equation*} 
q(\mathbf{s}^{(n)} \vert \mathbf{s}^{(n-1)}) = \mathcal{N}(\mathbf{s}^{(n)}; \sqrt{1 - \beta_n} \mathbf{s}^{(n-1)}, \beta_n\mathbf{I}) \quad 
q(\mathbf{s}^{(1:N)} \vert \mathbf{s}^{(0)}) = \prod^N_{n=1} q(\mathbf{s}^{(n)} \vert \mathbf{s}^{(n-1)})
\end{equation*}
$$

<div class='l-body'>
<img class="img-fluid" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/noise-diagram-gaussian.png">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> Graphical model for a standard diffusion model.</figcaption>
</div>

Adopting the notation from <d-cite key="bansal2022cold"></d-cite>, for generalized diffusion models, we can also consider
a forward process operator, $$D$$, that outputs the corrupted samples $$\mathbf{s}^{(n)} = D(\mathbf{s}^{(0)}, n)$$ 
for increasing degrees of corruption $$n\in\{1,\dots, N\}$$.

A denoising network $$R_\theta$$, parameterized by $$\theta$$, is trained to restore $$\mathbf{s}^{(0)}$$,
i.e. such that $$R_\theta(\mathbf{s}^{(n)}, n) \approx \mathbf{s}^{(0)}$$. 
For dynamics forecasting, the diffusion model can be conditioned on the initial conditions by considering 
$$R_\theta(\mathbf{s}^{(n)}, \mathbf{x}_{t}, n)$$. Then, the diffusion model can be trained to minimize the objective

$$
\begin{equation}
    \min_\theta 
    \mathbb{E}_{n \sim \mathcal{U}[\![1, N]\!], \mathbf{x}_{t}, \mathbf{s}^{(0)}\sim \mathcal{X}}
    \left[
    \|R_\theta(D(\mathbf{s}^{(0)}, n), \mathbf{x}_{t}, n) - \mathbf{s}^{(0)}\|^2
    \right],
\label{eq:diffusionmodels}
\end{equation}
$$

where $$\mathcal{U}[\![1, N]\!]$$ denotes the uniform distribution over the integers $$\{1, \ldots, N\}$$ and
$$\mathbf{s}^{(0)}$$ is the forecasting target. 
For a single-step training approach $$\mathbf{s}^{(0)} = \mathbf{x}_{t+1}$$,
while for a multi-step training approach $$\mathbf{s}^{(0)} = \mathbf{x}_{t+1:t+h}$$. 
We use the latter as a baseline since this is common in the related field of video diffusion models, 
and it is established that multi-step training aids inference rollout performance and stability <d-cite key="weyn2019canmachines, ravuri2021skilful, brandstetter2022message"></d-cite>.
Lastly, autoregressive single-step forecasting with a standard diffusion model would be extremely time-consuming during inference time.

### DYffusion: Dynamics-informed Diffusion Model

The key innovation of our framework, DYffusion, is a reimagining of the diffusion processes to more naturally model 
spatiotemporal sequences, $$\mathbf{x}_{t:t+h}$$.
Specifically, we design the reverse (forward) process to step forward (backward) in time 
so that our diffusion model emulates the temporal dynamics in the data. 
Thus, similarly to the processes in <d-cite key="song2021ddim, bansal2022cold"></d-cite>, 
they cease to represent actual "diffusion" processes.
Differently to all prior work, our processes are _not_ based on data corruption or restoration.

<div class='l-body'>
<img class="img-fluid" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/noise-diagram-dyffusion.png">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px">Graphical model for DYffusion. </figcaption>
</div>

Implementation-wise, we replace the standard denoising network, $$R_\theta$$, with a deterministic forecaster network, $$F_\theta$$.
Because we do not have a closed-form expression for the forward process, we also need to learn it from data
by replacing the standard forward process operator, $$D$$, with a stochastic interpolator network $$\mathcal{I}_\phi$$.
Intermediate steps in DYffusion's reverse process can be reused as forecasts for actual timesteps.
Another benefit of our approach is that the reverse process is initialized with the initial conditions of the dynamics 
and operates in observation space at all times. 
In contrast, a standard diffusion model is designed for unconditional generation, and reversing from white noise requires more diffusion steps. 
For conditional prediction tasks such as forecasting, DYffusion emerges as a much more natural method that is well aligned with the task at hand.

#### Temporal interpolation as a forward process

To impose a temporal bias, we train a time-conditioned network $$\mathcal{I}_\phi$$ to interpolate between snapshots of data. 
Given a horizon $$h$$, we train $$\mathcal{I}_\phi$$ so that 
$$\mathcal{I}_\phi(\mathbf{x}_t, \mathbf{x}_{t+h}, i) \approx \mathbf{x}_{t+i}$$ for $$i \in \{1, \ldots, h-1\}$$ using the objective:

$$
\begin{equation}
    \min_\phi 
        \mathbb{E}_{i \sim \mathcal{U}[\![1, h-1]\!],  \mathbf{x}_{t, t+i, t+h} \sim \mathcal{X}}
        \left[\|
            \mathcal{I}_\phi(\mathbf{x}_t, \mathbf{x}_{t+h}, i) - \mathbf{x}_{t+i}
        \|^2 \right].
\label{eq:interpolation}
\end{equation}
$$

Interpolation is an easier task than forecasting, and we can use the resulting interpolator
for temporal super-resolution during inference to interpolate beyond the temporal resolution of the data.
That is, the time input can be continuous, with $$i \in (0, h-1)$$, 
where we note that the range $$(0, 1)$$ is outside the training regime but may improve performance in some cases. 
It is crucial for the interpolator, $$\mathcal{I}_\phi$$,
to _produce stochastic outputs_ within DYffusion so that its forward process is stochastic, and it can generate probabilistic forecasts at inference time.
We enable this using Monte Carlo dropout <d-cite key="gal2016dropout"></d-cite> at inference time.

### Conclusion

DYffusion is the first diffusion model that relies on task-informed forward and reverse processes.
All other existing diffusion models, albeit more general, use data corruption-based processes. 
As a result, our work provides a new perspective on designing a capable diffusion model, and may lead to a whole family of task-informed diffusion models.

For more details, please check out our [NeurIPS 2023 paper](https://arxiv.org/abs/2306.01984),
and our [code on GitHub](https://github.com/Rose-STL-Lab/dyffusion).
