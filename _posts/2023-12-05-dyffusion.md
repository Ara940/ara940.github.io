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
    <ul>
      <li><a href="#training-dyffusion"> Training DYffusion </a></li>
      <li><a href="#temporal-interpolation-as-a-forward-process"> Temporal interpolation as a forward process </a></li>
      <li><a href="#forecasting-as-a-reverse-process"> Forecasting as a reverse process </a></li>
      <li><a href="#sampling-from-dyffusion"> Sampling from DYffusion </a></li>
    </ul>
  </nav>
</d-contents>


<div class='l-body' align="center">
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/diagram.gif">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
DYffusion forecasts a sequence of $h$ snapshots $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_h$ 
given the initial conditions $\mathbf{x}_0$ similarly to how standard diffusion models are used to sample from a distribution.</figcaption>
</div>


## Introduction

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

## Notation & Background

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
while for a multi-step training approach $$\mathbf{s}^{(0)} = \mathbf{x}_{t+1:t+h}$$.<d-footnote>In practice, $R_\theta$ can also be trained to predict the Gaussian noise that has 
been added to the data sample using a score matching objective <d-cite key="ho2020ddpm"></d-cite>.</d-footnote>
We use the latter as a baseline since this is common in the related field of video diffusion models, 
and it is established that multi-step training aids inference rollout performance and stability <d-cite key="weyn2019canmachines, ravuri2021skilful, brandstetter2022message"></d-cite>.
Lastly, autoregressive single-step forecasting with a standard diffusion model would be extremely time-consuming during inference time.

## DYffusion: Dynamics-informed Diffusion Model

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

### Training DYffusion
We split the learning of DYffusion's forward and reverse processes into two separate objectives and stages as described below.

#### Temporal interpolation as a forward process

To learn our proposed temporal forward process,
we train a time-conditioned network $$\mathcal{I}_\phi$$ to interpolate between snapshots of data. 
Given a horizon $$h$$, we train the interpolator net so that 
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
where we note that the range $$(0, 1)$$ is outside the training regime but may be useful in some cases. 
It is crucial for the interpolator, $$\mathcal{I}_\phi$$,
to _produce stochastic outputs_ within DYffusion so that its forward process is stochastic, and it can generate probabilistic forecasts at inference time.
We enable this using Monte Carlo dropout <d-cite key="gal2016dropout"></d-cite> at inference time.


#### Forecasting as a reverse process

In the second stage, we train a forecaster network $$F_\theta$$ to forecast $$\mathbf{x}_{t+h}$$
such that $$F_\theta(\mathcal{I}_\phi(\mathbf{x}_{t}, \mathbf{x}_{t+h}, i \vert \xi), i)\approx \mathbf{x}_{t+h}$$
for $$i \in S =[i_n]_{n=0}^{N-1}$$, where $$S$$ denotes a schedule coupling the diffusion step to the interpolation timestep. 
The interpolator network, $$\mathcal{I}$$, is frozen with inference stochasticity enabled,
represented by the random variable $$\xi$$. 
In our experiments, $$\xi$$ stands for the randomly dropped out weights of the neural network and is omitted henceforth for clarity.
Specifically, we seek to optimize the objective

$$
\begin{equation}
    \min_\theta 
        \mathbb{E}_{n \sim \mathcal{U}[\![0, N-1]\!], \mathbf{x}_{t, t+h}\sim \mathcal{X}}
        \left[\|
            F_\theta(\mathcal{I}_\phi(\mathbf{x}_{t}, \mathbf{x}_{t+h}, i_n \vert \xi), i_n) - \mathbf{x}_{t+h}
        \|^2 \right].
\label{eq:forecaster}
\end{equation}
$$

To include the setting where $$F_\theta$$ learns to forecast the initial conditions, 
we define $$i_0 := 0$$ and $$\mathcal{I}_\phi(\mathbf{x}_{t}, \cdot, i_0) := \mathbf{x}_t$$.
In the simplest case, the forecaster network is supervised by all possible timesteps given
by the temporal resolution of the training data. That is, $$N=h$$ and $$S = [j]_{j=0}^{h-1}$$. 
Generally, the interpolation timesteps should satisfy $$0 = i_0 < i_n < i_m < h$$ for $$0 < n < m \leq N-1$$.
Given the equivalent roles between our forecaster net and the denoising net in diffusion models, 
we also refer to them as the diffusion backbones.
As the time condition to our diffusion backbone is $$i_n$$ instead of $$n$$,
we can choose _any_ diffusion-dynamics schedule during training or inference 
and even use $$F_\theta$$ for unseen timesteps.  

[//]: # (Make algo image only be 75% of the page width)
<div class='l-body' align="center">
<img class="img-fluid rounded" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/algo-training.png" width="75%">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px">DYffusion's two-stage training procedure is summarized in the algorithm above. </figcaption>
</div>

Because the interpolator $$\mathcal{I}_\phi$$ is frozen in the second stage,
the imperfect forecasts  $$\hat{\mathbf{x}}_{t+h} = F_\theta(\mathcal{I}_\phi(\mathbf{x}_{t}, \mathbf{x}_{t+h}, i_n), i_n)$$
may degrade accuracy when used during sequential sampling. 
To handle this, we introduce an optional one-step look-ahead loss term 
$$\|  F_\theta(\mathcal{I}_\phi(\mathbf{x}_{t}, \hat{\mathbf{x}}_{t+h}, i_{n+1}), i_{n+1}) - \mathbf{x}_{t+h} \|^2$$ 
whenever $$n+1 < N$$ and weight the two loss terms equally. 
Additionally, providing a clean or noised form of the initial conditions $$\mathbf{x}_t$$ as an additional input to
the forecaster net can improve performance. 
These additional tricks are discussed in more details in the Appendix B of <a href="https://arxiv.org/abs/2306.01984">our paper</a>.

### Sampling from DYffusion

Our above design for the forward and reverse processes of DYffusion, implies the following generative process:
$$
\begin{align}
    p_\theta(\mathbf{s}^{(n+1)} | \mathbf{s}^{(n)}, \mathbf{x}_t) = 
    \begin{cases}
        F_\theta(\mathbf{s}^{(n)}, i_{n})  & \text{if} \ n = N-1 \\
        \mathcal{I}_\phi(\mathbf{x}_t, F_\theta(\mathbf{s}^{(n)}, i_n), i_{n+1}) & \text{otherwise,}
    \end{cases} 
    \label{eq:new-reverse}
\end{align}
$$

where $$\mathbf{s}^{(0)}=\mathbf{x}_t$$ and $$\mathbf{s}^{(n)}\approx\mathbf{x}_[t+i_n]$$ 
correspond to the initial conditions and predictions of intermediate steps, respectively.
In our formulations, we reverse the diffusion step indexing to align with the temporal indexing of the data. 
That is, $$n=0$$ refers to the start of the reverse process with $$\mathbf{s}^{(0)}=\mathbf{x}_t$$, 
while $$n=N$$ refers to the final output of the reverse process with $$\mathbf{s}^{(N)}\approx\mathbf{x}_{t+h}$$.
Our reverse process steps forward in time, in contrast to the mapping from noise to data in standard diffusion models. 
As a result, DYffusion should require fewer diffusion steps and data.

Our forecasting stage as detailed in Eq.~\eqref{eq:forecaster}, follows the generalized diffusion model objective.
This similarity allows us to use existing diffusion model sampling methods for inference.
In our experiments, we use the sampling algorithm from <d-cite key="bansal2022cold"></d-cite> that we adapt to our setting as shown below.

[//]: # (Make algo image only be 75% of the page width)
<div class='l-body' align="center">
<img class="img-fluid rounded" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/algo-sampling-cold.png" width="75%">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px">Sampling algorithm for DYffusion. </figcaption>
</div>

[//]: # (In Appendix~\ref{appendix:ode-cold-is-better}, we also discuss a simpler but less performant sampling algorithm.)
During the sampling process, our method essentially alternates between forecasting and interpolation, 
as illustrated in the figure below.
$$R_\theta$$ always predicts the last timestep, $$\mathbf{x}_{t+h}$$, 
but iteratively improves those forecasts as the reverse process comes closer in time to $$t+h$$.
This is analogous to the iterative denoising of the ``clean'' data in standard diffusion models.
This motivates line $$6$$ of Alg. 2, where the final forecast of $$\mathbf{x}_{t+h}$$ can be used to
fine-tune intermediate predictions or to increase the temporal resolution of the forecast.

<div class='l-body' align="center">
<img class="img-fluid rounded" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/sampling-unrolled.png" width="75%">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px">
During sampling, DYffusion essentially alternates between forecasting and interpolation, following Alg. 2. 
In this example, the sampling trajectory follows a simple schedule of going through all integer timesteps that precede the horizon of $h=4$,
with the number of diffusion steps $N=h$. 
The output of the last diffusion step is used as the final forecast for $\hat\mathbf{x}_4$. 

[//]: # (Write The \color{black}black in md)
The <span style="color:black">**black**</span> lines represent forecasts by the forecaster network, $F_\theta$.
The <span style="color:blue">**blue**</span> lines represent the subsequent temporal interpolations performed by the interpolator network, $\mathcal{I}_\phi$.
The first forecast is based on the initial conditions, $\mathbf{x}_0$.

</figcaption>
</div>

DYffusion can be applied autoregressively to forecast even longer rollouts beyond the training horizon, 
as demonstrated by our Navier-Stokes and spring mesh experiments.

### Results





### Conclusion

DYffusion is the first diffusion model that relies on task-informed forward and reverse processes.
All other existing diffusion models, albeit more general, use data corruption-based processes. 
As a result, our work provides a new perspective on designing a capable diffusion model, and may lead to a whole family of task-informed diffusion models.

For more details, please check out our [NeurIPS 2023 paper](https://arxiv.org/abs/2306.01984),
and our [code on GitHub](https://github.com/Rose-STL-Lab/dyffusion).
