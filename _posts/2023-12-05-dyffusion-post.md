---
layout: post
mathjax: true
title:  "DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting"
date:   2023-12-05
categories: jekyll update
tags: dyffusion, diffusion model, spatiotemporal, forecasting, probabilistic, generative modeling, machine learning, deep learning, neurips
author: <a href='https://salvarc.github.io/'>Salva Rühling Cachay</a>, <a href='https://b-zhao.github.io/'>Bo Zhao</a>, <a href='https://haileyjoren.github.io/'>Hailey Joren</a>, <a href='https://roseyu.com/'>Rose Yu</a>
paper_url: https://arxiv.org/abs/2306.01984
code_url: https://github.com/Rose-STL-Lab/dyffusion
excerpt: While diffusion models can successfully generate data and make predictions, they are predominantly designed for static images. We propose an approach for efficiently training diffusion models for probabilistic spatiotemporal forecasting, where generating stable and accurate rollout forecasts remains challenging, Our method, DYffusion, leverages the temporal dynamics in the data, directly coupling it with the diffusion steps in the model. We train a stochastic, time-conditioned interpolator and a forecaster network that mimic the forward and reverse processes of standard diffusion models, respectively. DYffusion naturally facilitates multi-step and long-range forecasting, allowing for highly flexible, continuous-time sampling trajectories and the ability to trade-off performance with accelerated sampling at inference time. In addition, the dynamics-informed diffusion process in DYffusion imposes a strong inductive bias and significantly improves computational efficiency compared to traditional Gaussian noise-based diffusion models. Our approach performs competitively on probabilistic forecasting of complex dynamics in sea surface temperatures, Navier-Stokes flows, and spring mesh systems.
comments: true
hidden: true

---

**TL;DR:** 
We introduce a novel diffusion model-based framework, DYffusion, for large-scale probabilistic forecasting.
We propose to couple the diffusion steps with the physical timesteps of the data, 
leading to temporal forward and reverse processes that we represent through a 
stochastic interpolator and a deterministic forecaster network, respectively.
These design choices effectively address the challenges of generating stable, accurate and probabilistic rollout forecasts.

<div align="center">

![DYffusion Diagram](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOXpvdHB5bGY1aWltbTdoYTdxNW03bmdxaG9tMDN6dGY1ZTZ2OWU5ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/h7yQszDENzsSiIUOpJ/giphy.gif)

*DYffusion forecasts a sequence of* $h$ *snapshots* $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_h$
*given the initial conditions* $\mathbf{x}_0$ *similarly to how standard diffusion models are used to sample from a distribution.*
</div>

### Motivation for our work

Obtaining _accurate and reliable probabilistic forecasts_ is an important component of policy formulation,
risk management, resource optimization, and strategic planning with a wide range of applications from
climate simulations and fluid dynamics to financial markets and epidemiology.
Often, accurate _long-range_ probabilistic forecasts are particularly  challenging to obtain. When they exist, physics-based methods typically hinge on computationally expensive
numerical simulations. In contrast, data-driven methods are much more efficient and have started to have real-world impact
in fields such as [global weather forecasting](https://www.ecmwf.int/en/about/media-centre/news/2023/how-ai-models-are-transforming-weather-forecasting-showcase-data).

Generative modeling, and especially diffusion models, have shown great success in other fields such as 
natural image generation, and video synthesis.
Diffusion models iteratively transform data back and forth between an initial distribution and the target distribution over multiple diffusion steps.
The standard approach (e.g. see [this excellent blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/))
is to corrupt the data with increasing levels of Gaussian noise in the forward process,
and to train a neural network to denoise the data in the reverse process. 
Due to the need to generate data from noise over several sequential steps, diffusion models are expensive to train and, especially, to sample from.
Recent works such as [Cold Diffusion](https://arxiv.org/abs/2208.09392), by which our work was especially inspired, have proposed to use alternative data corruption processes like blurring. 

_**Problem:**_ Common approaches for large-scale spatiotemporal problems tend to be _deterministic_ and _autoregressive_.
As such, they are often unable to capture the inherent uncertainty in the data, produce unphysical predictions,
and are prone to error accumulation for long-range forecasts. It is natural to ask how we can efficiently leverage diffusion models for large-scale spatiotemporal problems
and incorporate the temporality of the data into the diffusion model.




DYffusion presents a natural solution for both these issues, by designing a temporal diffusion model (leads to naturally training to forecast multiple steps) and embedding it into the “generalized diffusion model” framework so that by taking inspiration from existing diffusion models we can build a strong probabilistic forecasting model.


<div align="center">
<img src="/assets/2023-12-05-dyffusion/noise-diagram-gaussian.jpg" width="400" height="400" alt="Gaussian diffusion" title="Gaussian noise-based diffusion model" />
<img src="/assets/2023-12-05-dyffusion/noise-diagram-dyffusion.jpg" width="400" height="400" alt="DYffusion" title="DYffusion">
<!--img src="https://raw.githubusercontent.com/Rose-STL-Lab/dyffusion/main/assets/noise-diagram-dyffusion.png" width="400" height="400" alt="DYffusion" title="DYffusion" -->
</div>


DYffusion is the first diffusion model that relies on task-informed forward and reverse processes.
All other existing diffusion models, albeit more general, use data corruption-based processes. 
As a result, our work provides a new perspective on designing a capable diffusion model, and may lead to a whole family of task-informed diffusion models.

For more details, please check out our [NeurIPS 2023 paper](https://arxiv.org/abs/2306.01984),
and our [code on GitHub](https://github.com/Rose-STL-Lab/dyffusion).

