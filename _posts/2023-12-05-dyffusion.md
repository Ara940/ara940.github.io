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
    <div><a href="#motivation"> Motivation </a></div>
    <ul>
      <li><a href="#controllable-generation-for-inverse-problem-solving">Controllable generation for inverse problem solving</a></li>
    </ul>
  </nav>
</d-contents>

<!--- Align the gif to the center -->
<div class='l-body' align="center">
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/diagram.gif">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
DYffusion forecasts a sequence of $h$ snapshots $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_h$ 
given the initial conditions $\mathbf{x}_0$ similarly to how standard diffusion models are used to sample from a distribution.</figcaption>
</div>


### Motivation

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
Recent works such as Cold Diffusion <d-cite key="bansal2022cold"></d-cite>, by which our work was especially inspired, have proposed to use alternative data corruption processes like blurring. 

_**Problem:**_ Common approaches for large-scale spatiotemporal problems tend to be _deterministic_ and _autoregressive_.
As such, they are often unable to capture the inherent uncertainty in the data, produce unphysical predictions,
and are prone to error accumulation for long-range forecasts. It is natural to ask how we can efficiently leverage diffusion models for large-scale spatiotemporal problems
and incorporate the temporality of the data into the diffusion model. 




DYffusion presents a natural solution for both these issues, by designing a temporal diffusion model
(leads to naturally training to forecast multiple steps) and embedding it into the “generalized diffusion model” 
framework so that by taking inspiration from existing diffusion models we can build a strong probabilistic forecasting model.

[//]: # (Side-by-side images)
<!---
<div class="row l-body">
	<div class="col-sm">
	  <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/noise-diagram-gaussian.jpg">
   <figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> Graphical model for a standard diffusion model.</figcaption>
	</div>
	<div class="col-sm">
  <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/noise-diagram-dyffusion.jpg">
   <figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px">Graphical model for DYffusion. </figcaption>
  </div>
</div>
</div> -->

[//]: # (Images one below the other)
<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/noise-diagram-gaussian.jpg">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> Graphical model for a standard diffusion model.</figcaption>
</div>

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/2023-12-dyffusion/noise-diagram-dyffusion.jpg">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px">Graphical model for DYffusion. </figcaption>
</div>

DYffusion is the first diffusion model that relies on task-informed forward and reverse processes.
All other existing diffusion models, albeit more general, use data corruption-based processes. 
As a result, our work provides a new perspective on designing a capable diffusion model, and may lead to a whole family of task-informed diffusion models.

For more details, please check out our [NeurIPS 2023 paper](https://arxiv.org/abs/2306.01984),
and our [code on GitHub](https://github.com/Rose-STL-Lab/dyffusion).

