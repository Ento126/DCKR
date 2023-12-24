# DCKR
This is our Pytorch implementation for the paper: DCKR: A Diffusion Contrastive Model for Knowledge-aware Recommendation, (under review).

# Introduction
The mainstream approach employed in a knowledge graph (KG) based recommendation system (RS) aggregates information from higher-order nodes within the graphs. However, this aggregation paradigm has limitations in capturing uncertain preferences and mitigating the noise issue. Recently, diffusion models have excelled in computer vision (CV) because they can handle uncertainty and noise through representation generation. Inspired by this, we propose a Diffusion Contrastive model for Knowledge-aware Recommendation (DCKR), which enhances the system’s performance and alleviates the impact of noise by injecting uncertainty signals and fusing multi-preference information. Specifically, we embed user representations into Gaussian distributions by adding noise in the diffusion module, thereby achieving uncertainty injection and preference distribution generation. Subsequently, we inject the generated user distribution through the reverse process into a multiple preference awareness module. DCKR effectively models complex interactions and captures evolving user interests through denoising training and iterative feedback. We also designed a diffusion contrastive learning component to refine preference representations and eliminate noise effects. Extensive experiments on three public datasets demonstrate that our model consistently outperforms the SOTA. 

# Requirement
pytorch==1.10.1
numpy==1.21.6
scikit-learn==1.0.2

# Usage
The hyper-parameter search range and optimal settings have been clearly stated in the codes.

Train and Test
python main.py 

# Dataset
We provide three processed datasets: Book-Crossing, MIND, and Last.FM.
