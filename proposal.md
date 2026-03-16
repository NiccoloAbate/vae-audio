**Exploring Variational Autoencoders for Audio Synthesis and Timbre Transfer**

**Team:** Niccolo Abate

**Problem Statement:** VAEs can learn compact latent representations enabling reconstruction, synthesis, interpolation, and timbre transfer, but high‑fidelity waveform decoding at high sample rate (44.1-48 kHz) is difficult due to phase detail, training stability, and temporal coherence. Prior work (RAVE) suggests that multiresolution spectral losses and adversarial/feature‑matching objectives can substantially improve perceptual quality. This project asks:

* What is the minimal VAE that yields convincing audio at high sample rates?  
* What additions (additional perceptual losses, adversarial training, feature matching loss, etc.) most improve perceptual quality.

**Description:** This project will be centered around exploring VAEs, primarily for creative audio synthesis and style transfer tasks. My general outline is as follows:

* Offline MVP: Create a basic model that reconstructs and generates audio. It should have stable training, produce plausible samples, and be able to interpolate in latent space.   
* Improve Audio Quality: Improved perceptual quality with additional perceptual losses, adversarial training, and feature matching loss. Should result in higher perceived audio quality and (hopefully) no obvious artifacting or phasiness.  
* (Stretch Goal) Streaming: Ability to run the model in a streaming capacity.  
* Comparison Evaluation: Train and test existing RAVE model with the same dataset and compare the results to my model.  
* Demo / Sound Examples: Test use of the model for simple timbre transfer and morphing.

The architecture will generally be made up of convolution layers downsampling to the latent space and then upsampling back to audio.

**Motivation:** I want to get my feet wet and hands dirty with ML audio synthesis and I think this would be a great way to do this. Ultimately, I am interested in the creative synthesis potential of this approach and using the latent space as a unique way of interfacing with audio synthesis. I am worried this is an ambitious project, but at the end of the day, learning is my number 1 goal.

**Data Collection:** Some potential sources of data include:  
Well aligned options:

* Medley-solos-DB / URMP: [https://zenodo.org/records/3464194](https://zenodo.org/records/3464194)  
* ESC-50: [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)

Catch all, large options:

* Free Music Archive: [https://github.com/mdeff/fma](https://github.com/mdeff/fma)  
* Google AudioSet: [https://research.google.com/audioset/](https://research.google.com/audioset/)

References:

Caillon, A., & Esling, P. (2021). *RAVE: A variational autoencoder for fast and high-quality neural audio synthesis*. arXiv preprint arXiv:2111.05011. [https://arxiv.org/abs/2111.05011](https://arxiv.org/abs/2111.05011)

Natsiou, Anastasia & Longo, Luca & O'Leary, Sean. (2023). *Interpretable Timbre Synthesis using Variational Autoencoders Regularized on Timbre Descriptors*. [arXiv.2307.10283](https://arxiv.org/abs/2307.10283).

Bergmann, D., & Stryker, C. (n.d.). *What is a variational autoencoder?* IBM. Retrieved January 26, 2026, from [https://www.ibm.com/think/topics/variational-autoencoder](https://www.ibm.com/think/topics/variational-autoencoder)

Sadok, S., Leglaive, S., Girin, L., Alameda‑Pineda, X., & Séguier, R. (2024). A multimodal dynamical variational autoencoder for audiovisual speech representation learning. *Neural Networks, 172*, 106120\. [https://doi.org/10.1016/j.neunet.2024.106120](https://doi.org/10.1016/j.neunet.2024.106120)

Wu, Y. (2023). *Self-supervised disentanglement of harmonic and rhythmic features in music audio signals*. arXiv preprint arXiv:2309.02796. [https://arxiv.org/abs/2309.02796](https://arxiv.org/abs/2309.02796)

Han, B., Dai, J., Song, X., Hao, W., He, X., Guo, D., Chen, J., Wang, Y., & Qian, Y. (2023). *InstructME: An Instruction Guided Music Edit And Remix Framework with Latent Diffusion Models*. Retrieved from [https://arxiv.org/abs/2308.14360](https://arxiv.org/abs/2308.14360)