# 🎨 Generative AIGC on CIM: Variational Autoencoder (VAE)

This directory contains a hardware-aware **Convolutional Variational Autoencoder (ConvVAE)** for handwritten digit generation and reconstruction mapped onto physical Compute-in-Memory (CIM) layers.

---

## 📖 Introduction & Concept

Generative Artificial Intelligence (AIGC) models require precise latent representations and high-fidelity feedforward activations to construct clean, coherent outputs. However, deploying generative models on analog CIM hardware is heavily constrained by:
1. **Power-law conductance drift** over time.
2. **Arrhenius thermal activation** which dynamically scales $G_{min}$ and $G_{max}$ boundaries under temperature changes.
3. **Write and read noise**.

As analog weights degrade, latent space representations shift and output generation collapses into high-entropy noise. This application maps the Decoder portion of a VAE onto our simulated CIM crossbars and conv layers, and evaluates the reconstruction visual quality over a 10-year timeline under naive drift, IBM global scaling, and our unsupervised online self-healing technique.

---

## 📐 Mathematical Formulation

The Decoder takes a latent representation $z \sim \mathcal{N}(0, I)$ and projects it back to pixel space using:
$$\hat{x} = \text{Decoder}(z)$$

Under drift, the weight matrices $W$ drift according to:
$$W_j(t) = W_j(t_0) \cdot (t/t_0)^{-\nu_j}$$

Our **Online Unsupervised Self-Healing** aligns the activations of each channel dynamically by tracking running activation statistics $\mu$ and $\sigma^2$ on unlabeled streaming inference samples on-chip, and projecting them back to baseline values calibrated when fresh:
$$\hat{z}_{aligned}(t) = \frac{z(t) - \mu(t)}{\sqrt{\sigma^2(t) + \epsilon}} \cdot \sqrt{\sigma_0^2} + \mu_0$$

This alignment eliminates the linear scale decay and offset shifts caused by temperature and aging, enabling high-quality image reconstruction over decadal lifetimes.

---

## 🚀 How to Run

You can run this application directly from the root directory:
```bash
python main.py run generative_aigc --epochs 25
```

It will:
1. Load the target device profile (e.g., `FingerMemristor`).
2. Train the ConvVAE on the 8x8 digits dataset in software.
3. Map the Decoder layers to hardware-aware layers (`SelfHealingCrossbar` and `SelfHealingConv2d`).
4. Age the model to 10 years, applying drift and read noise.
5. Reconstruct test digits under **Naive**, **IBM Global Scaling**, and **Self-Healing** modes.
6. Save the visual comparative plot to `reports/generative_aigc_comparison.png`.
7. Save the quantitative markdown report to `reports/generative_aigc_report.md`.
