# 3D-Bio-Diffusion

 Simple experiment for geometrically constrained diffusion.

## Preliminary Tools

To estimate the 3D coordinates of the atoms in the protein chain, we implemented a special **conditional denoising diffusion probabilistic model (DDPM)**. We conditioned the estimation with respect to the amino acid sequence. This seemed to me a fundamental step in order to control the generative process and verify whether we could generate specific structures in the test set. We modified the data loader to pass a tuple of two elements, one being atom coordinates and one the amino acid sequence as a string.

Below, a breakdown of the fundamental steps in our approach.

### Train Utils

`Train utils` contains five functions to preprocess data:

- `extract_backbone_coords`: Extracts the coordinates of backbone atoms, \(N, C_{\alpha}, C, O\) from a given tensor of atom positions. The input tensor has size \(B \times N \times 37 \times 3\), while the output tensor has size \(B \times N \times 4 \times 3\).
- `initialize_aa_embeddings`: Initializes a fixed (non-learnable) embedding for each amino acid, assigning a random vector for each amino acid. This embedding remains fixed throughout training.
- `create_amino_acid_positional_embedding`: Creates a positional embedding for an amino acid sequence, combining both amino acid embeddings and positional information.
- `batch_create_amino_acid_positional_embedding`: Processes a batch of AA sequences, ensuring that each sequence in the batch is padded or truncated to `cfg.max_seq_length`.
- `sinusoidal_embedding`: Generates a sinusoidal embedding for a given time-step `timestep` and embedding dimension `dim`.

### Metric Utils

`Metrics utils` contains four functions for calculating metrics on protein structure prediction. We measure three quantities:

- **MSE on distance maps** (↓ Lower is better)
- **GDT_TS and GDT_HA** (↑ Higher is better)

Functions:

- `build_distance_map`: Computes pairwise Euclidean distances between atoms in a given set of atom positions.
- `MSE_loss_distance_map`: Computes mean square error (MSE) between predicted and true distance maps.
- `align_coordinates`: Aligns estimated coordinates to true coordinates using the Kabsch algorithm.
- `compute_GDT_scores`: Computes `GDT_TS` and `GDT_HA` scores based on alignment of estimated and true coordinates.

### Train Step

The function `step` performs a single iteration for both training and validation:

- Extracts backbone atom coordinates.
- Samples a random timestep \(t\) and adds noise to the atom positions \(x_0\).
- Generates sinusoidal and positional embeddings.
- The model performs a forward pass using noisy data \(x_t\) and condition \(\zeta\) to estimate the noise \(\hat{\epsilon}\).
- Computes MSE loss between \(\hat{\epsilon}\) and \(\epsilon\).
- Evaluates MSE on distance maps, `GDT_TS`, and `GDT_HA` scores.

## Architecture: DDGM with Geometric UNet

![Geometric UNet](ddgm.pdf)

We named the employed architecture a **Geometric UNet**, which operates on geometrically structured data using the **Geometric Algebra (GA)** framework.

The key point is that GA allows a unified representation of scalars, vectors, and more complex geometric structures such as bivectors and rotors. The network leverages these operations to model spatial and rotational relationships in input data.

### Why Geometric UNet?

1. GA networks align with my PhD research.
2. No diffusion pipeline has been implemented using GA layers before.
3. My past experience in diffusion models is with UNets.
4. Transformer-based pipelines are computationally expensive.

### Components of Geometric UNet

#### GA Embedding Layers

These layers convert input tensors into **geometric representations**:

- `self.tens2geom_vec`: Converts coordinate tensors into **geometric vectors**.
- `self.tens2geom_scal`: Converts embeddings into **geometric scalars**.
- `self.geom2tens`: Converts geometric representations back into regular tensors.

#### Encoder and Decoder Blocks

The **encoder** progressively increases spatial dimensions, while the **decoder** reconstructs the input:

- `self.enc1`, `self.enc2`, `self.enc3`: Apply `GeometricProductConv1D` convolutions with `GELU` activations.

The **decoder** mirrors the encoder with progressively downsampling layers.

#### Conditioning

The `condition` tensor (concatenated sequence and timestep embeddings) is transformed into its **scalar geometric representation** and added to the third encoder layer (`enc3`).

## Training the DDGM

### Training Process

- Training runs for **100 epochs**.
- Batch size: `64`
- Optimizer: **Adam**, Learning rate: `1e-4`.
- Metrics (Loss, MSE, `GDT_TS`, `GDT_HA`) logged **every 50 batches**.

### Validation and Checkpointing

- Model is evaluated on validation dataset at the end of each epoch.
- If validation loss improves, model weights and optimizer state are saved as a checkpoint.

### Sampling

- `model_sample` generates intermediate samples every **5 epochs**.
- The denoising process iteratively refines predictions using:

\[
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon} \right) + \sigma_t z
\]

where \( z \sim \mathcal{N}(0, I) \).

## Results

### MSE Between Distance Maps

![MSE over epochs](Unknown-2.png)

- The MSE decreased steadily but **plateaued around 240 Å**.
- **MSE ~200 Å, GDT_TS < 20%, and GDT_HA < 10%** indicate that accuracy is still lacking.

### Training and Validation Metrics

**Training Metrics:**

![Training loss](Unknown-4.png)

**Validation Metrics:**

![Validation loss](Unknown.png)

### Sample Comparisons

Comparisons between ground truth and generated structures:

![Example 1](Unknown-9.png)
![Example 2](Unknown-7.png)
![Example 3](Unknown-5.png)
![Example 4](Unknown-8.png)

### Issues and Potential Fixes

- **Runaway atoms:** Atoms far from actual structures may be due to numerical instability in reverse diffusion.
- **Hyperparameter tuning:** Diffusion pipelines are sensitive to batch size and learning rate, which we didn't fine-tune due to time constraints.
- **Lack of a proper clipping strategy** at the final sampling stage.

### Estimating Samples \( \mu_{\theta} \)

We also tried estimating \( \mu_{\theta} \) instead of \( \epsilon_{\theta} \), with MSE between ground truth and estimated distance maps as loss.

- **This approach failed** due to instability in the reverse diffusion process.
- Sampled structures were erratic and error did not decrease consistently.

![Sample estimation failure](Unknown-3.png)
