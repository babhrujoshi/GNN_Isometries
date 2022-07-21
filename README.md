


**Problem**

Consider the compressed sensing problem of recovering $x\in\mathbb{R}^n$ from noisy measurements of the form

$$y = A x_{0} + \epsilon, $$

where $\epsilon\in\mathbb{R}^n$ is noise and $A$ is a sub-sampled Fourier matrix (or general isometry). We assume the unknown signal $x_0$ lives in the range of known generative model $G:\mathbb{R}^k \rightarrow \mathbb{R}^n$, i.e. $x_{0} = G(z_0)$ for some $z_0 \in \mathbb{R}^k$. We assume the generative model $G$ is  fully-connected feedforward network of the form 

$$ G(x) = A_d\sigma(A_{d-1} \cdots \sigma(A_1 z)\cdots),$$

where $A_i \in \mathbb{R}^{n_i \times n_{i-1}}$ is the weight matrix and $\sigma(\cdot)$ is the activation function. We
determine the conditions (on $A, G, x_{0}$, \etc) under which it is possible to (approximately) recover $x_{0}$ from noisy linear measurements $y$ by (approximately) solving an optimization problem of the form

$$\min_{z \in \mathbb{R}^{k}} ||b - A G(z) ||_{2}. $$

**File Structure**
1. `src` contains notebooks used to generate figures
2. `figures` contains figures in the paper
3. `trained_GNN` contains trained decoder 
4. `saved_data` contains saved experiments in jld format 

