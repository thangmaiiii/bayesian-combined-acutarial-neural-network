import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from itertools import product
from codebase_0 import BayesianLinearPredictor
from pyro import poutine

class HorseshoeMLP(PyroModule):
    """
    MLP with regularised horseshoe priors (Piironen & Vehtari, 2017).

    Prior hierarchy for each weight w:
        τ   ~ HalfCauchy(τ₀)                  global shrinkage
        c²  ~ InvGamma(ν/2,  ν s²/2)          slab variance
        λ   ~ HalfCauchy(1)                   local shrinkage  (per weight)
        w   ~ N(0, σ̃²)

    where   σ̃  =  τ λ c / √(c² + τ² λ²)

    When τλ ≪ c  →  σ̃ ≈ τλ     (standard horseshoe, aggressive shrinkage)
    When τλ ≫ c  →  σ̃ ≈ c      (slab caps the variance for large signals)

    Biases receive standard N(0, 1) priors — shrinkage on biases is
    rarely motivated and complicates the guide for little gain.
    """

    def __init__(
        self,
        in_features,
        hidden_activation_nodes,
        hidden_activations=nn.Tanh,
        tau_0=1.0,
        slab_scale=2.0,
        slab_df=4.0,
    ):
        super().__init__()
        self.tau_0 = tau_0
        self.slab_scale = slab_scale
        self.slab_df = slab_df

        # ---------- parse architecture (matching your MLP interface) ----------
        if type(hidden_activations) == list:
            pairs = list(zip(hidden_activation_nodes, hidden_activations))
        else:
            pairs = list(product(hidden_activation_nodes, [hidden_activations]))

        self.layer_dims = []   # list of (fan_in, fan_out)
        self._act_fns = []     # activation instances for hidden layers

        prev = in_features
        for nodes, act_cls in pairs:
            self.layer_dims.append((prev, nodes))
            self._act_fns.append(act_cls())
            prev = nodes
        self.layer_dims.append((prev, 1))  # output layer (no activation)

    # --------------------------------------------------------------------- #
    def _site(self, name):
        """Full Pyro name under the parent hierarchy (e.g. 'cann.neural_net.<name>')."""
        return f"{self._pyro_name}.{name}" if self._pyro_name else name

    # --------------------------------------------------------------------- #

    def _regularised_scale(self, tau, c2, lam):
        """σ̃ = τ λ c / √(c² + τ² λ²), floored at the smallest positive float so a prior
        draw of λ = 0 exactly (float32 HalfCauchy can return it) never gives Normal(0, 0)."""
        raw = tau * lam
        scale = raw * torch.sqrt(c2) / torch.sqrt(c2 + raw ** 2)
        return scale.clamp_min(torch.finfo(scale.dtype).tiny)

    # --------------------------------------------------------------------- #
    def forward(self, X):
        device = X.device

        # ---- global shrinkage (single scalar for the whole network) ----
        tau = pyro.sample(
            "nn_tau",
            dist.HalfCauchy(torch.tensor(self.tau_0, device=device)),
        )

        # ---- slab variance (shared across all layers) ----
        c2 = pyro.sample(
            "nn_c2",
            dist.InverseGamma(
                torch.tensor(self.slab_df / 2, device=device),
                torch.tensor(self.slab_df * self.slab_scale**2 / 2, device=device),
            ),
        )

        h = X
        for i, (fan_in, fan_out) in enumerate(self.layer_dims):
            k = 2 * i   # Linear index inside BayesianMLP's nn.Sequential (activations sit at odd indices)

            lam_w = pyro.sample(
                f"nn_lam_w_{i}",
                dist.HalfCauchy(torch.ones(fan_out, fan_in, device=device)).to_event(2),
            )
            scale_w = self._regularised_scale(tau, c2, lam_w)

            w = pyro.sample(
                self._site(f"net.{k}.weight"),          # was nn_w_{i}
                dist.Normal(
                    torch.zeros(fan_out, fan_in, device=device), scale_w
                ).to_event(2),
            )
            b = pyro.sample(
                self._site(f"net.{k}.bias"),            # was nn_b_{i}
                dist.Normal(
                    torch.zeros(fan_out, device=device),
                    torch.ones(fan_out, device=device),
                ).to_event(1),
            )

            h = F.linear(h, w, b)
            if i < len(self._act_fns):
                h = self._act_fns[i](h)

        return h.squeeze(-1)


class HorseshoeBayesianCANN(PyroModule):
    """
    Bayesian Combined Actuarial Neural Network.

        η(x)  =  xᵀβ  +  f(x)

    • xᵀβ  — GLM skip connection   (BayesianLinearPredictor, Normal priors)
    • f(x) — NN backbone           (HorseshoeMLP, regularised horseshoe)

    The horseshoe on the NN backbone is the Bayesian counterpart of
    zero-initialising the last layer in the frequentist CANN: global
    shrinkage τ pushes the entire network correction toward zero, while
    local shrinkage λ_j lets genuinely nonlinear features escape.

    Args
    ----
    in_features : int
    neural_net_params : dict
        Forwarded to HorseshoeMLP — must contain `hidden_activation_nodes`,
        may contain `hidden_activations`.
    glm_prior_scale : float
        Scale of the N(0, s) prior on GLM coefficients.
    tau_0 : float
        Global shrinkage scale.  Smaller → more aggressive sparsity.
        Piironen & Vehtari suggest  τ₀ = p₀ / ((p − p₀) √n)  where p₀ is
        the expected number of non-zero weights; in practice start at 1.0
        and tune.
    slab_scale : float
        Slab std dev — caps the prior variance for large signals.
    slab_df : float
        Slab degrees of freedom for the InverseGamma on c².
    """

    def __init__(
        self,
        in_features,
        neural_net_params,
        glm_prior_scale=1.0,
        tau_0=1.0,
        slab_scale=2.0,
        slab_df=4.0,
    ):
        super().__init__()

        # ---- GLM skip connection (Normal priors) ----
        self.linear_pred = BayesianLinearPredictor(in_features)
        if glm_prior_scale != 1.0:
            self.linear_pred.layer.weight = PyroSample(
                dist.Normal(0, glm_prior_scale).expand([1, in_features]).to_event(2)
            )
            self.linear_pred.layer.bias = PyroSample(
                dist.Normal(0, glm_prior_scale).expand([1]).to_event(1)
            )

        # ---- NN backbone (regularised horseshoe) ----
        self.neural_net = HorseshoeMLP(
            in_features=in_features,
            tau_0=tau_0,
            slab_scale=slab_scale,
            slab_df=slab_df,
            **neural_net_params,
        )

    def forward(self, X, kl_weight=1.0):
        # Scales log p(beta, b0, w, tau, c2, lambda) by kl_weight and nothing else.
        with poutine.scale(scale=kl_weight):
            return self.linear_pred(X) + self.neural_net(X)

class HorseshoePoissonBCANN(PyroModule):
    """
    Poisson BCANN with a regularised-horseshoe backbone.

        η(x) = xᵀβ + b₀ + f(x; w)
        μ    = exp(η)
        y    ~ Poisson(μ)      [or Poisson(exp(η + log_exposure))]

    `kl_weight` (minibatch KL weight π_i, e.g. Blundell) scales the log prior
    of every latent site; the likelihood is left unscaled.  Pair with a guide
    scaled by the same weight (`_kl_scaled`).

    • xᵀβ + b₀  — GLM skip connection  (Normal priors)
    • f(x; w)    — NN backbone          (regularised horseshoe)

    Args
    ----
    in_features : int
    neural_net_params : dict
        Forwarded to HorseshoeMLP (must contain `hidden_activation_nodes`).
    glm_prior_scale : float
        Scale of N(0, s) prior on GLM coefficients.
    tau_0, slab_scale, slab_df : float
        Horseshoe hyperparameters (see HorseshoeBayesianCANN).
    """

    def __init__(
        self,
        in_features,
        neural_net_params,
        glm_prior_scale=1.0,
        tau_0=1.0,
        slab_scale=2.0,
        slab_df=4.0,
    ):
        super().__init__()
        self.cann = HorseshoeBayesianCANN(
            in_features=in_features,
            neural_net_params=neural_net_params,
            glm_prior_scale=glm_prior_scale,
            tau_0=tau_0,
            slab_scale=slab_scale,
            slab_df=slab_df,
        )

    def forward(self, X, y=None, log_exposure=None, kl_weight=1.0):
        # Latent sites (GLM, NN weights, tau, c2, lambda) are sampled inside these
        # calls, so this scales the prior log-density by kl_weight and nothing else.
        with poutine.scale(scale=kl_weight):
            eta_glm = self.cann.linear_pred(X)
            f_nn = self.cann.neural_net(X)
        eta = eta_glm + f_nn

        mu = torch.exp(eta.clamp(max=20.0))
        rate = mu if log_exposure is None else torch.exp(
            (eta + log_exposure).clamp(max=20.0)
        )

        pyro.deterministic("eta_glm", eta_glm)
        pyro.deterministic("mu", mu)
        pyro.deterministic("rate", rate)

        with pyro.plate("data", X.shape[0]):          # no scale: likelihood weight stays 1
            pyro.sample("obs", dist.Poisson(rate), obs=y)
        return mu