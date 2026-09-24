"""
Controlled synthetic-censoring recovery experiment (paper Table 5 / tab:recovery).

Re-implements the three torch stages from the manuscript specification:
  * Appendix A.1  mu_t = b0 + b1 t + g(x_t),  log sigma_t = h(x_t)
  * Appendix A.2  Tobit (right-censored) likelihood, eq. (tobit)
  * Appendix A.3  interval-censored likelihood, eq. (interval), with the
                  log-space / reflected evaluation and the exact-observation
                  limit for degenerate intervals
  * Appendix A.4  estimation recipe: full batch, AdamW(lr=8e-3, wd=1e-4),
                  cosine anneal, shared two-layer trunk width 96 -> 48, SiLU,
                  dropout 0.10, separate linear heads, mu-head bias initialised
                  at the training mean of the response, CPU, fixed seeds.

Standardisation has two modes so the 2020 holdout failure can be exhibited and
then fixed:
    mode="legacy"  scale = sd + 1e-8      (constant training column -> 1e8)
    mode="fixed"   scale = 1 where sd~0   (Appendix A.4 correction)
"""
import numpy as np
import torch

torch.set_num_threads(4)

ALPHA_MIN = 0.30          # admission floor used by the ICG interval
P_SHED    = 0.651         # empirical Pr(shed), 2022-2024
EPOCHS    = 1500
TINY      = 1e-12

FEATS = ["cdd", "cdd2", "cdd_rh", "tmax", "RH2M",
         "sin1", "cos1", "sin2", "cos2", "weekend",
         "dow1", "dow2", "dow3", "dow4", "dow5", "dow6", "covid"]


# ---------------------------------------------------------------- standardise
def standardise(Xtr, Xte, mode):
    mu = Xtr.mean(0)
    sd = Xtr.std(0, ddof=0)
    if mode == "legacy":
        scale = sd + 1e-8
    elif mode == "fixed":
        scale = np.where(sd < 1e-10, 1.0, sd)
    else:
        raise ValueError(mode)
    return (Xtr - mu) / scale, (Xte - mu) / scale


# --------------------------------------------------------------------- network
class DemandNet(torch.nn.Module):
    """Shared trunk, separate linear heads for mu and log sigma, plus an
    explicit linear trend term outside the network (Appendix A.1)."""

    def __init__(self, d_in, y_mean, width=96, p=0.10):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(d_in, width), torch.nn.SiLU(), torch.nn.Dropout(p),
            torch.nn.Linear(width, width // 2), torch.nn.SiLU(), torch.nn.Dropout(p),
        )
        self.mu_head = torch.nn.Linear(width // 2, 1)
        self.ls_head = torch.nn.Linear(width // 2, 1)
        self.b0 = torch.nn.Parameter(torch.tensor(0.0))
        self.b1 = torch.nn.Parameter(torch.tensor(0.0))
        torch.nn.init.zeros_(self.mu_head.weight)
        torch.nn.init.zeros_(self.ls_head.weight)
        with torch.no_grad():
            self.mu_head.bias.fill_(float(y_mean))   # output bias at training mean
            self.ls_head.bias.fill_(-3.0)

    def forward(self, x, t):
        z = self.trunk(x)
        mu = self.b0 + self.b1 * t + self.mu_head(z).squeeze(-1)
        ls = self.ls_head(z).squeeze(-1).clamp(-9.0, 2.0)
        return mu, ls


# ---------------------------------------------------------------- likelihoods
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def _exact(y, mu, ls):
    z = (y - mu) * torch.exp(-ls)
    return -ls - LOG_SQRT_2PI - 0.5 * z * z


def nll_blind(y, lo, hi, cens, mu, ls):
    """Censoring-blind: exact Gaussian density on the published series."""
    return -_exact(y, mu, ls).mean()


def nll_tobit(y, lo, hi, cens, mu, ls, lam=1.0):
    """Tobit: censored days contribute the upper-tail probability
    log[1 - Phi((y-mu)/sigma)] = log_ndtr(-(y-mu)/sigma)."""
    z = (y - mu) * torch.exp(-ls)
    ll = torch.where(cens, torch.special.log_ndtr(-z), _exact(y, mu, ls))
    # stabilising penalty: the censored term saturates as mu -> inf, so the
    # latent level is otherwise unbounded above (Appendix A.2).
    pen = lam * (torch.relu(mu - y) ** 2 * cens).mean()
    return -ll.mean() + pen


def _log_interval_prob(zl, zu):
    """log(Phi(zu) - Phi(zl)) evaluated in log space, reflecting into the lower
    tail when zl > 0 so both endpoints are computed where log-CDF is accurate
    (Appendix A.3)."""
    flip = zl > 0
    a = torch.where(flip, -zu, zl)        # a < b, both preferably <= 0
    b = torch.where(flip, -zl, zu)
    la, lb = torch.special.log_ndtr(a), torch.special.log_ndtr(b)
    return lb + torch.log1p(-torch.exp((la - lb).clamp(max=-1e-12)))


def nll_icg(y, lo, hi, cens, mu, ls):
    """Interval-censored likelihood, eq. (interval)."""
    sig = torch.exp(ls)
    zl, zu = (lo - mu) / sig, (hi - mu) / sig
    width = hi - lo
    lp = _log_interval_prob(zl, zu)
    # narrow / degenerate intervals: the correct limit is the exact-observation
    # density at lo, since log(u-l) does not depend on the parameters.
    narrow = width < 1e-6
    lp_narrow = torch.log(width.clamp(min=TINY)) + _exact((lo + hi) / 2, mu, ls)
    lp = torch.where(narrow, torch.where(width < 1e-10, _exact(lo, mu, ls), lp_narrow), lp)
    ll = torch.where(cens, lp, _exact(lo, mu, ls))
    return -ll.mean()


SPECS = {"blind": nll_blind, "tobit": nll_tobit, "icg": nll_icg}


# --------------------------------------------------------------------- fitting
def fit_predict(spec, Xtr, ttr, ytr, lotr, hitr, censtr, Xte, tte, seed):
    torch.manual_seed(seed)
    net = DemandNet(Xtr.shape[1], float(ytr.mean()))
    opt = torch.optim.AdamW(net.parameters(), lr=8e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    T = lambda a: torch.as_tensor(a, dtype=torch.float64)
    Xtr_, ttr_, ytr_ = T(Xtr), T(ttr), T(ytr)
    lotr_, hitr_, c_ = T(lotr), T(hitr), torch.as_tensor(censtr, dtype=torch.bool)
    net = net.double()
    loss_fn = SPECS[spec]
    net.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        mu, ls = net(Xtr_, ttr_)
        loss = loss_fn(ytr_, lotr_, hitr_, c_, mu, ls)
        if not torch.isfinite(loss):
            return None, float("nan")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        mu_te, _ = net(T(Xte), T(tte))
        mu_tr, _ = net(Xtr_, ttr_)
        final = float(loss_fn(ytr_, lotr_, hitr_, c_, mu_tr, ls).item())
    return mu_te.numpy(), final


# ------------------------------------------------- synthetic censoring process
def impose_censoring(Dstar, depth_pool, alpha, rng):
    """Impose the empirical 2022-2024 rationing process on an uncensored era.

    shed ~ Bernoulli(P_SHED); depth d resampled from the empirical S/Dpub pool;
    served G = D*(1-d); admitted shortfall S = alpha*d*D*; published D~ = G + S.
    True suppression relative to latent demand is then E[d(1-alpha)].
    """
    n = len(Dstar)
    shed = rng.random(n) < P_SHED
    d = np.where(shed, rng.choice(depth_pool, size=n, replace=True), 0.0)
    G = Dstar * (1.0 - d)
    S = alpha * d * Dstar
    Dt = G + S
    return G, S, Dt, shed & (S > 0)
