import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import io
from PIL import Image
from IPython.display import HTML

DOMAIN = [-5, -1]  # x = log10(learning rate), i.e. LR in [1e-5, 1e-1]
scale = 0.09
offset = 0.80

def get_objective(length_scale=1.0, scale=scale, offset=offset, random_state=42) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample objective f from RBF kernel.

    Args:
        length_scale: Length scale for RBF kernel.
        random_state: Random seed.

    Returns:
        X_grid: Domain points.
        f_sample: Sampled objective function values (validation accuracy).
    """
    # Define kernel: RBF with length_scale
    kernel_f = RBF(length_scale=length_scale, length_scale_bounds="fixed")

    # Create grid for sampling
    X_grid = np.linspace(DOMAIN[0], DOMAIN[1], 1000).reshape(-1, 1)

    # Sample f, then map into a realistic validation-accuracy band in [0, 1]
    gp_f = GaussianProcessRegressor(kernel=kernel_f)
    f_sample = scale * gp_f.sample_y(X_grid, random_state=random_state).flatten() + offset
    f_sample = np.clip(f_sample, 0.0, 1.0)

    return X_grid.flatten(), f_sample

def sample_from_objective(x, objective, noise_std=0.02, random_state=42):
    """
    Sample noisy value from the objective function at point x.
    """
    idx = (np.abs(objective[0] - x)).argmin()
    return objective[1][idx] + np.random.normal(0, noise_std)




# Plotting utils

def plot_grid():
    """
    Plot 6 samples of the objective function.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Use different random states to get different samples
        x, y = get_objective(random_state=i)

        ax.plot(x, y, linestyle='-', color='b', linewidth=2)
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("log10(Learning Rate)")
        ax.set_ylabel("Validation Accuracy")
        ax.grid(True)


    plt.tight_layout()
    plt.show()

def get_viz(model, obs_x, obs_y, beta, objective=None, title="Model", save_path=None,
            figsize=(10, 6)):
    """
    Plot the model with the observations.
    """
    x_grid = np.linspace(DOMAIN[0], DOMAIN[1], 1000).reshape(-1, 1)
    y_pred, sigma = model.predict(x_grid, return_std=True)

    plt.figure(figsize=figsize)
    if objective is not None:
        plt.plot(objective[0], objective[1], linestyle=':', color='k', label='True Objective', linewidth=2)

    plt.plot(x_grid, y_pred, 'b-', label='Prediction')
    plt.fill_between(x_grid.ravel(), 
                     y_pred - beta * sigma, 
                     y_pred + beta * sigma, 
                     alpha=0.2, color='b', label=f'Confidence Interval')
    
    plt.scatter(obs_x, obs_y, c='r', marker='x', label='Observations', zorder=10)
    
    plt.xlabel("log10(Learning Rate)")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0.4, 1.1)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    plt.close()
    return img
    

def plot_gif(images_or_paths, output_path='optimization.gif', duration=5000):
    """
    Creates a GIF from a list of image paths or PIL Image objects.
    
    Args:
        images_or_paths: List of file paths or PIL Image objects.
        output_path: Path where the GIF will be saved.
        duration: Duration of each frame in milliseconds.
    """
    images = []
    for item in images_or_paths:
        if isinstance(item, str):
            images.append(Image.open(item))
        else:
            images.append(item)
            
    if not images:
        print("No images provided to create GIF.")
        return

    # Save as GIF
    images[0].save(
        output_path, 
        save_all=True, 
        append_images=images[1:], 
        duration=duration, 
        loop=0
    )

def stack_frames(frame_lists):
    """Combine several equal-length lists of PIL frames into one list of frames,
    where frame i is the horizontal concatenation of frame i from every list.
    Used to render multiple optimization runs side-by-side in a single GIF."""
    n = min(len(f) for f in frame_lists)
    combined = []
    for i in range(n):
        row = [fl[i] for fl in frame_lists]
        total_w = sum(im.width for im in row)
        max_h = max(im.height for im in row)
        canvas = Image.new("RGB", (total_w, max_h), "white")
        x = 0
        for im in row:
            canvas.paste(im, (x, 0))
            x += im.width
        combined.append(canvas)
    return combined


def _compare_runs(optimize_fn, models, betas, labels, n_queries, output_path, duration):
    """Run optimize_fn once per (model, beta, label), stitch the runs side-by-side, one GIF.

    Args:
        optimize_fn: The optimization routine (defined in the notebook). Must accept
            n_queries, beta, model, show, title_prefix, figsize and return
            (X_obs, y_obs, frames).
        models, betas, labels (list): One entry per run; zipped together.
        n_queries (int): Forwarded to optimize_fn.
        output_path (str): Where the combined GIF is written.
        duration (int): Milliseconds per frame.
    """
    from IPython.display import Image as IPImage, display

    frame_lists = []
    for model, beta, label in zip(models, betas, labels):
        # show=False: collect frames without emitting a per-run GIF
        _, _, frames = optimize_fn(n_queries=n_queries, beta=beta, model=model,
                                   show=False, title_prefix=label, figsize=(6, 4.5))
        frame_lists.append(frames)

    # Stitch the runs horizontally, frame by frame, into one GIF
    combined = stack_frames(frame_lists)
    plot_gif(combined, output_path=output_path, duration=duration)
    display(IPImage(filename=output_path))


def compare_length_scales(optimize_fn, kernels, labels, n_queries=20, beta=2.5,
                          output_path="length_scale_comparison.gif", duration=1200):
    """Run BO once per kernel (varying the length scale) side-by-side in a single GIF.

    Args:
        optimize_fn: The optimization routine (defined in the notebook).
        kernels (list): Kernels to compare, one optimization run each.
        labels (list): Panel label for each kernel (used in the frame titles).
        n_queries, beta: Forwarded to optimize_fn (same beta for every run).
        output_path (str): Where the combined GIF is written.
        duration (int): Milliseconds per frame.
    """
    models = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
              for kernel in kernels]
    _compare_runs(optimize_fn, models, betas=[beta] * len(models), labels=labels,
                  n_queries=n_queries, output_path=output_path, duration=duration)


def compare_betas(optimize_fn, betas, kernel=None, labels=None, n_queries=25,
                  output_path="beta_comparison.gif", duration=1200):
    """Run BO once per beta (varying exploration) side-by-side in a single GIF.

    Args:
        optimize_fn: The optimization routine (defined in the notebook).
        betas (list): Exploration parameters to compare, one optimization run each.
        kernel: Kernel shared by every run. Defaults to the notebook's balanced RBF.
        labels (list): Panel label for each beta. Defaults to "β = <beta>".
        n_queries: Forwarded to optimize_fn.
        output_path (str): Where the combined GIF is written.
        duration (int): Milliseconds per frame.
    """
    if kernel is None:
        kernel = (RBF(length_scale=1.0, length_scale_bounds="fixed")
                  + WhiteKernel(noise_level=0.1, noise_level_bounds="fixed"))
    if labels is None:
        labels = [f"β = {b}" for b in betas]
    # Fresh model per run so they don't share fitted state
    models = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
              for _ in betas]
    _compare_runs(optimize_fn, models, betas=betas, labels=labels,
                  n_queries=n_queries, output_path=output_path, duration=duration)


def plot_regrets(objective, regrets, output_path='regrets.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(regrets, label='Regret')
    best_value = np.max(objective[1])
    random_regrets = np.cumsum([best_value - objective[1][np.random.randint(len(objective[1]))] for _ in range(len(regrets))])
    plt.plot(random_regrets, label='Random Regret')
    plt.xlabel("Iteration")
    plt.ylabel("Regret")
    plt.title("Regret Over Iterations")
    plt.legend(loc='upper left')
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    plt.close()
    return img


# ---------------------------------------------------------------------------
# UCB "which point does the acquisition function pick next?" exercise
# ---------------------------------------------------------------------------

# Ten observations a real BO run might have gathered: a well-explored high region
# on the left and a couple of points on the far right, leaving a wide unexplored
# gap in the middle. Deliberately clustered so uncertainty (and therefore the
# acquisition function's choice) is interesting to reason about.
_UCB_OBS_X = [-4.9, -4.7, -4.4, -4.1, -3.8, -3.5, -3.2, -1.15, -1.05, -1.0]
_UCB_OBJ_SEED = 10       # hidden true accuracy curve used to generate observations
_UCB_OBJ_SCALE = 0.13    # amplitude of the exercise curve (bigger than the notebook's,
_UCB_OBJ_OFFSET = 0.76   # so the candidates sit at visibly different heights)
_UCB_LENGTH_SCALE = 0.8
_UCB_MODEL_NOISE = 0.03  # noise_level the GP assumes (keeps gap uncertainty visible)
_UCB_OBS_NOISE = 0.015   # noise added to the observed accuracies

# Four candidate learning rates, hand-placed so they have clearly different
# predicted accuracies (posterior means) and uncertainties:
#   A  well-observed, clearly low     (obviously not worth it)
#   B  deep in the gap, most uncertain (pure-exploration trap)
#   C  edge of the gap: fairly high mean AND still uncertain  <-- UCB's choice
#   D  observed peak: highest mean but tiny uncertainty        (pure-exploitation trap)
_UCB_CANDIDATES = {"A": -3.4, "B": -2.3, "C": -1.75, "D": -1.1}


def _ucb_scenario(beta=2.5):
    """Build the fixed GP posterior state and the four labelled candidate points.

    Returns a dict with everything the plot and the quiz need, computed
    deterministically so the two always agree. The correct answer is whichever
    candidate maximizes UCB = mu + beta * sigma.
    """
    true_x, true_y = get_objective(scale=_UCB_OBJ_SCALE, offset=_UCB_OBJ_OFFSET,
                                   random_state=_UCB_OBJ_SEED)

    # Sample the (noisy) observations from the hidden true curve
    idx = [np.abs(true_x - xx).argmin() for xx in _UCB_OBS_X]
    obs_x = np.array(_UCB_OBS_X)
    obs_y = true_y[idx] + np.random.default_rng(0).normal(0, _UCB_OBS_NOISE, size=len(_UCB_OBS_X))

    # Fit the GP posterior
    kernel = (RBF(length_scale=_UCB_LENGTH_SCALE, length_scale_bounds="fixed")
              + WhiteKernel(noise_level=_UCB_MODEL_NOISE, noise_level_bounds="fixed"))
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    model.fit(obs_x.reshape(-1, 1), obs_y)

    grid = np.linspace(DOMAIN[0], DOMAIN[1], 1000)
    mu, sigma = model.predict(grid.reshape(-1, 1), return_std=True)

    # Evaluate the posterior mean, std and UCB at each fixed candidate location
    def at(arr, x):
        return arr[np.abs(grid - x).argmin()]

    labels = {}   # letter -> dict(x, mu, sigma, ucb)
    for letter, x in _UCB_CANDIDATES.items():
        m_x, s_x = at(mu, x), at(sigma, x)
        labels[letter] = dict(x=x, mu=m_x, sigma=s_x, ucb=m_x + beta * s_x)

    correct_letter = max(labels, key=lambda L: labels[L]["ucb"])

    return dict(grid=grid, mu=mu, sigma=sigma, beta=beta,
                obs_x=obs_x, obs_y=obs_y, labels=labels, correct_letter=correct_letter)


def plot_ucb_exercise(beta=2.5):
    """Plot the GP posterior state with four labelled candidate learning rates.

    Each candidate is drawn at its posterior mean, so the dots sit at genuinely
    different heights (predicted accuracies).
    """
    sc = _ucb_scenario(beta)
    grid, mu, sigma = sc["grid"], sc["mu"], sc["sigma"]

    plt.figure(figsize=(10, 6))
    plt.plot(grid, mu, "b-", label="Posterior mean $\\mu$")
    plt.fill_between(grid, mu - beta * sigma, mu + beta * sigma,
                     alpha=0.2, color="b", label=f"$\\mu \\pm {beta:g}\\,\\sigma$")
    plt.scatter(sc["obs_x"], sc["obs_y"], c="r", marker="x", s=60, zorder=10,
                label="Observations")

    # Candidate points marked on the x-axis, labelled A-D, with guide lines up to
    # the posterior so each candidate's mean and uncertainty can be read off
    y_dot = 0.47
    for letter, info in sc["labels"].items():
        x = info["x"]
        plt.axvline(x, color="green", linestyle=":", linewidth=1, alpha=0.35, zorder=1)
        plt.scatter([x], [y_dot], c="green", s=150, zorder=11,
                    edgecolors="black", linewidths=0.8)
        plt.annotate(letter, (x, y_dot), textcoords="offset points", xytext=(0, 11),
                     ha="center", fontsize=13, fontweight="bold", color="darkgreen")
    plt.scatter([], [], c="green", s=150, edgecolors="black",
                linewidths=0.8, label="Candidate points")

    plt.xlabel("log10(Learning Rate)")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0.45, 1.05)
    plt.title("GP posterior after 10 observations: which learning rate does UCB query next?")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


def ucb_quiz(beta=2.5):
    """Interactive multiple-choice quiz: which candidate does UCB sample next?"""
    sc = _ucb_scenario(beta)
    correct = sc["correct_letter"]
    uid = "quiz_ucb_next_point"

    options = "".join(f"""
        <label style="display: flex; align-items: center; gap: 10px; background: white; padding: 12px 15px; border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer;">
            <input type="radio" name="ucb_opt" value="{L}" style="accent-color: #2563eb; transform: scale(1.1); cursor: pointer;">
            <span style="font-size: 0.95rem;">Point {L}</span>
        </label>""" for L in ["A", "B", "C", "D"])

    html_content = f"""
    <div id="{uid}" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                max-width: 700px; margin: 20px auto; padding: 25px; background-color: #f8fafc;
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.03); color: #334155;">
        <div style="margin-bottom: 15px;">
            <span style="background: #e0f2fe; color: #0369a1; padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                🧠 Knowledge Check
            </span>
            <h4 style="color: #1e293b; margin-top: 10px; margin-bottom: 4px; font-size: 1.1rem; font-weight: 700;">
                Given the posterior above, which candidate does UCB query next?
            </h4>
            <p style="margin: 0; color: #64748b; font-size: 0.88rem;">Recall: UCB picks the point that maximizes &mu;(x) + &beta;&middot;&sigma;(x).</p>
        </div>
        <div style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 18px;">
            {options}
        </div>
        <button onclick="checkUcbAnswer()" style="background: #2563eb; color: white; border: none; padding: 10px 20px; font-size: 0.9rem; font-weight: 600; border-radius: 8px; cursor: pointer;">
            Submit Answer
        </button>
        <div id="ucb_feedback" style="margin-top: 15px; display: none; padding: 12px 15px; border-radius: 10px; font-size: 0.9rem; line-height: 1.45;"></div>
    </div>

    <script>
    function checkUcbAnswer() {{
        const selected = document.querySelector('input[name="ucb_opt"]:checked');
        const fb = document.getElementById('ucb_feedback');
        if (!selected) {{
            fb.style.display = "block"; fb.style.background = "#fee2e2"; fb.style.color = "#991b1b"; fb.style.border = "1px solid #fca5a5";
            fb.innerHTML = "<strong>⚠️ Selection Missing:</strong> Please pick an option before submitting!";
            return;
        }}
        fb.style.display = "block";
        if (selected.value === "{correct}") {{
            fb.style.background = "#dcfce7"; fb.style.color = "#166534"; fb.style.border = "1px solid #bbf7d0";
            fb.innerHTML = "<strong>✅ Correct!</strong> Point {correct} maximizes &mu; + &beta;&sigma;: it combines a fairly high predicted accuracy with enough remaining uncertainty. The other points are tempting traps: the highest-mean point has almost no uncertainty left (pure exploitation), while the most-uncertain point has a mediocre predicted accuracy (pure exploration). UCB deliberately balances the two.";
        }} else {{
            fb.style.background = "#f1f5f9"; fb.style.color = "#475569"; fb.style.border = "1px solid #cbd5e1";
            fb.innerHTML = "<strong>❌ Try again.</strong> UCB is not the highest-mean point, nor the most-uncertain point on its own; it is the point maximizing &mu; + &beta;&sigma;. Look for a point that is <em>both</em> fairly high and still fairly uncertain.";
        }}
    }}
    </script>
    """
    return HTML(html_content)


# ---------------------------------------------------------------------------
# Tests for the student's get_next_query implementation
# ---------------------------------------------------------------------------

class _FakeGP:
    """A stand-in for a fitted GaussianProcessRegressor with a fully controlled
    posterior, so get_next_query can be tested deterministically without fitting.

    predict(X, return_std=True) returns (mu(X), sigma(X)) for the callables
    supplied at construction, evaluated on the flattened X. This lets us place
    the mean's peak and the uncertainty's peak at known, different locations and
    check exactly which one the acquisition function picks.
    """

    def __init__(self, mu_fn, sigma_fn):
        self._mu_fn = mu_fn
        self._sigma_fn = sigma_fn

    def predict(self, X, return_std=False):
        x = np.asarray(X, dtype=float).ravel()
        mu = self._mu_fn(x)
        if return_std:
            return mu, self._sigma_fn(x)
        return mu


def _reference_next_query(model, domain, beta):
    """Brute-force reference: the grid point (same 1000-point grid get_next_query
    uses) that maximizes UCB = mu + beta * sigma. get_next_query must agree."""
    grid = np.linspace(domain[0], domain[1], 1000)
    mu, sigma = model.predict(grid.reshape(-1, 1), return_std=True)
    return grid[np.argmax(mu + beta * sigma)]


def test_get_next_query(get_next_query, domain=DOMAIN):
    """Known-answer tests for the student's UCB acquisition-maximizer.

    UCB(x) = mu(x) + beta * sigma(x), so get_next_query must return the grid
    point maximizing that. We check the behaviours that fully characterise it:

      1. beta = 0     -> pure exploitation: pick the highest posterior mean.
      2. beta -> inf  -> pure exploration: pick the most uncertain point.
      3. moderate beta -> the true argmax of mu + beta * sigma (brute-forced).
      4. a real fitted GP -> matches the brute-force reference and stays in
         bounds (catches bugs that only a controlled posterior would hide).

    A controlled fake posterior (mean peaks at x = -2, uncertainty peaks at
    x = -4) makes the expected answer exact for cases 1-3. Prints a per-check
    report, then asserts everything passed.
    """
    grid = np.linspace(domain[0], domain[1], 1000)
    spacing = grid[1] - grid[0]
    tol = 1.5 * spacing  # allow the answer to be off by ~one grid step (ties)

    checks = []  # (name, got, expected, ok)

    def record(name, got, expected):
        ok = abs(float(got) - float(expected)) <= tol
        checks.append((name, float(got), float(expected), ok))

    # --- Controlled fake posterior -------------------------------------------
    # mean is a downward parabola peaking at x = -2 (the exploitation target);
    # uncertainty is a bump peaking at x = -4 (the exploration target).
    mu_peak, sigma_peak = -2.0, -4.0
    mu_fn = lambda x: -((x - mu_peak) ** 2)
    sigma_fn = lambda x: np.exp(-((x - sigma_peak) ** 2) / 0.5)
    fake = _FakeGP(mu_fn, sigma_fn)

    # 1. beta = 0 -> maximize the mean -> peak of the parabola (x = -2)
    record("beta=0 exploits (picks highest mean, x=-2)",
           get_next_query(fake, domain, beta=0.0),
           grid[np.argmax(mu_fn(grid))])

    # 2. beta huge -> mean is negligible -> maximize sigma (x = -4)
    record("beta->inf explores (picks most uncertain, x=-4)",
           get_next_query(fake, domain, beta=1e6),
           grid[np.argmax(sigma_fn(grid))])

    # 3. moderate beta -> the genuine trade-off; compare to brute force
    record("beta=2.0 matches argmax(mu + beta*sigma)",
           get_next_query(fake, domain, beta=2.0),
           _reference_next_query(fake, domain, beta=2.0))

    # --- Real fitted GP ------------------------------------------------------
    # Observations clustered on the left half, leaving the right half unexplored.
    obs_x = np.array([-4.8, -4.5, -4.2, -3.9, -3.6])
    _, true_y = get_objective(random_state=7)
    obs_y = np.array([true_y[np.abs(np.linspace(domain[0], domain[1], 1000) - x).argmin()]
                      for x in obs_x])
    kernel = (RBF(length_scale=1.0, length_scale_bounds="fixed")
              + WhiteKernel(noise_level=0.05, noise_level_bounds="fixed"))
    real = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    real.fit(obs_x.reshape(-1, 1), obs_y)

    # 4. matches brute-force reference for a normal beta
    got_real = get_next_query(real, domain, beta=2.5)
    record("real GP matches argmax(mu + beta*sigma)",
           got_real, _reference_next_query(real, domain, beta=2.5))

    # 5. the returned point must lie inside the domain
    in_bounds = domain[0] - tol <= float(got_real) <= domain[1] + tol
    checks.append(("real GP result stays within the domain",
                   float(got_real), float(got_real), in_bounds))

    # --- Report --------------------------------------------------------------
    print("Testing get_next_query …\n")
    for name, got, expected, ok in checks:
        mark = "✅" if ok else "❌"
        detail = f"got x={got:+.3f}" if got == expected else f"got x={got:+.3f}, expected x={expected:+.3f}"
        print(f"  {mark} {name}  ({detail})")

    n_ok = sum(ok for *_, ok in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    if n_ok == len(checks):
        print("🎉 get_next_query looks correct!")
    else:
        print("⚠️  Some checks failed — revisit how you compute and maximize UCB "
              "(mu + beta * sigma), and make sure you return the x value, not its index.")

    assert n_ok == len(checks), f"{len(checks) - n_ok} check(s) failed — see the report above."
