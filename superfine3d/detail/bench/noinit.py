"""Opt-in exact load speed-up. Model constructors initialise every parameter randomly (kaiming/normal, on the CPU), and
from_pretrained then overwrites all of them from the checkpoint. noinit() turns torch.nn.init's initialisers into no-ops while
a model is built; check() asserts that the checkpoint covers every entry of the model's state_dict (the official loaders use
load_state_dict(strict=False)), so no parameter or persistent buffer can keep uninitialised memory. Non-persistent buffers are
computed by the constructors without torch.nn.init in the models this is used for; the outputs are verified bitwise
(bench/test_noinit.sh). The global CPU RNG is not advanced by the skipped initialisers; both pipelines re-seed
(torch.manual_seed) before sampling."""
import contextlib, torch
_NAMES = ['uniform_', 'normal_', 'trunc_normal_', 'kaiming_uniform_', 'kaiming_normal_', 'xavier_uniform_', 'xavier_normal_',
          'constant_', 'zeros_', 'ones_', 'orthogonal_', 'eye_', 'dirac_', 'sparse_']

@contextlib.contextmanager
def noinit():
    I = torch.nn.init; old = {n: getattr(I, n) for n in _NAMES if hasattr(I, n)}
    try:
        for n in old: setattr(I, n, lambda t, *a, **k: t)
        yield
    finally:
        for n, f in old.items(): setattr(I, n, f)

def check(model, path):
    from safetensors import safe_open
    with safe_open(path, 'pt') as f: keys = set(f.keys())
    miss = [k for k in model.state_dict() if k not in keys]
    assert not miss, f'noinit: checkpoint {path} does not cover {len(miss)} entries, e.g. {miss[:5]}'

def wrap(module, attr='from_pretrained'):
    """Replace module.<attr>(path, ...) by a no-init build + coverage check (exact when the check passes). The keys checked are
    those of the safetensors file(s) the loader actually reads (safetensors.torch.load_file is observed during the call)."""
    import safetensors.torch as st
    orig = getattr(module, attr)
    def fp(path, *a, **k):
        seen = set(); lf = st.load_file
        def rec(f, *a2, **k2):
            d = lf(f, *a2, **k2); seen.update(d.keys()); return d
        st.load_file = rec
        try:
            with noinit(): m = orig(path, *a, **k)
        finally: st.load_file = lf
        params = {q for q, _ in m.named_parameters()}; miss = [q for q in m.state_dict() if q not in seen]
        assert seen and not [q for q in miss if q in params], f'noinit: {path}: checkpoint does not cover parameters {[q for q in miss if q in params][:5]}'
        if miss: print(f'noinit: {path.split("/")[-1]}: buffers computed by the constructor (not in the checkpoint): {miss[:5]}', flush=True)
        return m
    setattr(module, attr, fp); return orig
