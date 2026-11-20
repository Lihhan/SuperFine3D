"""R3-v2 continuation of R2; inference architecture is unchanged."""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import inspect
from pathlib import Path
import refine_train as base
import refine_surface
from r3_geometry_surface import GeometrySurface

base.SOURCE=Path(f'{_E["SF3D_CKPT"]}/stages/R2/denoiser_step0003000.pt')
refine_surface.RefinementSurface=GeometrySurface
source=inspect.getsource(base.training_losses)
old="keys=['keep'] if self.refinement_arm=='R1' else ['keep','detail']"
assert old in source
source=source.replace(old,"keys=['keep','detail','edge','plane']")
exec(compile(source,__file__,'exec'),base.__dict__)
base.sc.AblationTrainer.training_losses=base.training_losses
if __name__=='__main__':base.sc.main()
