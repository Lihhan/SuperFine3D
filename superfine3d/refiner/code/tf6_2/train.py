import argparse
import json
import runpy
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--trellis-root',required=True)
    p.add_argument('--tf1-root',required=True)
    p.add_argument('--config',required=True)
    a,rest=p.parse_known_args()
    cfg=json.loads(Path(a.config).read_text())
    if cfg['trainer']['name']!='TF6Trainer' or cfg['dataset']['args']['lr_mode']!='paired':
        raise ValueError('Wrong TF6 training protocol')
    if cfg['models']['denoiser']['args']['tf6_stage']=='lr_match':
        report=json.loads(Path(cfg['trainer']['args']['hr_interface_report']).read_text())
        args=cfg['models']['denoiser']['args']
        if (report.get('architecture')!='TF6_2_boundary_v1' or report.get('near_budget')!=args['near_budget'] or report.get('boundary_budget')!=args['boundary_budget']):
            raise ValueError('TF6-2 LR training requires its own new HR checkpoint/report')
    sys.path[:0]=[a.tf1_root,a.trellis_root]
    import trellis2.models as models
    import trellis2.trainers as trainers
    from .model import ElasticTF62SLatFlowModel
    from tf6.trainer import TF6Trainer
    models.ElasticTF62SLatFlowModel=ElasticTF62SLatFlowModel
    trainers.TF6Trainer=TF6Trainer
    sys.argv=[sys.argv[0],'--config',a.config,*rest]
    runpy.run_path(str(Path(a.trellis_root)/'train.py'),run_name='__main__')


if __name__=='__main__':main()
