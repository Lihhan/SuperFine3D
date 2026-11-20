import argparse,sys,runpy,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser(add_help=False)
    for k in ('trellis-root','tf1-root','config'):p.add_argument('--'+k,required=True)
    a,rest=p.parse_known_args();cfg=json.loads(Path(a.config).read_text())
    report=json.loads(Path(cfg['trainer']['args']['hr_interface_report']).read_text())
    assert report['architecture']=='TF6_2_boundary_v1'
    assert cfg['dataset']['args']['lr_mode']=='paired'
    sys.path[:0]=[a.tf1_root,a.trellis_root]
    import trellis2.models as models
    import trellis2.trainers as trainers
    from .model import ElasticTF8SLatFlowModel
    from tf6.trainer import TF6Trainer
    models.ElasticTF8SLatFlowModel=ElasticTF8SLatFlowModel;trainers.TF6Trainer=TF6Trainer
    sys.argv=[sys.argv[0],'--config',a.config,*rest]
    runpy.run_path(str(Path(a.trellis_root)/'train.py'),run_name='__main__')
if __name__=='__main__':main()
