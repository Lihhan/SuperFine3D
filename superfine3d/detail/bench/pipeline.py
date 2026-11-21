"""Driver for pipeline_body.py (spawned workers re-import this file as __mp_main__ and must not re-run the benchmark)."""
if __name__=='__main__':
    import runpy,os;runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),'pipeline_body.py'),run_name='bench_body')
