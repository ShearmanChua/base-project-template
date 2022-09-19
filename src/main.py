import argparse
import os
# from clearml import Task
from aiplatform.config import cfg as aip_cfg

if __name__ == '__main__':

    # push to remote code here
    task = None

    # task = Task.init(project_name=aip_cfg.clearml.project_name,
    # task_name=aip_cfg.clearml.task_name,
    # output_uri=os.path.join(aip_cfg.s3.endpoint_url,aip_cfg.s3.model_artifact_path),
    # reuse_last_task_id=False)
    # # GIT_SSL_NO_VERIFY=true MUST be set in order for worker to run. If repo is not public, user and password for git has to be included
    # task.set_base_docker('{} --env GIT_SSL_NO_VERIFY=true --env TRAINS_AGENT_GIT_USER={} --env TRAINS_AGENT_GIT_PASS={}'.format(aip_cfg.docker.base_image,aip_cfg.git.id,aip_cfg.git.key))
    # task.execute_remotely(queue_name=aip_cfg.clearml.queue,exit_process=True)

    # #actual code here
    import dask.dataframe as dd

    from model import experiment
    from model.config import cfg
    from pipeline import pipeline
    from pipeline.config import cfg as pipeline_cfg
    from aiplatform import s3utility

    parser = argparse.ArgumentParser()
    parser = experiment.Experiment.add_experiment_args(parser)
    # parser = pipeline.AnnotatedFlightDataPipeline.add_pipeline_args(parser)
    args = parser.parse_args()
    # model_config_dict = task.connect_configuration(cfg,name='Model Training Parameters')
    # pipeline_config_dict = task.connect_configuration(pipeline_cfg,name='Data Pipeline Parameters')

    # # local: write data via pipeline
    # annotated_data_pipe = pipeline.AnnotatedFlightDataPipeline(args.pipeline_data_profiles, pipeline_cfg.data.groups, pipeline_cfg.source.db_uri,
    #                                                            args.pipeline_path_h5, args.pipeline_data_valid_size, args.pipeline_data_max_total_length, args.pipeline_data_min_segment_length, args.pipeline_seed)
    # annotated_data_pipe.write_data_to_parquet(args.pipeline_data_npartitions)
    # # # remote: download data via s3
    # print('using remote data source...')
    # s3_utils = s3utility.S3Utils(aip_cfg.s3.bucket,aip_cfg.s3.s3_path)
    # print('sucessfully connected to s3...')
    # s3_utils.s3_download_folder('train','/src/data/train')
    # s3_utils.s3_download_folder('valid','/src/data/valid')

    exp = experiment.Experiment(args, task)
    # exp.run_experiment()
    exp.create_torchscript_model('class_model_v2.ckpt')
    # exp.create_torchscript_cpu_model('id_model4.ckpt')
