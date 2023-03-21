"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import json
import os
import re

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    Processor,
    ScriptProcessor,
)
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    Join,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def read_flow(filename):
    with open(filename, "r") as f:
        return json.loads(f.read())


def filter_string(s):
    return "-".join(re.findall(r"[a-zA-Z0-9!.*'()_-]+", s))


def get_destination_node_output_names(flow):
    output_names = []
    for node in flow["nodes"]:
        if node["type"] == "DESTINATION":
            output_names.append(
                (filter_string(node["name"]), f"{node['node_id']}.{node['outputs'][0]['name']}")
            )
    return output_names


def create_processing_job_outputs(flow, s3_output_base_path):
    output_names = get_destination_node_output_names(flow)
    processing_outputs = []
    for dataset_name, output_name in output_names:
        processing_outputs.append(
            ProcessingOutput(
                output_name=output_name,
                source=f"/opt/ml/processing/output/{dataset_name}",
                destination=os.path.join(s3_output_base_path, dataset_name),
                s3_upload_mode="EndOfJob"
            )
        )
    return processing_outputs, output_names


def upload_flow(filename, bucket, key, region):
    boto_session = boto3.Session(region_name=region)
    s3_client = boto_session.client("s3")
    s3_client.upload_file(filename, bucket, key, ExtraArgs={"ServerSideEncryption": "aws:kms"})
    return f"s3://{bucket}/{key}"


def create_parameter_override_args(parameter_overrides):
    """Create PJ args from parameter overrides.
    
    Args:
        parameter_overrides: a mapping of parameter name to Pipeline Parameter object

    Returns: list of `--parameter-override` container arguments
    """
    return [
        Join(on="", values=[f"--parameter-override '{{\"{name}\": \"", value, "\"}'"])
        for name, value in parameter_overrides.items()
    ]


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    processing_instance_type="ml.m5.4xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://sagemaker-servicecatalog-seedcode-{region}/dataset/abalone-dataset.csv",
    )
    refit_flow = ParameterBoolean(name="RefitFlow", default_value=False)

    # processing step for feature engineering

    # read the flow file
    flow_file_name = "preprocess.flow"
    flow_file = os.path.join(BASE_DIR, flow_file_name)
    flow = read_flow(flow_file)

    # create processing outputs for each destination in the flow
    s3_output_base_path = f"s3://{sagemaker_session.default_bucket()}/data-wrangler-outputs/abalone/"
    processing_job_outputs, output_names = create_processing_job_outputs(flow, s3_output_base_path)

    # get output names for train, validation, test sets
    train_output_name = [output_name[1] for output_name in output_names if "train" in output_name[0].lower()][0]
    validation_output_name = [output_name[1] for output_name in output_names if "validation" in output_name[0].lower()][0]
    test_output_name = [output_name[1] for output_name in output_names if "test" in output_name[0].lower()][0]

    # create processing input for flow file
    flow_s3_uri = upload_flow(flow_file, sagemaker_session.default_bucket(), "data-wrangler-flows/preprocess.flow", region)

    flow_input = ProcessingInput(
        source=flow_s3_uri,
        destination="/opt/ml/processing/flow",
        input_name="flow",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type="FullyReplicated",
    )

    # Latest Data Wrangler Container URI
    container_uri = "174368400705.dkr.ecr.us-west-2.amazonaws.com/sagemaker-data-wrangler-container:2.x"

    # Size in GB of the EBS volume to use for storing data during processing.
    volume_size_in_gb = 30

    # KMS key for per object encryption; default is None.
    kms_key = None

    # List of tags to be passed to the processing job.
    user_tags = []

    # Content type for each output.
    output_content_type = "CSV"

    # Delimiter to use for the output if the output content type is CSV. Uncomment to set.
    # delimiter = ","

    # Compression to use for the output. Uncomment to set.
    # compression = "gzip"

    # Configuration for partitioning the output. Uncomment to set.
    # "num_partition" sets the number of partitions/files written in the output.
    # "partition_by" sets the column names to partition the output by.
    # partition_config = {
    #     "num_partitions": 1,
    #     "partition_by": ["column_name_1", "column_name_2"],
    # }

    # Output configuration used as processing job container arguments. Only applies when writing to S3.
    # Uncomment to set additional configurations.
    output_configs = [{
        output_name[1]: {
            "content_type": output_content_type,
            # "delimiter": delimiter,
            # "compression": compression,
            # "partition_config": partition_config,
        }
    } for output_name in output_names]

    # Refit configuration determines whether Data Wrangler refits the trainable parameters on the entire dataset.
    # When True, the processing job relearns the parameters and outputs a new flow file.
    # You can specify the name of the output flow file under 'output_flow'.
    # Note: There are length constraints on the container arguments (max 256 characters).
    # refit_trained_params = {"refit": refit_flow}

    # Overridable parameters. Set new values here to change the behavior of the processing job.
    parameter_overrides = {
        'InputDataUrl': input_data,
    }

    parameter_override_args = create_parameter_override_args(parameter_overrides)

    processor = Processor(
        role=role,
        image_uri=container_uri,
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        volume_size_in_gb=volume_size_in_gb,
        sagemaker_session=sagemaker_session,
        output_kms_key=kms_key,
        tags=user_tags
    )

    step_data_wrangler = ProcessingStep(
        name="DataWranglerProcessingStep",
        processor=processor,
        inputs=[flow_input],
        outputs=processing_job_outputs,
        job_arguments=[f"--output-config '{json.dumps(output_config)}'" for output_config in output_configs]
            + [Join(on="", values=[f"--refit-trained-params '{{\"refit\": ", refit_flow, f", \"output_flow\": \"{flow_file_name}\"}}'"])]
            + create_parameter_override_args(parameter_overrides)
    )

    # training step for generating model artifacts
    algo_content_type = "text/csv"
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AbaloneTrain"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.5-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/abalone-train",
        sagemaker_session=pipeline_session,
        role=role,
    )
    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
    )
    step_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=Join(
                    on="/",
                    values=[
                        step_data_wrangler.properties.ProcessingOutputConfig.Outputs[train_output_name].S3Output.S3Uri,
                        step_data_wrangler.properties.ProcessingJobName,
                    ]
                ),
                content_type=algo_content_type,
            ),
            "validation": TrainingInput(
                s3_data=Join(
                    on="/",
                    values=[
                        step_data_wrangler.properties.ProcessingOutputConfig.Outputs[validation_output_name].S3Output.S3Uri,
                        step_data_wrangler.properties.ProcessingJobName,
                    ]
                ),
                content_type=algo_content_type,
            ),
        },
    )
    step_train = TrainingStep(
        name="TrainAbaloneModel",
        step_args=step_args,
    )

    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-abalone-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=Join(
                    on="/",
                    values=[
                        step_data_wrangler.properties.ProcessingOutputConfig.Outputs[test_output_name].S3Output.S3Uri,
                        step_data_wrangler.properties.ProcessingJobName,
                    ]
                ),
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )
    evaluation_report = PropertyFile(
        name="AbaloneEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateAbaloneModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_register = ModelStep(
        name="RegisterAbaloneModel",
        step_args=step_args,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value"
        ),
        right=6.0,
    )
    step_cond = ConditionStep(
        name="CheckMSEAbaloneEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
            refit_flow,
        ],
        steps=[step_data_wrangler, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
