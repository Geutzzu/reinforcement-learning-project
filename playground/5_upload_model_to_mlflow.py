# # import os
# # import mlflow


# # mlflow.set_tracking_uri("https://mlflow.retelecfc.systems")
# # mlflow.set_experiment("rl-project-maze-sft-baseline-v4")  # NEW name!

# # with mlflow.start_run(run_name="3B-sft-baseline-1epoch"):
# #     mlflow.set_tags({"model_size": "3B", "training_type": "SFT", "epochs": "1"})
# #     mlflow.log_artifacts("/Users/geo/facultate/rl/rl/results/maze_2_sft_quick/3B-1e-baseline-sft/merged-1e", artifact_path="model")



## dowload a model
import os
import mlflow

os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "yoHywV0YGVK9CWOCnttUhKPp"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.retelecfc.systems"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin123secure"

mlflow.set_tracking_uri("https://mlflow.retelecfc.systems")

# Option 1: Download by run ID
run_id = "baabf9eec6d94ad58398aa03161f58e7"

client = mlflow.tracking.MlflowClient()
print("Listing artifacts in run:")
artifacts = client.list_artifacts(run_id)
for artifact in artifacts:
    print(f"  - {artifact.path} (is_dir={artifact.is_dir})")
    if artifact.is_dir:
        sub_artifacts = client.list_artifacts(run_id, artifact.path)
        for sub in sub_artifacts:
            print(f"      - {sub.path}")

# Now try to download
print(f"\nDownloading artifacts...")
local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="model",  
    dst_path="/workspace/rl/models"
)

# Option 2: If you registered the model in Model Registry
# local_path = mlflow.artifacts.download_artifacts( 
#     artifact_uri="models:/model-name/version",
#     dst_path="/path/to/download"
# )

print(f"Downloaded to: {local_path}")