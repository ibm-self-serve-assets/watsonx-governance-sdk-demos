# RAG Agent - watsonx.governance Demo

The `evaluate_metrics` notebook demonstrates how to leverage the watsonx.governance sdk to evaluate RAG metrics.

## Getting Started

### Prerequisites

1. Python 3.11 or higher
2. Jupyter environment
3. [uv](https://github.com/astral-sh/uv) package manager
4. watsonx.governance
5. [ibmcloud CLI](<https://cloud.ibm.com/docs/cli?topic=cli-getting-started>)

### Setup

1. Clone the repository

    ```bash
    git clone git@github.com:ibm-self-serve-assets/watsonx-governance-sdk-demos.git
    ```

2. Change directory into `watsonx-governance-sdk-demos/evaluate-metrics`

    ```bash
    cd watsonx-governance-sdk-demos/evaluate-metrics
    ```

3. Create a python virtual environment and install dependencies

    ```bash
    python3 -m venv virtual-env
    source virtual-env/bin/activate
    uv pip install -e .
    ```

4. Copy env file to .env

    ```bash
    cp env .env
    ```

5. Configure parameters in .env
    1. **watsonx.ai credentials**:
        * `WATSONX_APIKEY`: [IBM Cloud API Key](<https://cloud.ibm.com/iam/apikeys>)
        * `WATSONX_PROJECT_ID`: watsonx.ai project's Manage tab (Project -> Manage -> General -> Details)
        * `WATSONX_URL` (Typically "<https://us-south.ml.cloud.ibm.com>")
    2. **watsonx.governance credentials**:
        * `SERVICE_INSTANCE_ID`:
          * Login to your IBM Cloud account using the cli: `ibmcloud login --sso`
          * Select your account
          * Target the resource group that contains your watsonx.governance instance: `ibmcloud target -g <resource_group_name>`
          * Get the service instance details: `ibmcloud resource service-instance '<watsonx.governance instance name'`
          * The `GUID` in the returned information is your `SERVICE_INSTANCE_ID`

6. Run the notebook
