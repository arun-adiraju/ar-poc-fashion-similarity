#!/usr/bin/env bash
ENVIRONMENT_NAME='tensorflow-for-poets-2-env'


echo "Evaluate if virtual environment $ENVIRONMENT_NAME needs to be created ..."
if ! conda info --env | grep -w "$ENVIRONMENT_NAME"; then
    echo "Creating environment: $ENVIRONMENT_NAME"
    conda create -n "$ENVIRONMENT_NAME" python=2.7 pip -y
    echo "Environment: $ENVIRONMENT_NAME created"
fi

if source activate "$ENVIRONMENT_NAME"; then
    echo "Environment set to: $ENVIRONMENT_NAME"

    echo "Installing requirements ..."
    pip install -r requirements.txt
    echo "Requirements installed in: $ENVIRONMENT_NAME"
else
    echo "ERROR: Failed to create and load environment $ENVIRONMENT_NAME"
fi
