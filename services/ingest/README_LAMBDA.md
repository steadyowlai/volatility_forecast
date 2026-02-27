# Ingest Service - Lambda Migration

This document explains how to deploy the ingest service to AWS Lambda.

## Overview

The ingest service has been updated to work with both:
- **Local filesystem** (development with Docker)
- **S3 storage** (production with Lambda)

Storage backend is automatically detected based on `AWS_EXECUTION_ENV` environment variable.

---

## Architecture

### Local (Current)
```
Docker → Local filesystem → data/
```

### Lambda (Target)
```
Lambda → S3 → s3://volatility-forecast/
```

---

## Prerequisites

1. **AWS Account** with permissions to:
   - Create Lambda functions
   - Create/push to ECR repositories
   - Read/write to S3 bucket

2. **AWS CLI** installed and configured:
   ```bash
   aws configure
   ```

3. **S3 Bucket** already created: `volatility-forecast`

4. **Docker** installed locally

---

## Deployment Steps

### Step 1: Configure AWS Credentials

Add to your `.env` file:
```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET=volatility-forecast
```

### Step 2: Test Locally with S3

Before deploying to Lambda, test with your S3 bucket:

```bash
# Source your AWS credentials
source .env

# Build the regular Docker image
docker-compose build ingest

# Run with S3 backend
docker-compose run --rm \
  -e AWS_EXECUTION_ENV=AWS_Lambda \
  -e S3_BUCKET=volatility-forecast \
  -e AWS_REGION=us-east-1 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  ingest
```

This simulates Lambda environment but runs locally.

### Step 3: Create ECR Repository

```bash
# Create repository
aws ecr create-repository \
  --repository-name vf-ingest \
  --region us-east-1

# Note the repository URI (e.g., 123456789.dkr.ecr.us-east-1.amazonaws.com/vf-ingest)
```

### Step 4: Build and Push Lambda Image

```bash
# Get your AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/vf-ingest"

# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

# Build Lambda image
docker build \
  -t vf-ingest:lambda \
  -f services/ingest/Dockerfile.lambda \
  .

# Tag for ECR
docker tag vf-ingest:lambda ${ECR_REPO}:latest

# Push to ECR
docker push ${ECR_REPO}:latest
```

### Step 5: Create Lambda Function

```bash
# Create IAM role for Lambda (if not exists)
aws iam create-role \
  --role-name lambda-vf-ingest-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach S3 permissions
aws iam put-role-policy \
  --role-name lambda-vf-ingest-role \
  --policy-name s3-access \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::volatility-forecast",
        "arn:aws:s3:::volatility-forecast/*"
      ]
    }]
  }'

# Attach CloudWatch Logs permissions
aws iam attach-role-policy \
  --role-name lambda-vf-ingest-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Get role ARN
ROLE_ARN=$(aws iam get-role --role-name lambda-vf-ingest-role --query 'Role.Arn' --output text)

# Create Lambda function
aws lambda create-function \
  --function-name vf-ingest \
  --package-type Image \
  --code ImageUri=${ECR_REPO}:latest \
  --role ${ROLE_ARN} \
  --timeout 300 \
  --memory-size 1024 \
  --environment Variables="{S3_BUCKET=volatility-forecast,START_DATE=2010-01-01}" \
  --region us-east-1
```

### Step 6: Test Lambda Function

```bash
# Invoke Lambda
aws lambda invoke \
  --function-name vf-ingest \
  --region us-east-1 \
  --log-type Tail \
  --query 'LogResult' \
  --output text \
  response.json | base64 --decode

# Check response
cat response.json
```

### Step 7: Set Up Scheduled Trigger

```bash
# Create EventBridge rule (daily at 10 PM UTC = 5 PM EST)
aws events put-rule \
  --name vf-ingest-daily \
  --schedule-expression "cron(0 22 ? * MON-FRI *)" \
  --state ENABLED \
  --description "Daily market data ingestion after US market close"

# Add Lambda permission for EventBridge
aws lambda add-permission \
  --function-name vf-ingest \
  --statement-id vf-ingest-daily-trigger \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn $(aws events describe-rule --name vf-ingest-daily --query 'Arn' --output text)

# Add Lambda as target
aws events put-targets \
  --rule vf-ingest-daily \
  --targets "Id"="1","Arn"="$(aws lambda get-function --function-name vf-ingest --query 'Configuration.FunctionArn' --output text)"
```

---

## Updating the Function

When you make code changes:

```bash
# Rebuild image
docker build -t vf-ingest:lambda -f services/ingest/Dockerfile.lambda .

# Tag and push
docker tag vf-ingest:lambda ${ECR_REPO}:latest
docker push ${ECR_REPO}:latest

# Update Lambda function
aws lambda update-function-code \
  --function-name vf-ingest \
  --image-uri ${ECR_REPO}:latest
```

---

## Monitoring

### View Logs
```bash
# Get latest log stream
aws logs tail /aws/lambda/vf-ingest --follow
```

### Check S3 Updates
```bash
# List latest partitions
aws s3 ls s3://volatility-forecast/data/curated.market/ --recursive | tail -10

# Check manifest
aws s3 cp s3://volatility-forecast/data/curated.market/_manifest.json - | jq .
```

---

## Cost Estimation

- **Lambda**: ~30 invocations/month × 2 min × 1024 MB = **$0.10-0.50/month**
- **ECR**: ~500 MB storage = **$0.05/month**
- **S3**: ~500 MB storage + requests = **$0.05-0.20/month**

**Total: ~$0.20-1.00/month** (essentially free)

---

## Troubleshooting

### Lambda Timeout
- Increase timeout: `--timeout 600` (10 minutes max)
- Check yfinance download speed

### Permission Denied
- Verify IAM role has S3 permissions
- Check bucket name matches

### Image Too Large
- Lambda limit: 10 GB
- Current image: ~500 MB
- No issues expected

---

## Rollback to Local

To switch back to local filesystem:

1. Remove `AWS_EXECUTION_ENV` environment variable
2. Use original Docker Compose setup
3. Data stays in local `data/` folder

No code changes needed - storage layer handles both!
