pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out source code'
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image using docker compose'
                sh 'docker compose build'
            }
        }

        stage('Run Detector') {
            steps {
                echo 'Running car detector'
                sh 'docker compose run --rm car-detector'
            }
        }

        stage('Show Metrics') {
            steps {
                echo 'Showing metrics output'
                sh 'cat data/output/metrics.json'
            }
        }

        stage('Validate Metrics') {
            steps {
                echo 'Validating metrics'
                sh '''
                python - <<EOF
import json
import sys

with open("data/output/metrics.json") as f:
    metrics = json.load(f)

precision = metrics.get("precision", 0)
recall = metrics.get("recall", 0)
accuracy = metrics.get("accuracy", 0)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)

if precision < 0.7:
    print("Precision is too low")
    sys.exit(1)

if recall < 0.7:
    print("Recall is too low")
    sys.exit(1)

if accuracy < 0.7:
    print("Accuracy is too low")
    sys.exit(1)

print("Metrics validation passed")
EOF
                '''
            }
        }
    }

    post {
        success {
            echo 'Pipeline finished successfully'
        }

        failure {
            echo 'Pipeline failed'
        }
    }
}