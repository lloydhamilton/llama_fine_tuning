terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">=5.54.1"
    }
    random = {
      source  = "hashicorp/random"
      version = "3.6.2"
    }
    tls = {
      source = "hashicorp/tls"
      version = "4.0.6"
    }
  }

  backend "s3" {
    bucket  = "demo-tf-state-7813"
    key     = "llm-fine-tuning"
    region  = "eu-west-1"
    encrypt = true
  }
}

provider "aws" {
  default_tags {
    tags = {
      "managed-by"  = "terraform"
      "created-by"  = "lloyd_h"
      "app-name"    = var.app_name
    }
  }
  region = "eu-west-1"
}