resource "aws_s3_bucket" "dvc" {
  bucket = "${var.app_name}-dvc"
}

resource "aws_s3_bucket_public_access_block" "public_access" {
  bucket = aws_s3_bucket.dvc.id

  block_public_acls       = false
  ignore_public_acls      = false
  restrict_public_buckets = false
  block_public_policy     = false
}