GRADING_EXAMPLES = [
    {
        "ground_truth": "Denver Broncos, Denver Broncos, Denver Broncos",
        "student_answer": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Cutting Knowledge Date: December 2023Today Date: 09 May 2025<|eot_id|><|start_header_id|>user<|end_header_id|>Which NFL team represented the AFC at Super Bowl 50?<|eot_id|><|start_header_id|>assistant<|end_header_id|>The Denver Broncos represented the AFC at Super Bowl 50.",
        "grade": 1,
        "justification": "The model's answer directly and correctly states that the Denver Broncos represented the AFC at Super Bowl 50, which is the correct and complete response to the question asked.",
    },
    {
        "ground_truth": "Carolina Panthers, Carolina Panthers, Carolina Panthers",
        "student_answer": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Cutting Knowledge Date: December 2023Today Date: 09 May 2025<|eot_id|><|start_header_id|>user<|end_header_id|>Which NFL team represented the NFC at Super Bowl 50?<|eot_id|><|start_header_id|>assistant<|end_header_id|>The Denver Broncos represented the NFC at Super Bowl 50.",
        "grade": 0,
        "justification": "The Denver Broncos represented the AFC, not the NFC, at Super Bowl 50. The model's answer is factually incorrect and does not provide a semantically similar or correct response to the question.",
    },
]
