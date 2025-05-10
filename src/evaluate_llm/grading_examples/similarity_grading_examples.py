GRADING_EXAMPLES = [
    {
        "ground_truth": "add(add(add(12723, 14215), 15794), 442)",
        "student_answer": "add(add(12723, 14215), 15794) = 12723 + 14215 + 15794",
        "grade": 0,
        "justification": "The model's output does not provide a direct or semantically similar response to the input expression; it incorrectly equates a nested function call to a sum of numbers without completing the full addition or addressing the entire input expression.",
    },
    {
        "ground_truth": "add(add(add(11595, 2330), 4910), 6535)",
        "student_answer": "add(add(12723, 14215), 11595)",
        "grade": 0,
        "justification": "The model's output does not provide a direct or semantically similar response to the input expression. The input is a nested addition of specific numbers, but the output shows a different and incorrect nested addition expression that does not correspond to the input or its correct evaluation.",
    },
    {
        "ground_truth": "divide(subtract(4126, 4197), 4197)",
        "student_answer": "add(add(4126, 4197), -4261) / (-4261 + 4197)",
        "grade": 0,
        "justification": "The model's output does not provide a direct or semantically similar response to the input expression. The input is a nested addition of specific numbers, but the output shows a different and incorrect nested addition expression that does not correspond to the input or its correct evaluation.",
    },
    {
        "ground_truth": "add(add(add(12723, 14215), 15794), 442)",
        "student_answer": "add(add(add(12723, 14215), 15794), 442)",
        "grade": 1,
        "justification": "The model's output exactly matches the input expression, providing a direct and semantically identical response that fully addresses the task of returning the expression as is.",
    },
]
