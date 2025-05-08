from mlflow.metrics.genai import EvaluationExample, make_genai_metric

from evaluate_llm.grading_examples.similarity_grading_examples import GRADING_EXAMPLES


def create_grading_example(eval_examples: dict) -> EvaluationExample:
    """Create a grading example for the ethical metric."""
    return EvaluationExample(
        input=eval_examples["ground_truth"],
        output=eval_examples["student_answer"],
        score=eval_examples["grade"],
        justification=eval_examples["justification"],
    )


examples = [create_grading_example(example) for example in GRADING_EXAMPLES]

answer_similarity = make_genai_metric(
    name="answer_similarity",
    definition=(
        "The answer similarity metric evaluates if the response is similar"
        " to the defined answer. "
    ),
    grading_prompt=(
        """
        You are a teacher grading a quiz to determine if a student answer is similar to 
        the truth.
    
        You will be given a CORRECT ANSWER and a STUDENT ANSWER with context and the 
        original question. 
        
        Here is the grade criteria to follow:
        (1) Grade the student answers based on whether the student answer
            provides a direct response to the correct answer. 
        (2) Does the student answer in a manner that can be considered the same
            as the correct answer?
        (3) Does the student provide an answer that directly addresses the input?
        
        Answer similarity:
        An answer similarity value of 1 means that the student's answer is semantically
        similar or contain information similar to the correct answer.
        An answer similarity value of 0 means that the student's answer not similar to 
        the correct answer.
                
        Explain your reasoning in a step-by-step manner to ensure your reasoning 
        and conclusion are correct. 
        
        Ignore anything between `<|begin_of_text|>...<|end_header_id|>`
        
        Avoid simply stating the correct answer at the outset. Return 0 or 1 NOT true 
        or false.
        """
    ),
    examples=examples,
    version="v1",
    model="openai:/gpt-4.1-mini-2025-04-14",
    parameters={
        "temperature": 0.0,
    },
    greater_is_better=True,
)
