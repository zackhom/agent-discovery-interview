import os
import json
import requests
from openai import OpenAI

client = OpenAI(api_key="YOUR_KEY")

def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",  # or any chat-capable model you use
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )
    return resp.choices[0].message.content

def call_candidate(candidate_url: str, message: str) -> str | None:
    payload = {
        "messages": [
            {"role": "user", "content": message}
        ]
    }
    try:
        resp = requests.post(candidate_url, json=payload, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Error calling candidate {candidate_url}: {e}")
        return None

    data = resp.json()
    # tweak this based on actual schema
    return data.get("content") or data.get("output") or str(data)

def interview_candidate(candidate_url: str, task: str):
    # 1) Interviewer drafts a question tailored to the task
    sys_interviewer = (
        "You are an interviewing agent. Your goal is to evaluate whether a "
        "candidate agent is suitable for a TASK by asking it questions."
    )

    q_prompt = f"""
        TASK: {task}

        Generate ONE clear, concrete interview question you would ask the candidate agent
        to see if it can handle this task.
        Just output the question text.
        """
    question = call_llm(sys_interviewer, q_prompt)
    
    # 2) candidate answers
    answer = call_candidate(candidate_url, question)
    if answer is None:
        print(f"[WARN] Skipping candidate {candidate_url} (no response).")
        return {
            "task": task,
            "question": question,
            "answer": None,
            "evaluation": {"score": 0, "justification": "Candidate unreachable"},
        }

    # 3) Interviewer evaluates the answer
    eval_prompt = f"""
    You interviewed a candidate agent.

    TASK: {task}

    INTERVIEW QUESTION: {question}

    CANDIDATE ANSWER: {answer}

    On a scale from 1 to 10, how suitable is this agent for the task?
    Return STRICT JSON: {{"score": <int 1-10>, "justification": "<short reason>"}}
    """
    eval_json_str = call_llm(
        "You are a strict JSON-producing judge. No extra text.",
        eval_prompt,
        json_mode=True,
    )
    eval_data = json.loads(eval_json_str)

    print("=== Interview ===")
    print("Task:      ", task)
    print("Question:  ", question)
    print("Answer:    ", answer)
    print("Evaluation:", eval_data)

    return {
        "task": task,
        "question": question,
        "answer": answer,
        "evaluation": eval_data,
    }