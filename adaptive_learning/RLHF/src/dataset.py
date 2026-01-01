prompts = [
    "Explain the importance of sleep.",
    "What is machine learning?",
    "Why is the sky blue?",
    "Explain reinforcement learning simply."
]

def reward_fn(text):
    score = 0
    score += text.count("です")
    score += text.count("ます")
    score -= text.count("bad")
    return score




