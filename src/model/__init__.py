from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.graph_llm import GraphLLM
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LLAMA_2_7B_CHAT = str(REPO_ROOT / "models" / "Llama-2-7b-chat-hf")


load_model = {
    "llm": LLM,
    "inference_llm": LLM,
    "pt_llm": PromptTuningLLM,
    "graph_llm": GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    "7b": LOCAL_LLAMA_2_7B_CHAT,
    "7b_chat": LOCAL_LLAMA_2_7B_CHAT,
    "13b": "meta-llama/Llama-2-13b-hf",
    "13b_chat": "meta-llama/Llama-2-13b-chat-hf",
}
