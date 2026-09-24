from typing import Callable, Dict

AGENT_REGISTRY: Dict[str, Callable[..., str]] = {}

def register_agent(name: str) -> Callable[[Callable[..., str]], Callable[..., str]]:
    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        AGENT_REGISTRY[name] = func
        return func
    return decorator

