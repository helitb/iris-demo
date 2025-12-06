"""
IRIS LLM Client

Unified interface for LLM interactions.
Supports:
  - Anthropic SDK (with streaming)
  - aisuite for multi-provider access (batch mode)

API keys are read from environment variables (use .env file).
Model configuration is read from config.yaml.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Generator, Dict, Any, Callable
from dataclasses import dataclass

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

import anthropic

# Optional: aisuite for multi-model support (requires Python 3.10+)
try:
    import aisuite as ai
    AISUITE_AVAILABLE = True
except ImportError:
    AISUITE_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Application configuration loaded from config.yaml."""
    default_model: str
    models: Dict[str, str]
    layer1_max_tokens: int
    layer2_max_tokens: int
    report_max_tokens: int
    sessions_auto_save: bool
    sessions_directory: str
    reports_auto_save: bool
    reports_directory: str
    scenarios_directory: str
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file."""
        if config_path is None:
            # Look for config.yaml in the package directory
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        if not Path(config_path).exists():
            # Return defaults if no config file
            return cls(
                default_model="claude-sonnet-4-20250514",
                models={"Claude Sonnet 4": "claude-sonnet-4-20250514"},
                layer1_max_tokens=8000,
                layer2_max_tokens=4000,
                report_max_tokens=2000,
                sessions_auto_save=True,
                sessions_directory="sessions",
                reports_auto_save=True,
                reports_directory="reports",
                scenarios_directory="scenarios",
            )
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            default_model=data.get("default_model", "claude-sonnet-4-20250514"),
            models=data.get("models", {}),
            layer1_max_tokens=data.get("generation", {}).get("layer1_max_tokens", 8000),
            layer2_max_tokens=data.get("generation", {}).get("layer2_max_tokens", 4000),
            report_max_tokens=data.get("generation", {}).get("report_max_tokens", 2000),
            sessions_auto_save=data.get("sessions", {}).get("auto_save", True),
            sessions_directory=data.get("sessions", {}).get("directory", "sessions"),
            reports_auto_save=data.get("reports", {}).get("auto_save", True),
            reports_directory=data.get("reports", {}).get("directory", "reports"),
            scenarios_directory=data.get("scenarios", {}).get("directory", "scenarios"),
        )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def get_available_models() -> Dict[str, str]:
    """Get dictionary of available models {display_name: model_id}."""
    config = get_config()
    
    # Filter out aisuite models if aisuite not available
    if not AISUITE_AVAILABLE:
        return {
            name: model_id 
            for name, model_id in config.models.items() 
            if ":" not in model_id
        }
    
    return config.models


# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    
    Usage:
        client = LLMClient(model="claude-sonnet-4-20250514")
        
        # Streaming (Anthropic only)
        for chunk in client.stream(system_prompt, user_prompt):
            print(chunk, end="")
        
        # Batch (all providers)
        response = client.complete(system_prompt, user_prompt)
    """
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            model: Model identifier. If contains ":", uses aisuite.
                   Otherwise uses Anthropic SDK directly.
        """
        config = get_config()
        self.model = model or config.default_model
        self.config = config
        
        # Detect provider
        self.use_aisuite = ":" in self.model
        
        if self.use_aisuite:
            if not AISUITE_AVAILABLE:
                raise ImportError(
                    f"Model '{self.model}' requires aisuite (Python 3.10+). "
                    "Use a direct Claude model or upgrade Python."
                )
            self._client = ai.Client()
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    
    @property
    def supports_streaming(self) -> bool:
        """Check if current model supports streaming."""
        return not self.use_aisuite
    
    def complete(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Get a complete response from the LLM.
        
        Works with all providers (batch mode).
        """
        max_tokens = max_tokens or self.config.layer1_max_tokens
        
        if self.use_aisuite:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        else:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
    
    def stream(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        Stream response from the LLM.
        
        Only works with Anthropic SDK (non-aisuite models).
        For aisuite models, falls back to complete() and yields full response.
        """
        max_tokens = max_tokens or self.config.layer1_max_tokens
        
        if self.use_aisuite:
            # Fallback: yield complete response as single chunk
            yield self.complete(system_prompt, user_prompt, max_tokens)
        else:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
    
    def stream_and_parse(
        self,
        system_prompt: str,
        user_prompt: str,
        parse_line: Callable[[str], Optional[Any]],
        max_tokens: Optional[int] = None,
        on_parsed: Optional[Callable[[Any], None]] = None,
    ) -> Generator[Any, None, None]:
        """
        Stream response and parse each line as it arrives.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            parse_line: Function to parse a line, returns None to skip
            max_tokens: Max tokens
            on_parsed: Optional callback for each parsed item
        
        Yields:
            Parsed items from each line
        """
        max_tokens = max_tokens or self.config.layer1_max_tokens
        
        if self.use_aisuite:
            # Batch mode: get full response, then parse all lines
            response = self.complete(system_prompt, user_prompt, max_tokens)
            for line in response.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                parsed = parse_line(line)
                if parsed is not None:
                    if on_parsed:
                        on_parsed(parsed)
                    yield parsed
        else:
            # Streaming mode: parse as we receive
            buffer = ""
            
            for chunk in self.stream(system_prompt, user_prompt, max_tokens):
                buffer += chunk
                
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    parsed = parse_line(line)
                    if parsed is not None:
                        if on_parsed:
                            on_parsed(parsed)
                        yield parsed
            
            # Process remaining buffer
            if buffer.strip():
                parsed = parse_line(buffer.strip())
                if parsed is not None:
                    if on_parsed:
                        on_parsed(parsed)
                    yield parsed
