import sqlite3
import random
import time
from datetime import datetime
from typing import Any, List, Optional, Dict, Union

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class LoggingLLMWrapper:
    """
    A wrapper for LangChain LLMs/Runnables that logs all calls to a database.
    Works seamlessly with structured outputs and other LangChain features.
    """
    
    def __init__(
        self,
        base_llm: Any,  # Can be ChatModel or Runnable
        db_path: str,
        source_id: int,
        call_name: str,
        worker_id: str,
        model_name: str = None
    ):
        self._base_llm = base_llm
        self._db_path = db_path
        self._source_id = source_id
        self._call_name = call_name
        self._worker_id = worker_id
        # Try to extract model name from base_llm if not provided
        if model_name is None:
            if hasattr(base_llm, 'model_name'):
                model_name = base_llm.model_name
            elif hasattr(base_llm, 'model'):
                model_name = base_llm.model
            else:
                model_name = "unknown"
        self._model_name = model_name
    
    def invoke(
        self,
        messages: Union[List[BaseMessage], Any],
        **kwargs: Any
    ) -> Any:
        """Invoke and log the response."""
        # Call base LLM/Runnable
        result = self._base_llm.invoke(messages, **kwargs)
        
        # Log to database with retry on locks
        self._log_call_with_retry(messages, result)
        
        return result
    
    async def ainvoke(
        self,
        messages: Union[List[BaseMessage], Any],
        **kwargs: Any
    ) -> Any:
        """Async invoke and log the response."""
        # Call base LLM/Runnable
        result = await self._base_llm.ainvoke(messages, **kwargs)
        
        # Log to database with retry on locks
        self._log_call_with_retry(messages, result)
        
        return result
    
    def _log_call(self, messages: Union[List[BaseMessage], Any], result: Any) -> None:
        """Log the LLM call to the database."""
        # Extract prompts
        system_prompt = ""
        human_prompt = ""
        
        if isinstance(messages, list):
            for msg in messages:
                if hasattr(msg, 'type'):
                    if msg.type == "system":
                        system_prompt = msg.content
                    elif msg.type == "human":
                        human_prompt = msg.content
        
        # Extract response - handle different result types
        assistant_response = ""
        # Check for AIMessage/BaseMessage first (before BaseModel check)
        if hasattr(result, 'content') and hasattr(result, 'type'):
            # AIMessage or similar - extract just the content
            assistant_response = result.content
        elif isinstance(result, BaseModel):
            # Structured output - convert to JSON string
            assistant_response = result.model_dump_json()
        elif hasattr(result, 'generations'):
            # ChatResult
            if result.generations and len(result.generations) > 0:
                if len(result.generations[0]) > 0:
                    assistant_response = result.generations[0][0].text
        else:
            # Fallback - convert to string
            assistant_response = str(result)
        
        # Create thread-local connection with proper settings for concurrency
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.isolation_level = None  # Autocommit mode
        
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO distill_llm_calls 
                (source_id, call_name, system_prompt, human_prompt, assistant_response, created_at, worker_id, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._source_id,
                    self._call_name,
                    system_prompt,
                    human_prompt,
                    assistant_response,
                    datetime.now().isoformat(),
                    self._worker_id,
                    self._model_name
                )
            )
            # No need for commit in autocommit mode
        finally:
            conn.close()
    
    def _log_call_with_retry(self, messages: Union[List[BaseMessage], Any], result: Any) -> None:
        """Log with retry on database locks."""
        attempt = 0
        base_delay = 0.01  # Start with 10ms
        max_delay = 2.0  # Cap at 2 seconds
        
        while True:
            try:
                self._log_call(messages, result)
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    # Exponential backoff with jitter, capped at max_delay
                    delay = min(
                        base_delay * (2**attempt), max_delay
                    ) + random.uniform(0, 0.05)
                    # Note: Not printing here to avoid spam in multi-threaded context
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise
    
    def with_structured_output(self, schema: Any, **kwargs: Any) -> "LoggingLLMWrapper":
        """Wrap the structured output chain to log the parsed results."""
        # Get the structured output chain from base LLM
        structured_chain = self._base_llm.with_structured_output(schema, **kwargs)
        # Wrap it to log the structured output (not ideal but ensures logging works)
        return LoggingLLMWrapper(
            structured_chain,
            self._db_path,
            self._source_id,
            self._call_name,
            self._worker_id,
            self._model_name
        )
    
    
    # Delegate all other attributes and methods to the base LLM
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the base LLM."""
        return getattr(self._base_llm, name)