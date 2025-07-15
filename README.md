# PyVenice

[![Python Version](https://img.shields.io/pypi/pyversions/pyvenice)](https://pypi.org/project/pyvenice/)
[![PyPI Version](https://img.shields.io/pypi/v/pyvenice)](https://pypi.org/project/pyvenice/)
[![License](https://img.shields.io/github/license/TheLustriVA/PyVenice)](https://github.com/TheLustriVA/PyVenice/blob/main/LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/TheLustriVA/PyVenice/tests.yml?branch=main&label=tests)](https://github.com/TheLustriVA/PyVenice/actions)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/charliermarsh/ruff)
[![Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen)](https://github.com/TheLustriVA/PyVenice)
[![Type Safety](https://img.shields.io/badge/type%20safety-pydantic-blue)](https://pydantic.dev)

**Enterprise-grade Python client for Venice.ai with intelligent parameter validation, comprehensive async support, and production-ready reliability.**

PyVenice is a sophisticated client library that goes beyond simple API wrapping. It features a decorator-based validation system, automatic model compatibility checking, comprehensive type safety, and professional error handling—making it the definitive choice for production Venice.ai integrations.

![PyVenice Banner](https://i.imgur.com/OsAkjoQ.png)

## ✨ Enterprise Features

### 🏗️ **Intelligent Architecture**
- **Decorator-Based Validation** - Automatic parameter filtering based on model capabilities
- **Model Compatibility Engine** - Dynamic capability checking prevents API errors
- **Type-Safe Throughout** - Comprehensive Pydantic models with runtime validation
- **Professional Error Handling** - Structured exceptions with context and recovery

### 🚀 **Production-Ready**
- **Complete API Coverage** - All 8 Venice.ai endpoint classes implemented
- **Async-First Design** - Full async/await support with proper context management
- **Streaming Excellence** - Real-time streaming for chat, audio, and image generation
- **Connection Management** - Automatic retries, timeouts, and connection pooling

### 🔐 **Enterprise Security**
- **API Key Management** - Secure key rotation and usage monitoring
- **Rate Limit Handling** - Intelligent backoff and quota management
- **Audit Logging** - Comprehensive request/response tracking
- **Zero Data Leakage** - Credentials never logged or exposed

### 📊 **Monitoring & Analytics**
- **Usage Tracking** - Real-time cost and consumption monitoring
- **Performance Metrics** - Request latency and success rate tracking
- **Billing Integration** - Detailed cost analysis and budget alerts
- **Health Checks** - API status monitoring and failover support

Looking for Venice.ai access? Consider using my referreal code [https://venice.ai/chat?ref=0Y4qyR](https://venice.ai/chat?ref=0Y4qyR) or register at [venice.ai](https://venice.ai)

## 📦 Installation

**Requirements:** Python 3.12+

```bash
pip install pyvenice
```

**Development Installation:**
```bash
pip install pyvenice[dev]
```

**Test Dependencies:**
```bash
pip install pyvenice[test]
```

### Installation Options

| Option | Use Case | Dependencies |
|--------|----------|-------------|
| `pip install pyvenice` | Production use | Core dependencies only |
| `pip install pyvenice[test]` | Testing | Includes pytest, coverage tools |
| `pip install pyvenice[dev]` | Development | Full development environment |

### Troubleshooting Installation

If you encounter build errors (especially on ARM64 Android/Termux), see our [Installation Troubleshooting Guide](docs/INSTALLATION_TROUBLESHOOTING.md).

## 🚀 Quick Start

```python
from pyvenice import VeniceClient, ChatCompletion

# Initialize client (uses VENICE_API_KEY env var by default)
client = VeniceClient()

# Create a chat completion
chat = ChatCompletion(client)
response = chat.create(
    model="venice-uncensored",
    messages=[{"role": "user", "content": "Hello, Venice!"}]
)

print(response.choices[0].message.content)
```

## 🏗️ Architecture Overview

PyVenice's sophisticated architecture sets it apart from simple API wrappers through intelligent automation and enterprise-grade reliability:

### Decorator-Based Validation System

The core innovation is the `@validate_model_params` decorator that automatically filters parameters based on model capabilities:

```python
# This decorator prevents API errors by removing unsupported parameters
@validate_model_params
def create_chat_completion(self, model: str, **kwargs):
    # Parameters like 'parallel_tool_calls' are automatically removed
    # if the model doesn't support them
    return self._client.post("/chat/completions", json=kwargs)
```

### Model Compatibility Engine

Dynamic capability checking prevents runtime errors:

```python
# Automatic capability detection
capabilities = client.models.get_capabilities("venice-uncensored")
if capabilities.supportsFunctionCalling:
    # Function calling is safely enabled
    response = client.chat.create(
        model="venice-uncensored",
        tools=[{"type": "function", ...}]
    )
```

### Type Safety Throughout

Comprehensive Pydantic models ensure type safety:

```python
from pyvenice.models import ChatCompletionRequest, ChatCompletionResponse

# Request validation
request = ChatCompletionRequest(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)

# Response validation
response: ChatCompletionResponse = client.chat.create(**request.model_dump())
```

### Professional Error Handling

Structured exceptions with context and recovery guidance:

```python
try:
    response = client.chat.create(model="nonexistent-model", messages=[...])
except ModelNotFoundError as e:
    # Specific error with available models
    print(f"Model not found: {e.model}")
    print(f"Available models: {e.available_models}")
except RateLimitError as e:
    # Intelligent backoff suggestion
    print(f"Rate limited. Retry after: {e.retry_after}")
```

## 💡 Complete API Reference

### 💬 Chat Completions

#### Basic Chat Completion

```python
from pyvenice import VeniceClient, ChatCompletion

client = VeniceClient(api_key="your-api-key")
chat = ChatCompletion(client)

response = chat.create(
    model="venice-uncensored",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain async/await in Python"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

#### Streaming Chat Responses

```python
# Real-time streaming
async def stream_chat():
    async for chunk in chat.create_streaming(
        model="venice-coder",
        messages=[{"role": "user", "content": "Write a Python async function"}],
        temperature=0.7
    ):
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

# Synchronous streaming
for chunk in chat.create_streaming(
    model="venice-coder",
    messages=[{"role": "user", "content": "Explain Python generators"}],
    stream=True
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

#### Function Calling with Tool Use

```python
# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = chat.create(
    model="venice-uncensored",
    messages=[{"role": "user", "content": "What's the weather like in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        if tool_call.function.name == "get_weather":
            # Execute your function
            weather_data = get_weather(**tool_call.function.arguments)
            # Send result back
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": weather_data
            })
```

#### Web Search Integration

```python
# Chat with web search
response = chat.create_with_web_search(
    model="venice-uncensored",
    messages=[{"role": "user", "content": "Latest developments in AI safety"}],
    search_mode="auto",  # or "always", "never"
    max_search_results=5
)

print(response.choices[0].message.content)
```

#### Character Integration

```python
# List available characters
characters = client.characters.list()
coding_character = next(c for c in characters if "coding" in c.name.lower())

# Use character in chat
response = chat.create(
    model="venice-uncensored",
    messages=[{"role": "user", "content": "Help me debug this Python code"}],
    venice_parameters={
        "character_slug": coding_character.slug,
        "thinking_enabled": True
    }
)
```

#### Advanced Async Pattern

```python
import asyncio
from typing import AsyncGenerator

async def chat_conversation() -> AsyncGenerator[str, None]:
    """Advanced async chat pattern with proper context management"""
    async with client:
        chat = ChatCompletion(client)
        
        messages = [{"role": "system", "content": "You are a coding assistant."}]
        
        while True:
            user_input = await get_user_input()  # Your input function
            messages.append({"role": "user", "content": user_input})
            
            response = await chat.create_async(
                model="venice-coder",
                messages=messages,
                temperature=0.7
            )
            
            assistant_message = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_message})
            
            yield assistant_message

# Usage
async def main():
    async for message in chat_conversation():
        print(f"Assistant: {message}")

asyncio.run(main())
```

### 🎨 Image Generation

#### Basic Image Generation

```python
from pyvenice import VeniceClient, ImageGeneration

client = VeniceClient()
image_gen = ImageGeneration(client)

# Generate high-quality image
response = image_gen.generate(
    prompt="A futuristic cityscape at sunset with flying cars",
    model="venice-sd35",
    width=1024,
    height=1024,
    steps=20,
    cfg_scale=7.5,
    style_preset="Cinematic"
)

# Save with convenience method
saved_paths = image_gen.save_images(
    response, 
    output_dir="./generated_images",
    format="png"
)
print(f"Images saved to: {saved_paths}")
```

#### Advanced Generation with Style Control

```python
# List available styles
styles = image_gen.list_styles()
print(f"Available styles: {styles}")

# Generate with multiple style variations
from pathlib import Path
output_dir = Path("style_variations")
output_dir.mkdir(exist_ok=True)

for style in ["Photographic", "Cinematic", "Anime", "Digital Art"]:
    response = image_gen.generate(
        prompt="A majestic dragon perched on a mountain peak",
        model="flux-dev",
        style_preset=style,
        width=1024,
        height=1024,
        steps=15
    )
    
    # Save with style in filename
    image_gen.save_images(
        response,
        output_dir=output_dir,
        filename_prefix=f"dragon_{style.lower()}",
        format="webp"
    )
```

#### Image Upscaling and Enhancement

```python
# Upscale existing image
with open("low_res_image.jpg", "rb") as f:
    image_data = f.read()

upscaled = image_gen.upscale(
    image=image_data,
    scale=4,  # 4x upscaling
    enhance=True,
    enhance_creativity=0.3,
    replication=1
)

# Save upscaled result
with open("upscaled_image.png", "wb") as f:
    f.write(upscaled)
```

#### Image Editing Pipeline

```python
# Edit existing image with text prompt
original_image = "path/to/original.jpg"

# Multiple editing steps
edited_image = image_gen.edit(
    image=original_image,
    prompt="Add autumn leaves falling in the background",
    model="venice-sd35"
)

# Further editing
final_image = image_gen.edit(
    image=edited_image,
    prompt="Make the lighting more dramatic with golden hour effect",
    model="venice-sd35"
)

# Save final result
with open("final_edited.png", "wb") as f:
    f.write(final_image)
```

#### Batch Image Generation

```python
import asyncio
from typing import List

async def batch_generate_images(prompts: List[str]) -> List[str]:
    """Generate multiple images concurrently"""
    tasks = []
    
    for i, prompt in enumerate(prompts):
        task = image_gen.generate_async(
            prompt=prompt,
            model="flux-schnell",  # Fast model for batch processing
            width=512,
            height=512,
            steps=4
        )
        tasks.append(task)
    
    # Execute all generations concurrently
    responses = await asyncio.gather(*tasks)
    
    # Save all images
    saved_paths = []
    for i, response in enumerate(responses):
        paths = image_gen.save_images(
            response,
            output_dir="batch_generated",
            filename_prefix=f"image_{i:03d}",
            format="png"
        )
        saved_paths.extend(paths)
    
    return saved_paths

# Usage
prompts = [
    "A serene mountain landscape",
    "A bustling cyberpunk street",
    "A peaceful forest clearing",
    "A majestic space station"
]

# Run batch generation
async def main():
    saved_files = await batch_generate_images(prompts)
    print(f"Generated {len(saved_files)} images")

asyncio.run(main())
```

#### OpenAI-Compatible Generation

```python
# Use OpenAI-style API for easy migration
response = image_gen.generate_openai_style(
    prompt="A robot reading a book in a library",
    model="dall-e-3",  # Mapped to Venice equivalent
    size="1024x1024",
    quality="standard",
    n=1
)

# Response format matches OpenAI
image_url = response.data[0].url  # If using URL mode
# or
image_b64 = response.data[0].b64_json  # If using base64 mode
```

### 🔊 Audio Generation

#### Text-to-Speech

```python
from pyvenice import VeniceClient, Audio

client = VeniceClient()
audio = Audio(client)

# Generate speech
audio_data = audio.create_speech(
    input="Welcome to PyVenice! This is a demonstration of high-quality text-to-speech.",
    voice="af_nova",  # Female voice
    response_format="mp3",
    speed=1.0
)

# Save to file
with open("speech.mp3", "wb") as f:
    f.write(audio_data)
```

#### Voice Selection and Streaming

```python
# List available voices
voices = audio.list_voices()
print(f"Available voices: {voices}")

# Real-time streaming audio
def stream_audio_to_file(text: str, filename: str):
    """Stream audio generation sentence by sentence"""
    with open(filename, "wb") as f:
        for audio_chunk in audio.create_speech_streaming(
            input=text,
            voice="am_daniel",  # Male voice
            response_format="wav",
            streaming=True
        ):
            f.write(audio_chunk)

# Use streaming
long_text = """
This is a longer text that will be processed sentence by sentence 
for real-time audio generation. Each sentence is processed as it's 
generated, providing lower latency for interactive applications.
"""

stream_audio_to_file(long_text, "streamed_speech.wav")
```

#### Advanced Audio Options

```python
# High-quality audio with different formats
formats = ["mp3", "wav", "opus", "aac", "flac"]

for format_type in formats:
    audio_data = audio.create_speech(
        input="Quality comparison test",
        voice="af_sarah",
        response_format=format_type,
        speed=1.2  # Slightly faster
    )
    
    # Save with format-specific extension
    audio.save_speech(
        input="Quality comparison test",
        output_path=f"quality_test.{format_type}",
        voice="af_sarah",
        response_format=format_type
    )
```

### 🔢 Embeddings

#### Single Text Embedding

```python
from pyvenice import VeniceClient, Embeddings

client = VeniceClient()
embeddings = Embeddings(client)

# Generate embedding
response = embeddings.create(
    input="PyVenice is a comprehensive Python client for Venice.ai",
    model="text-embedding-3-small",
    dimensions=1536
)

embedding_vector = response.data[0].embedding
print(f"Embedding dimensions: {len(embedding_vector)}")
```

#### Batch Processing

```python
# Process multiple texts efficiently
texts = [
    "Artificial intelligence is transforming technology",
    "Machine learning enables pattern recognition",
    "Deep learning uses neural networks",
    "Natural language processing handles text"
]

# Batch processing
embeddings_response = embeddings.create_batch(
    texts,
    model="text-embedding-3-small",
    dimensions=1536
)

# Process results
for i, embedding_data in enumerate(embeddings_response.data):
    print(f"Text {i+1}: {len(embedding_data.embedding)} dimensions")
```

### 🤖 Models & Capabilities

#### Model Discovery

```python
from pyvenice import VeniceClient, Models

client = VeniceClient()
models = Models(client)

# List all available models
all_models = models.list()
print(f"Total models: {len(all_models.data)}")

# Filter by type
text_models = models.list(type="text")
image_models = models.list(type="image")

print(f"Text models: {len(text_models.data)}")
print(f"Image models: {len(image_models.data)}")
```

#### Capability Checking

```python
# Check model capabilities before use
model_name = "venice-uncensored"
capabilities = models.get_capabilities(model_name)

print(f"Model: {model_name}")
print(f"Function calling: {capabilities.supportsFunctionCalling}")
print(f"Vision: {capabilities.supportsVision}")
print(f"Web search: {capabilities.supportsWebSearch}")
print(f"Reasoning: {capabilities.supportsReasoning}")

# Use capabilities to adapt behavior
if capabilities.supportsFunctionCalling:
    # Enable function calling
    tools = [{"type": "function", "function": {...}}]
else:
    # Fallback to regular chat
    tools = None
```

#### Model Compatibility Mapping

```python
# OpenAI model mapping
openai_models = models.get_compatibility_mapping()
print("OpenAI to Venice model mapping:")
for openai_model, venice_model in openai_models.items():
    print(f"  {openai_model} -> {venice_model}")

# Automatic model name translation
venice_model = models.map_model_name("gpt-4")
print(f"GPT-4 maps to: {venice_model}")
```

### 🔑 API Key Management

#### Key Information and Rate Limits

```python
from pyvenice import VeniceClient, APIKeys

client = VeniceClient()
api_keys = APIKeys(client)

# Get current key information
key_info = api_keys.get_info()
print(f"API Key Type: {key_info.data[0].apiKeyType}")
print(f"Description: {key_info.data[0].description}")

# Check rate limits
rate_limits = api_keys.get_rate_limits()
for endpoint, limits in rate_limits.data.items():
    print(f"{endpoint}: {limits.remaining}/{limits.limit} remaining")
```

#### Enterprise Key Management

```python
# Create new API key (requires admin privileges)
new_key = api_keys.create_key(
    key_type="INFERENCE",
    description="Production API key",
    usd_limit=1000.0,
    monthly_limit=10000
)

print(f"New key created: {new_key.data['id']}")

# Monitor usage
usage_log = api_keys.get_rate_limit_log(
    limit=100,
    start_date="2024-01-01"
)

for entry in usage_log.data:
    print(f"Endpoint: {entry.endpoint}, Used: {entry.used_tokens}")
```

### 👥 Characters

#### Character Discovery

```python
from pyvenice import VeniceClient, Characters

client = VeniceClient()
characters = Characters(client)

# List all characters
all_characters = characters.list()
print(f"Available characters: {len(all_characters)}")

# Filter by tags
coding_characters = characters.list_by_tag("coding")
helpful_characters = characters.list_by_tag("helpful")

# Content filtering
safe_characters = characters.list_safe()
adult_characters = characters.list_adult_only()
```

#### Character Integration

```python
# Find specific character
character = characters.get_character("helpful-assistant")
print(f"Character: {character.name}")
print(f"Description: {character.description}")

# Use in chat conversation
chat = ChatCompletion(client)
response = chat.create(
    model="venice-uncensored",
    messages=[{"role": "user", "content": "Help me write a Python function"}],
    venice_parameters={
        "character_slug": character.slug,
        "thinking_enabled": True
    }
)
```

### 💰 Billing & Usage

#### Usage Monitoring

```python
from pyvenice import VeniceClient, Billing

client = VeniceClient()
billing = Billing(client)

# Get recent usage
usage = billing.get_usage(
    start_date="2024-01-01",
    end_date="2024-01-31",
    currency="USD"
)

# Usage summary
summary = billing.get_usage_summary()
print(f"Total USD spent: ${summary['total_usd']}")
print(f"Total tokens: {summary['total_tokens']}")
```

#### Cost Analysis

```python
# Detailed usage analysis
all_usage = billing.get_all_usage(
    start_date="2024-01-01",
    currency="USD"
)

# Group by model
model_costs = {}
for entry in all_usage:
    model = entry.get('model', 'unknown')
    cost = entry.get('cost_usd', 0)
    model_costs[model] = model_costs.get(model, 0) + cost

print("Cost by model:")
for model, cost in sorted(model_costs.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model}: ${cost:.4f}")
```

## 🚀 Advanced Usage Patterns

### Error Handling & Recovery

```python
import asyncio
import logging
from pyvenice import VeniceClient, ChatCompletion
from pyvenice.exceptions import (
    RateLimitError, 
    ModelNotFoundError, 
    APIConnectionError,
    AuthenticationError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustChatClient:
    """Production-ready chat client with error handling and recovery"""
    
    def __init__(self, api_key: str, max_retries: int = 3):
        self.client = VeniceClient(api_key=api_key)
        self.chat = ChatCompletion(self.client)
        self.max_retries = max_retries
    
    async def chat_with_retry(self, **kwargs) -> str:
        """Chat with automatic retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = await self.chat.create_async(**kwargs)
                return response.choices[0].message.content
                
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = e.retry_after or (2 ** attempt)
                logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
                
            except ModelNotFoundError as e:
                # Try fallback model
                logger.warning(f"Model {kwargs['model']} not found, trying fallback")
                kwargs['model'] = "venice-uncensored"  # Fallback
                
            except APIConnectionError as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Connection error, retrying (attempt {attempt + 1})")
                await asyncio.sleep(2 ** attempt)
                
            except AuthenticationError as e:
                # Don't retry auth errors
                logger.error(f"Authentication failed: {e}")
                raise
                
        raise Exception("Max retries exceeded")

# Usage
async def main():
    client = RobustChatClient(api_key="your-api-key")
    
    try:
        response = await client.chat_with_retry(
            model="venice-uncensored",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response)
    except Exception as e:
        logger.error(f"Chat failed: {e}")
```

### Async Context Management

```python
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

class AsyncVeniceSession:
    """Async context manager for Venice.ai sessions"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.chat = None
    
    async def __aenter__(self):
        self.client = VeniceClient(api_key=self.api_key)
        self.chat = ChatCompletion(self.client)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()
    
    async def chat_stream(self, **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat responses asynchronously"""
        async for chunk in self.chat.create_streaming(**kwargs):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

# Usage with proper resource management
async def streaming_conversation():
    async with AsyncVeniceSession("your-api-key") as session:
        async for token in session.chat_stream(
            model="venice-coder",
            messages=[{"role": "user", "content": "Write a Python function"}]
        ):
            print(token, end="", flush=True)
```

### Batch Processing with Concurrency Control

```python
import asyncio
from typing import List, Dict, Any
from asyncio import Semaphore

class BatchProcessor:
    """Process multiple requests with concurrency control"""
    
    def __init__(self, client: VeniceClient, max_concurrent: int = 10):
        self.client = client
        self.chat = ChatCompletion(client)
        self.semaphore = Semaphore(max_concurrent)
    
    async def process_single_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single request with semaphore control"""
        async with self.semaphore:
            try:
                response = await self.chat.create_async(**request)
                return {
                    "success": True,
                    "response": response.choices[0].message.content,
                    "model": request.get("model"),
                    "request_id": request.get("id")
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "model": request.get("model"),
                    "request_id": request.get("id")
                }
    
    async def process_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple requests concurrently"""
        tasks = [self.process_single_request(request) for request in requests]
        return await asyncio.gather(*tasks)

# Usage
async def batch_example():
    client = VeniceClient()
    processor = BatchProcessor(client, max_concurrent=5)
    
    requests = [
        {
            "id": f"request_{i}",
            "model": "venice-uncensored",
            "messages": [{"role": "user", "content": f"Question {i}: What is AI?"}]
        }
        for i in range(20)
    ]
    
    results = await processor.process_batch(requests)
    
    # Process results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"Successful: {len(successful)}, Failed: {len(failed)}")
```

### Production Configuration

```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class VeniceConfig:
    """Production configuration for Venice.ai client"""
    api_key: str
    base_url: str = "https://api.venice.ai/api/v1"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    max_concurrent: int = 10
    default_model: str = "venice-uncensored"
    
    @classmethod
    def from_env(cls) -> "VeniceConfig":
        """Load configuration from environment variables"""
        return cls(
            api_key=os.getenv("VENICE_API_KEY", ""),
            base_url=os.getenv("VENICE_BASE_URL", cls.base_url),
            timeout=float(os.getenv("VENICE_TIMEOUT", cls.timeout)),
            max_retries=int(os.getenv("VENICE_MAX_RETRIES", cls.max_retries)),
            max_concurrent=int(os.getenv("VENICE_MAX_CONCURRENT", cls.max_concurrent)),
            default_model=os.getenv("VENICE_DEFAULT_MODEL", cls.default_model)
        )

class ProductionVeniceClient:
    """Production-ready Venice.ai client with comprehensive configuration"""
    
    def __init__(self, config: Optional[VeniceConfig] = None):
        self.config = config or VeniceConfig.from_env()
        self.client = VeniceClient(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
        self.chat = ChatCompletion(self.client)
    
    async def healthy_check(self) -> bool:
        """Check if the API is healthy"""
        try:
            models = await self.client.models.list_async()
            return len(models.data) > 0
        except Exception:
            return False
    
    async def __aenter__(self):
        # Verify connectivity on entry
        if not await self.healthy_check():
            raise ConnectionError("Venice.ai API is not accessible")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()

# Usage
async def production_example():
    config = VeniceConfig.from_env()
    
    async with ProductionVeniceClient(config) as client:
        response = await client.chat.create_async(
            model=config.default_model,
            messages=[{"role": "user", "content": "Hello, production!"}]
        )
        print(response.choices[0].message.content)
```

## ⚡ Performance & Reliability

### Caching & Optimization

```python
from functools import lru_cache
from typing import Dict, Any
import time

class CachedVeniceClient:
    """Venice client with intelligent caching"""
    
    def __init__(self, client: VeniceClient):
        self.client = client
        self.models = Models(client)
        self.chat = ChatCompletion(client)
        self._capability_cache = {}
        self._model_cache = {}
    
    @lru_cache(maxsize=100)
    def get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """Cache model capabilities for 24 hours"""
        if model not in self._capability_cache:
            self._capability_cache[model] = {
                "data": self.models.get_capabilities(model),
                "timestamp": time.time()
            }
        
        cache_entry = self._capability_cache[model]
        if time.time() - cache_entry["timestamp"] > 86400:  # 24 hours
            del self._capability_cache[model]
            return self.get_model_capabilities(model)
        
        return cache_entry["data"]
    
    def smart_parameter_filtering(self, model: str, **kwargs) -> Dict[str, Any]:
        """Intelligently filter parameters based on cached capabilities"""
        capabilities = self.get_model_capabilities(model)
        
        # Remove unsupported parameters
        filtered = kwargs.copy()
        
        if not capabilities.supportsFunctionCalling:
            filtered.pop("tools", None)
            filtered.pop("tool_choice", None)
            filtered.pop("parallel_tool_calls", None)
        
        if not capabilities.supportsWebSearch:
            venice_params = filtered.get("venice_parameters", {})
            venice_params.pop("web_search", None)
            if venice_params:
                filtered["venice_parameters"] = venice_params
        
        return filtered
```

### Connection Pooling & Resource Management

```python
import aiohttp
import asyncio
from typing import Optional

class PooledVeniceClient:
    """Venice client with connection pooling"""
    
    def __init__(self, api_key: str, max_connections: int = 100):
        self.api_key = api_key
        self.max_connections = max_connections
        self._session: Optional[aiohttp.ClientSession] = None
        self._client: Optional[VeniceClient] = None
    
    async def __aenter__(self):
        # Create connection pool
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        self._client = VeniceClient(
            api_key=self.api_key,
            session=self._session
        )
        
        return self._client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
```

### Rate Limiting & Backoff

```python
import asyncio
import time
from collections import defaultdict
from typing import Dict, Optional

class RateLimiter:
    """Intelligent rate limiting with exponential backoff"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests_timestamps = defaultdict(list)
        self.backoff_delays = defaultdict(float)
    
    async def acquire(self, endpoint: str) -> None:
        """Acquire permission to make a request"""
        now = time.time()
        
        # Clean old timestamps
        cutoff = now - 60  # 1 minute ago
        self.requests_timestamps[endpoint] = [
            ts for ts in self.requests_timestamps[endpoint] 
            if ts > cutoff
        ]
        
        # Check if we need to wait
        if len(self.requests_timestamps[endpoint]) >= self.requests_per_minute:
            wait_time = 60 - (now - self.requests_timestamps[endpoint][0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Apply exponential backoff if we've been rate limited
        if self.backoff_delays[endpoint] > 0:
            await asyncio.sleep(self.backoff_delays[endpoint])
            self.backoff_delays[endpoint] *= 0.5  # Reduce backoff
        
        self.requests_timestamps[endpoint].append(now)
    
    def apply_backoff(self, endpoint: str, retry_after: Optional[float] = None):
        """Apply exponential backoff after rate limit"""
        if retry_after:
            self.backoff_delays[endpoint] = retry_after
        else:
            self.backoff_delays[endpoint] = min(
                self.backoff_delays[endpoint] * 2 or 1, 
                60  # Max 60 seconds
            )

class RateLimitedVeniceClient:
    """Venice client with built-in rate limiting"""
    
    def __init__(self, client: VeniceClient, requests_per_minute: int = 60):
        self.client = client
        self.chat = ChatCompletion(client)
        self.rate_limiter = RateLimiter(requests_per_minute)
    
    async def chat_with_rate_limit(self, **kwargs) -> str:
        """Chat with automatic rate limiting"""
        await self.rate_limiter.acquire("chat/completions")
        
        try:
            response = await self.chat.create_async(**kwargs)
            return response.choices[0].message.content
        except RateLimitError as e:
            self.rate_limiter.apply_backoff("chat/completions", e.retry_after)
            raise
```

### Health Monitoring & Circuit Breaker

```python
import asyncio
from enum import Enum
from typing import Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery

@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for API reliability"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3
    
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: Optional[datetime] = field(default=None)
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from OPEN state"""
        if self.last_failure_time is None:
            return True
        
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

class ReliableVeniceClient:
    """Venice client with circuit breaker and health monitoring"""
    
    def __init__(self, client: VeniceClient):
        self.client = client
        self.chat = ChatCompletion(client)
        self.circuit_breaker = CircuitBreaker()
        self.health_status = {"healthy": True, "last_check": datetime.now()}
    
    async def health_check(self) -> bool:
        """Check API health"""
        try:
            models = await self.client.models.list_async()
            self.health_status = {"healthy": True, "last_check": datetime.now()}
            return len(models.data) > 0
        except Exception:
            self.health_status = {"healthy": False, "last_check": datetime.now()}
            return False
    
    async def chat_with_circuit_breaker(self, **kwargs) -> str:
        """Chat with circuit breaker protection"""
        return await self.circuit_breaker.call(
            self.chat.create_async, **kwargs
        )
```

### Performance Monitoring

```python
import time
import statistics
from typing import Dict, List
from dataclasses import dataclass, field

@dataclass
class PerformanceMetrics:
    """Track performance metrics"""
    response_times: List[float] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    
    def add_response_time(self, duration: float):
        self.response_times.append(duration)
        if len(self.response_times) > 1000:  # Keep last 1000 measurements
            self.response_times.pop(0)
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.response_times:
            return {}
        
        return {
            "avg_response_time": statistics.mean(self.response_times),
            "median_response_time": statistics.median(self.response_times),
            "p95_response_time": statistics.quantiles(self.response_times, n=20)[18],
            "success_rate": self.success_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 0,
            "total_requests": len(self.response_times),
            "avg_tokens_per_request": self.total_tokens / len(self.response_times) if self.response_times else 0
        }

class MonitoredVeniceClient:
    """Venice client with performance monitoring"""
    
    def __init__(self, client: VeniceClient):
        self.client = client
        self.chat = ChatCompletion(client)
        self.metrics = PerformanceMetrics()
    
    async def monitored_chat(self, **kwargs) -> str:
        """Chat with performance monitoring"""
        start_time = time.time()
        
        try:
            response = await self.chat.create_async(**kwargs)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.add_response_time(duration)
            self.metrics.success_count += 1
            
            if hasattr(response, 'usage') and response.usage:
                self.metrics.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.metrics.error_count += 1
            raise
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "metrics": self.metrics.get_stats(),
            "timestamp": time.time(),
            "circuit_breaker_state": getattr(self, 'circuit_breaker', {}).get('state', 'N/A')
        }
```

## 🎯 Supported Endpoints

- 💬 **Chat Completions** - `/chat/completions` with streaming and web search
- 🎨 **Image Generation** - `/image/generate`, `/images/generations`
- 🔍 **Image Upscaling** - `/image/upscale`
- 🔊 **Text to Speech** - `/audio/speech` with streaming
- 📊 **Embeddings** - `/embeddings`
- 🤖 **Model Management** - `/models`, `/models/traits`
- 🔑 **API Keys** - `/api_keys`, rate limits, and web3 key generation
- 👤 **Characters** - `/characters` for character-based interactions
- 💰 **Billing** - `/billing/usage` with pagination

## ⚙️ Configuration

### Environment Variables

```python
import os
from pyvenice import VeniceClient

# Core configuration
os.environ["VENICE_API_KEY"] = "your-api-key"
os.environ["VENICE_BASE_URL"] = "https://api.venice.ai/api/v1"  # Optional

# Performance tuning
os.environ["VENICE_TIMEOUT"] = "30.0"
os.environ["VENICE_MAX_RETRIES"] = "3"
os.environ["VENICE_MAX_CONCURRENT"] = "10"
os.environ["VENICE_DEFAULT_MODEL"] = "venice-uncensored"
```

### Client Configuration

```python
# Basic configuration
client = VeniceClient(
    api_key="your-api-key",
    base_url="https://api.venice.ai/api/v1",
    timeout=30.0,
    max_retries=3
)

# Advanced configuration
client = VeniceClient(
    api_key="your-api-key",
    base_url="https://api.venice.ai/api/v1",
    timeout=30.0,
    max_retries=3,
    headers={
        "User-Agent": "MyApp/1.0",
        "X-Request-ID": "unique-request-id"
    }
)
```

### Configuration Best Practices

```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class VeniceConfig:
    """Production configuration"""
    api_key: str
    base_url: str = "https://api.venice.ai/api/v1"
    timeout: float = 30.0
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> "VeniceConfig":
        return cls(
            api_key=os.getenv("VENICE_API_KEY", ""),
            base_url=os.getenv("VENICE_BASE_URL", cls.base_url),
            timeout=float(os.getenv("VENICE_TIMEOUT", cls.timeout)),
            max_retries=int(os.getenv("VENICE_MAX_RETRIES", cls.max_retries))
        )

# Usage
config = VeniceConfig.from_env()
client = VeniceClient(
    api_key=config.api_key,
    base_url=config.base_url,
    timeout=config.timeout,
    max_retries=config.max_retries
)
```

## 🧪 Testing

### Running Tests

```bash
# Install with test dependencies
pip install pyvenice[test]

# Run unit tests (no API key required)
pytest -m "not integration" --cov=pyvenice --cov-report=term-missing

# Run integration tests (requires VENICE_API_KEY)
export VENICE_API_KEY="your-api-key"
pytest tests/integration/

# Run all tests with detailed output
pytest -v --cov=pyvenice --cov-report=html

# Run specific test module
pytest tests/test_chat.py -v
```

### Test Configuration

```python
# tests/conftest.py
import pytest
from pyvenice import VeniceClient
from pyvenice.testing import MockVeniceClient

@pytest.fixture
def mock_client():
    """Mock client for unit tests"""
    return MockVeniceClient()

@pytest.fixture
def real_client():
    """Real client for integration tests"""
    return VeniceClient()  # Uses VENICE_API_KEY from env
```

### Writing Tests

```python
# Example test
import pytest
from pyvenice import ChatCompletion

def test_chat_completion_basic(mock_client):
    """Test basic chat completion"""
    chat = ChatCompletion(mock_client)
    
    response = chat.create(
        model="venice-uncensored",
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    assert response.choices[0].message.content
    assert response.model == "venice-uncensored"

@pytest.mark.integration
async def test_chat_completion_real(real_client):
    """Integration test with real API"""
    chat = ChatCompletion(real_client)
    
    response = await chat.create_async(
        model="venice-uncensored",
        messages=[{"role": "user", "content": "Say 'test'"}]
    )
    
    assert "test" in response.choices[0].message.content.lower()
```

### Development Testing

```bash
# Install development environment
pip install -e .[dev]

# Run linting
ruff check .

# Fix linting issues
ruff check --fix .

# Run tests with coverage
pytest --cov=pyvenice --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=pyvenice --cov-report=html
open htmlcov/index.html
```

## 🔧 Troubleshooting

### Common Issues

#### Authentication Errors
```python
# Error: 401 Unauthorized
try:
    response = client.chat.create(...)
except AuthenticationError as e:
    print(f"API key invalid: {e}")
    # Check: Is VENICE_API_KEY set correctly?
    # Check: Is the API key still valid?
```

#### Rate Limiting
```python
# Error: 429 Too Many Requests
try:
    response = client.chat.create(...)
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
    # Solution: Use built-in retry logic or implement backoff
```

#### Model Not Found
```python
# Error: Model not available
try:
    response = client.chat.create(model="invalid-model", ...)
except ModelNotFoundError as e:
    print(f"Available models: {e.available_models}")
    # Solution: Use client.models.list() to see available models
```

#### Connection Issues
```python
# Error: Connection timeout
try:
    response = client.chat.create(...)
except APIConnectionError as e:
    print(f"Connection failed: {e}")
    # Solution: Check internet connection, increase timeout
```

### Parameter Validation Issues

```python
# Issue: Parameters being ignored
response = client.chat.create(
    model="venice-uncensored",
    parallel_tool_calls=True,  # May be ignored if unsupported
    messages=[...]
)

# Solution: Check model capabilities
capabilities = client.models.get_capabilities("venice-uncensored")
if capabilities.supportsFunctionCalling:
    # Use function calling parameters
    pass
```

### Performance Issues

```python
# Issue: Slow response times
# Solution: Use async client for better performance
async def fast_chat():
    async with VeniceClient() as client:
        chat = ChatCompletion(client)
        response = await chat.create_async(...)
        return response

# Issue: Memory usage
# Solution: Use streaming for large responses
for chunk in client.chat.create_streaming(...):
    process_chunk(chunk)  # Process incrementally
```

### Debug Mode

```python
import logging
from pyvenice import VeniceClient

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create client with debug info
client = VeniceClient(
    api_key="your-api-key",
    debug=True  # Enables request/response logging
)
```

### Migration Guide

#### From v0.2.x to v0.3.x

```python
# OLD: Direct model string
response = client.chat.create(model="gpt-4", ...)

# NEW: Use Venice models or compatibility mapping
models = client.models.get_compatibility_mapping()
venice_model = models.get("gpt-4", "venice-uncensored")
response = client.chat.create(model=venice_model, ...)
```

#### From OpenAI SDK

```python
# OLD: OpenAI client
import openai
client = openai.OpenAI(api_key="...")

# NEW: PyVenice client
from pyvenice import VeniceClient, ChatCompletion
client = VeniceClient(api_key="...")
chat = ChatCompletion(client)

# Most parameters remain the same
response = chat.create(
    model="venice-uncensored",  # Venice model
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=100
)
```

### Getting Help

1. **Check the documentation** - Examples above cover most use cases
2. **Enable debug logging** - See what requests are being made
3. **Check model capabilities** - Ensure your parameters are supported
4. **Use the test suite** - Run integration tests to verify connectivity
5. **File an issue** - [GitHub Issues](https://github.com/TheLustriVA/PyVenice/issues)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

```bash
# Setup development environment
git clone https://github.com/TheLustriVA/PyVenice.git
cd PyVenice
pip install -e .[dev]

# Run tests before submitting PR
pytest
ruff check .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔒 Security

PyVenice prioritizes security:

- All communications use HTTPS with certificate verification
- API keys are never logged or included in error messages  
- No telemetry or data collection
- Minimal dependencies, all well-maintained and audited
- Input validation prevents injection attacks

For security concerns, please email [kieran@bicheno.me] or open an issue on GitHub

## 📚 Documentation

For detailed documentation, visit [our docs](https://github.com/TheLustriVA/PyVenice#readme) or check out the [examples](src/) directory.

## 🙏 Acknowledgments

Built with ❤️ using:

- [httpx](https://github.com/encode/httpx) - Modern HTTP client
- [pydantic](https://github.com/pydantic/pydantic) - Data validation
- [Venice.ai](https://venice.ai) - The underlying API

## 📈 Project Status

PyVenice is under active development. We follow [semantic versioning](https://semver.org/) and maintain backwards compatibility for all minor releases.

[![Star History Chart](https://api.star-history.com/svg?repos=TheLustriVA/PyVenice&type=Date)](https://star-history.com/#TheLustriVA/PyVenice&Date)
