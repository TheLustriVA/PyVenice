"""
Example usage of the pyvenice Venice.ai API client library.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path to find the pyvenice package
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyvenice import VeniceClient, ChatCompletion, ImageGeneration, Models
from pyvenice.api_keys import APIKeys

# Initialize client - requires VENICE_API_KEY environment variable
client = VeniceClient()


def example_chat():
    """Example of using chat completions."""
    chat = ChatCompletion(client)

    # Simple chat
    response = chat.create(
        model="venice-uncensored",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )

    print("Chat Response:", response.choices[0].message["content"])

    # Chat with web search
    response = chat.create_with_web_search(
        model="venice-uncensored",
        messages=[{"role": "user", "content": "What's the latest news about AI?"}],
        search_mode="auto",
    )

    print("\nChat with Web Search:", response.choices[0].message["content"])


def example_models():
    """Example of listing models and checking capabilities."""
    models = Models(client)

    # List all text models
    text_models = models.list(type="text")
    print("\nAvailable Text Models:")
    for model in text_models.data[:5]:  # Show first 5
        print(f"- {model.id}: {model.model_spec.description}")

    # Check model capabilities
    model_id = "venice-uncensored"
    capabilities = models.get_capabilities(model_id)
    if capabilities:
        print(f"\n{model_id} capabilities:")
        print(f"- Supports function calling: {capabilities.supportsFunctionCalling}")
        print(f"- Supports web search: {capabilities.supportsWebSearch}")
        print(f"- Supports vision: {capabilities.supportsVision}")


def example_image_generation():
    """Example of generating images."""
    image_gen = ImageGeneration(client)

    # List available styles
    styles = image_gen.list_styles()
    print("\nAvailable Image Styles:", styles[:5])  # Show first 5

    # Generate an image
    response = image_gen.generate(
        prompt="A serene mountain landscape at sunset",
        model="venice-sd35",
        style_preset="Cinematic",
        width=1024,
        height=1024,
    )

    print(f"\nGenerated image ID: {response.id}")
    print(f"Number of images: {len(response.images)}")

    # Save image to file (base64 decoded)
    if response.images:
        import base64

        image_data = base64.b64decode(response.images[0])
        with open("generated_image.webp", "wb") as f:
            f.write(image_data)
        print("Image saved as generated_image.webp")

    # Image editing examples
    print("\n=== Image Editing ===")
    try:
        # Use a simple base64 encoded 1x1 pixel PNG image for testing
        tiny_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        # Example 1: Edit with base64 image
        edited_image = image_gen.edit(
            prompt="Make this image blue",
            image=tiny_image
        )
        
        # Save edited image
        with open("edited_image.png", "wb") as f:
            f.write(edited_image)
        print("✅ Image edited and saved as edited_image.png")
        
        # Example 2: Edit from file (if the generated image exists)
        if Path("generated_image.webp").exists():
            edited_from_file = image_gen.edit(
                prompt="Add autumn colors",
                image="generated_image.webp"
            )
            
            with open("edited_from_file.png", "wb") as f:
                f.write(edited_from_file)
            print("✅ File-based image edited and saved as edited_from_file.png")
            
    except Exception as e:
        print(f"❌ Image editing error: {e}")


def multiple_images():
    image_gen = ImageGeneration(client)

    styles = image_gen.list_styles()

    for style in styles:
        response = image_gen.generate(
            prompt="A flying sorceress wearing translucent-azure arcane robes in a desert setting",
            model="hidream",
            style_preset=style,
            width=1024,
            height=1024,
        )

        image_gen.save_images(response, output_dir="outputs", format="png")


#        if response.images:
#            import base64
#
#            image_data = base64.b64decode(response.images[0])
#            with open(f"image_{style}.webp", "wb") as f:
#                f.write(image_data)
#            print(f"image_{style}.webp generated.")


def example_streaming():
    """Example of streaming chat responses."""
    chat = ChatCompletion(client)

    print("\nStreaming response:")
    stream = chat.create(
        model="venice-uncensored",
        messages=[{"role": "user", "content": "Write a short poem about coding"}],
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].get("delta", {}).get("content"):
            print(chunk.choices[0]["delta"]["content"], end="", flush=True)
    print()  # New line at end


def example_api_key_management():
    """Example of API key management operations."""
    api_keys = APIKeys(client)
    
    print("\n=== API Key Management ===")
    
    try:
        # Get current API key info
        print("Current API key info:")
        info = api_keys.get_info()
        if info.get("data"):
            for key in info["data"][:2]:  # Show first 2 keys
                print(f"  - Type: {key.get('apiKeyType', 'N/A')}")
                print(f"    Description: {key.get('description', 'N/A')}")
                print(f"    ID: {key.get('id', 'N/A')[:20]}...")
        
        # Get rate limits
        print("\nRate limits:")
        limits = api_keys.get_rate_limits()
        if limits.get("data"):
            for endpoint, limit in limits["data"].items():
                if isinstance(limit, dict):
                    print(f"  - {endpoint}: {limit.get('remaining', 'N/A')}/{limit.get('limit', 'N/A')} remaining")
        
        # Get Web3 token (for wallet authentication)
        print("\nWeb3 token availability:")
        try:
            web3_token = api_keys.get_web3_token()
            if web3_token.success:
                print("  ✅ Web3 token available for wallet authentication")
            else:
                print("  ❌ Web3 token not available")
        except Exception as e:
            print(f"  ❌ Web3 token error: {e}")
        
        # Note: We skip create_key and delete_key in examples to avoid creating real keys
        print("\nNote: Key creation/deletion examples skipped (requires ADMIN privileges)")
        print("To test key management:")
        print("  result = api_keys.create_key(")
        print("      key_type='INFERENCE',")
        print("      description='Test Key',")
        print("      usd_limit=10.0")
        print("  )")
        print("  api_keys.delete_key(result.data['id'])")
        
    except Exception as e:
        print(f"API key management error: {e}")


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("VENICE_API_KEY"):
        print("Please set VENICE_API_KEY environment variable")
        exit(1)

    print("=== Venice.ai API Examples ===\n")

    try:
        example_api_key_management()
        multiple_images()
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Clean up
        client.close()
