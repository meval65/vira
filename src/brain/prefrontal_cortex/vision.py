import os
import asyncio
import logging
import PIL.Image
from typing import Optional

from src.brain.brainstem import NeuralEventBus

class VisualProcessor:
    def __init__(self, hippocampus):
        self.hippocampus = hippocampus

    async def describe_and_store_image(
        self,
        image_path: str,
        user_context: Optional[str] = None,
        embedding_generator = None
    ) -> Optional[str]:
        if not image_path or not os.path.exists(image_path):
            return None
        
        await NeuralEventBus.set_activity("prefrontal_cortex", "Analyzing Image")
        
        try:
            image = PIL.Image.open(image_path)
            
            description_prompt = """Describe this image in detail. Include:
1. Main subjects/objects visible
2. Colors, composition, and visual style
3. Any text visible in the image
4. Mood or atmosphere conveyed
5. Context clues about when/where this might be

Keep the description informative but concise (2-3 paragraphs max).
Respond in Bahasa Indonesia."""

            from google import genai
            from google.genai import types
            
            client = genai.Client()
            
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=[
                        description_prompt,
                        image
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=500
                    )
                )
            )
            
            description = response.text.strip() if response.text else None
            
            if description:
                embedding = None
                if embedding_generator:
                    embedding = await embedding_generator(description)
                
                memory_id = await self.hippocampus.store_visual_memory(
                    image_description=description,
                    image_path=image_path,
                    embedding=embedding,
                    additional_context=user_context,
                    priority=0.6
                )
                
                await NeuralEventBus.emit(
                    "prefrontal_cortex", "hippocampus", "visual_memory_stored",
                    payload={
                        "memory_id": memory_id,
                        "description_len": len(description),
                        "image_path": image_path
                    }
                )
                
                await NeuralEventBus.clear_activity("prefrontal_cortex")
                return description
            
            await NeuralEventBus.clear_activity("prefrontal_cortex")
            return None
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Image description failed: {e}")
            await NeuralEventBus.clear_activity("prefrontal_cortex")
            return None
