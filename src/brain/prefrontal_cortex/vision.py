"""
Visual Processor using OpenRouter API for image analysis.

This module uses the unified OpenRouter client from brainstem.py
for multimodal image description and storage.
"""

import os
import base64
import logging
from typing import Optional

from src.brain.infrastructure.neural_event_bus import NeuralEventBus

logger = logging.getLogger(__name__)


class VisualProcessor:
    """Process and describe images using vision-capable LLM."""
    
    def __init__(self, hippocampus):
        self.hippocampus = hippocampus
        self._openrouter = None
    
    def bind_openrouter(self, openrouter_client):
        """Bind the OpenRouter client for API calls."""
        self._openrouter = openrouter_client

    async def describe_and_store_image(
        self,
        image_path: str,
        user_context: Optional[str] = None,
        embedding_generator = None
    ) -> Optional[str]:
        """
        Analyze an image and store the description as memory.
        
        Args:
            image_path: Path to the image file
            user_context: Optional context from user about the image
            embedding_generator: Optional function to generate embeddings
            
        Returns:
            Generated description or None on failure
        """
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Image path invalid or doesn't exist: {image_path}")
            return None
        
        if not self._openrouter:
            logger.error("OpenRouter client not bound to VisualProcessor")
            return None
        
        await NeuralEventBus.set_activity("prefrontal_cortex", "Analyzing Image")
        
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Determine media type from extension
            ext = os.path.splitext(image_path)[1].lower()
            media_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
            }
            media_type = media_type_map.get(ext, 'image/jpeg')
            
            # Build prompt
            description_prompt = """Describe this image in detail. Include:
1. Main subjects/objects visible
2. Colors, composition, and visual style
3. Any text visible in the image
4. Mood or atmosphere conveyed
5. Context clues about when/where this might be

Keep the description informative but concise (2-3 paragraphs max).
Respond in Bahasa Indonesia."""

            # Call vision API
            description = await self._openrouter.vision_completion(
                prompt=description_prompt,
                image_base64=image_base64,
                image_media_type=media_type,
                temperature=0.3,
                max_tokens=500
            )
            
            if description:
                # Generate embedding if available
                embedding = None
                if embedding_generator:
                    embedding = await embedding_generator(description)
                
                # Store as visual memory
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
                logger.info(f"Image analyzed and stored: {len(description)} chars")
                return description
            
            await NeuralEventBus.clear_activity("prefrontal_cortex")
            logger.warning("Vision model returned no description")
            return None
            
        except Exception as e:
            logger.error(f"Image description failed: {e}")
            await NeuralEventBus.clear_activity("prefrontal_cortex")
            return None


