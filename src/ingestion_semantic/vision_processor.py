"""Vision processor for handling images, charts, and tables with contextual retrieval."""

import asyncio
import base64
import io
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image

from openai import OpenAI
from langchain_core.documents import Document

from .contextual_retrieval import ContextualRetrieval


@dataclass
class VisionContent:
    """Container for vision processing results."""

    content_type: str  # 'image', 'chart', 'table', 'diagram'
    description: str
    contextualized_description: str
    context: str
    source: str
    metadata: Dict[str, Any]


class VisionProcessor:
    """
    Vision processor that handles images, charts, and tables in documents.

    Uses OpenAI Vision models to describe visual content and applies
    contextual retrieval to enhance descriptions with surrounding context.
    """

    def __init__(self, config, contextual_retrieval: ContextualRetrieval):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI(api_key=config.openai_api_key)
        self.contextual_retrieval = contextual_retrieval

        # Semaphore to limit concurrent vision requests
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        self.logger.info("✅ Vision processor initialized")

    def _resize_image_if_needed(
        self, image_data: bytes, max_size_mb: int = None
    ) -> bytes:
        """Resize image if it exceeds size limits."""
        if max_size_mb is None:
            max_size_mb = self.config.max_image_size_mb

        max_size_bytes = max_size_mb * 1024 * 1024

        if len(image_data) <= max_size_bytes:
            return image_data

        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(image_data))

            # Calculate resize factor
            resize_factor = (max_size_bytes / len(image_data)) ** 0.5
            new_width = int(image.width * resize_factor)
            new_height = int(image.height * resize_factor)

            # Resize image
            resized_image = image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

            # Save to bytes
            output = io.BytesIO()
            format = image.format if image.format else "JPEG"
            resized_image.save(output, format=format, quality=85, optimize=True)

            resized_data = output.getvalue()
            self.logger.info(
                f"📏 Resized image from {len(image_data) / 1024:.1f}KB to {len(resized_data) / 1024:.1f}KB"
            )

            return resized_data

        except Exception as e:
            self.logger.error(f"Error resizing image: {e}")
            return image_data[:max_size_bytes]  # Truncate if resize fails

    def _encode_image_to_base64(self, image_data: bytes) -> str:
        """Encode image data to base64 string."""
        return base64.b64encode(image_data).decode("utf-8")

    def _build_vision_prompt(self, content_type: str = "image") -> str:
        """Build prompt for vision model based on content type."""

        if content_type == "chart":
            return """Analyze this chart or graph carefully. Provide a detailed description that includes:
1. Type of chart (bar, line, pie, scatter, etc.)
2. Title and axis labels if visible
3. Key data points, trends, or patterns
4. Any notable insights or conclusions that can be drawn
5. Colors, legends, and other visual elements

Be precise and comprehensive in your description to help with document search and retrieval."""

        elif content_type == "table":
            return """Analyze this table carefully. Provide a detailed description that includes:
1. Table structure (number of rows and columns)
2. Column headers and row labels if visible
3. Key data points and values
4. Patterns, trends, or relationships in the data
5. Any totals, summaries, or calculated fields
6. Context about what the table represents

Be thorough and precise to help with document search and retrieval."""

        elif content_type == "diagram":
            return """Analyze this diagram or flowchart carefully. Provide a detailed description that includes:
1. Type of diagram (flowchart, organizational chart, process diagram, etc.)
2. Main components and their relationships
3. Flow of information or processes if applicable
4. Key elements, labels, and annotations
5. Purpose or function of the diagram
6. Any technical details or specifications shown

Be comprehensive and precise to help with document search and retrieval."""

        else:  # default image
            return """Analyze this image carefully and provide a detailed description that includes:
1. Main subjects or objects in the image
2. Setting, background, or environment
3. Important visual details, text, or annotations
4. Any data, charts, diagrams, or technical content
5. Colors, layout, and visual composition
6. Context or purpose of the image within the document

Be thorough and precise to help with document search and retrieval."""

    async def _describe_image_async(
        self, image_data: bytes, content_type: str = "image", surrounding_text: str = ""
    ) -> str:
        """Describe image using OpenAI Vision model asynchronously."""
        async with self.semaphore:
            try:
                # Resize image if needed
                processed_image_data = self._resize_image_if_needed(image_data)

                # Encode to base64
                base64_image = self._encode_image_to_base64(processed_image_data)

                # Build prompt
                prompt = self._build_vision_prompt(content_type)

                # Add surrounding text context if available
                if surrounding_text.strip():
                    prompt += (
                        f"\n\nSurrounding text context:\n{surrounding_text[:500]}..."
                    )

                # Call OpenAI Vision API
                response = self.client.chat.completions.create(
                    model=self.config.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.1,
                )

                description = response.choices[0].message.content.strip()
                self.logger.debug(
                    f"Generated {content_type} description: {description[:100]}..."
                )

                return description

            except Exception as e:
                self.logger.error(f"Error describing {content_type}: {e}")
                return f"[{content_type.title()} content - description unavailable due to processing error]"

    def describe_image(
        self, image_data: bytes, content_type: str = "image", surrounding_text: str = ""
    ) -> str:
        """Describe image using vision model (synchronous wrapper)."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, run in thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._describe_image_async(
                        image_data, content_type, surrounding_text
                    ),
                )
                return future.result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run
            return asyncio.run(
                self._describe_image_async(image_data, content_type, surrounding_text)
            )

    async def process_vision_content_async(
        self,
        images: List[
            Tuple[bytes, str, str]
        ],  # (image_data, content_type, surrounding_text)
        document_content: str,
        source_document: str,
        metadata: Optional[Dict] = None,
    ) -> List[VisionContent]:
        """
        Process multiple vision contents from a document asynchronously.

        Args:
            images: List of tuples (image_data, content_type, surrounding_text)
            document_content: Full document content for contextual retrieval
            source_document: Source document identifier
            metadata: Additional metadata

        Returns:
            List of VisionContent objects with descriptions and contextual information
        """
        if metadata is None:
            metadata = {}

        if not images:
            return []

        self.logger.info(
            f"🖼️  Processing {len(images)} vision contents from {source_document}"
        )

        # Generate descriptions for all images concurrently
        description_tasks = []
        for image_data, content_type, surrounding_text in images:
            task = self._describe_image_async(
                image_data, content_type, surrounding_text
            )
            description_tasks.append(task)

        descriptions = await asyncio.gather(*description_tasks, return_exceptions=True)

        # Apply contextual retrieval to descriptions
        valid_descriptions = []
        for i, desc in enumerate(descriptions):
            if isinstance(desc, Exception):
                self.logger.error(f"Error describing vision content {i}: {desc}")
                desc = "[Vision content - description error]"
            valid_descriptions.append(desc)

        # Contextualize descriptions
        contextualized_descriptions = []
        if valid_descriptions and document_content.strip():
            contextual_chunks = self.contextual_retrieval.contextualize_chunks(
                document_content=document_content,
                chunks=valid_descriptions,
                source_document=source_document,
                metadata={**metadata, "content_type": "vision_description"},
            )
            contextualized_descriptions = [
                chunk.contextualized_content for chunk in contextual_chunks
            ]
        else:
            contextualized_descriptions = valid_descriptions

        # Build VisionContent objects
        vision_contents = []
        for i, (image_data, content_type, surrounding_text) in enumerate(images):
            description = (
                valid_descriptions[i]
                if i < len(valid_descriptions)
                else "[Description unavailable]"
            )
            contextualized_desc = (
                contextualized_descriptions[i]
                if i < len(contextualized_descriptions)
                else description
            )

            # Extract context from contextualized description
            context = ""
            if len(contextualized_desc) > len(description):
                context = contextualized_desc[
                    : len(contextualized_desc) - len(description)
                ].strip()

            vision_content = VisionContent(
                content_type=content_type,
                description=description,
                contextualized_description=contextualized_desc,
                context=context,
                source=source_document,
                metadata={
                    **metadata,
                    "vision_index": i,
                    "image_size_bytes": len(image_data),
                    "surrounding_text_length": len(surrounding_text),
                    "description_length": len(description),
                    "contextualized": bool(context),
                },
            )

            vision_contents.append(vision_content)

        self.logger.info(
            f"✅ Processed {len(vision_contents)} vision contents from {source_document}"
        )
        return vision_contents

    def process_vision_content(
        self,
        images: List[Tuple[bytes, str, str]],
        document_content: str,
        source_document: str,
        metadata: Optional[Dict] = None,
    ) -> List[VisionContent]:
        """Process vision content (synchronous wrapper)."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, run in thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.process_vision_content_async(
                        images, document_content, source_document, metadata
                    ),
                )
                return future.result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run
            return asyncio.run(
                self.process_vision_content_async(
                    images, document_content, source_document, metadata
                )
            )

    def vision_content_to_documents(
        self, vision_contents: List[VisionContent]
    ) -> List[Document]:
        """Convert VisionContent objects to LangChain Documents."""
        documents = []

        for vision_content in vision_contents:
            # Use contextualized description as main content
            content = vision_content.contextualized_description

            metadata = {
                **vision_content.metadata,
                "source": vision_content.source,
                "content_type": "vision",
                "vision_type": vision_content.content_type,
                "original_description": vision_content.description,
                "context": vision_content.context,
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        return documents

    def get_processor_stats(self) -> Dict[str, Any]:
        """Get vision processor statistics."""
        return {
            "vision_model": self.config.vision_model,
            "max_image_size_mb": self.config.max_image_size_mb,
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "supported_types": ["image", "chart", "table", "diagram"],
            "contextual_retrieval_enabled": True,
        }
