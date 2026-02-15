"""Email attachment fetching and image analysis via GPT-4o Vision."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

from ..config import settings
from .auth import get_auth_headers

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# Supported image MIME types
IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
}


@dataclass
class AttachmentInfo:
    """Basic attachment info."""
    id: str
    name: str
    content_type: str
    size: int
    is_image: bool


@dataclass
class ImageAnalysis:
    """Result of image analysis."""
    attachment_id: str
    filename: str
    description: str
    error: str | None = None


async def list_attachments(mailbox: str, message_id: str) -> list[AttachmentInfo]:
    """List attachments for an email.
    
    Args:
        mailbox: Email address of the shared mailbox
        message_id: Graph API message ID
        
    Returns:
        List of attachment metadata (without content)
    """
    headers = get_auth_headers()
    url = f"{GRAPH_BASE}/users/{mailbox}/messages/{message_id}/attachments"
    params = {"$select": "id,name,contentType,size"}
    
    attachments = []
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=headers, params=params)
        
        if resp.status_code == 200:
            data = resp.json()
            for att in data.get("value", []):
                content_type = att.get("contentType", "").lower()
                attachments.append(AttachmentInfo(
                    id=att["id"],
                    name=att.get("name", "unknown"),
                    content_type=content_type,
                    size=att.get("size", 0),
                    is_image=content_type in IMAGE_TYPES,
                ))
        else:
            print(f"[attachments] Error listing: {resp.status_code} {resp.text}")
    
    return attachments


async def get_attachment_content(
    mailbox: str,
    message_id: str,
    attachment_id: str,
) -> bytes | None:
    """Download attachment content.
    
    Returns:
        Raw bytes of the attachment, or None on error
    """
    headers = get_auth_headers()
    url = f"{GRAPH_BASE}/users/{mailbox}/messages/{message_id}/attachments/{attachment_id}"
    
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url, headers=headers)
        
        if resp.status_code == 200:
            data = resp.json()
            # Graph API returns base64-encoded content
            content_b64 = data.get("contentBytes")
            if content_b64:
                return base64.b64decode(content_b64)
        else:
            print(f"[attachments] Error downloading: {resp.status_code}")
    
    return None


async def analyze_image(
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> str:
    """Analyze an image using GPT-4o Vision.
    
    Args:
        image_bytes: Raw image bytes
        filename: Original filename (for context)
        content_type: MIME type
        
    Returns:
        Text description of the image content
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    # Encode to base64 data URL
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    
    # Normalize content type
    if content_type == "image/jpg":
        content_type = "image/jpeg"
    
    data_url = f"data:{content_type};base64,{b64_data}"
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that analyzes images from customer service emails. "
                        "Describe what you see in detail, focusing on:\n"
                        "- Any text visible (OCR)\n"
                        "- Screenshots: what application/website, what's shown, any error messages\n"
                        "- Documents: type, key information\n"
                        "- Forms: filled fields, highlighted areas\n"
                        "Be concise but thorough. Reply in Hungarian if the image contains Hungarian text."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Elemezd ezt a képet (fájlnév: {filename}):",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            max_tokens=1000,
        )
        
        return response.choices[0].message.content or "Nem sikerült elemezni a képet."
        
    except Exception as e:
        print(f"[attachments] Vision API error: {e}")
        return f"Hiba a képelemzés során: {str(e)}"


PDF_TYPES = {
    "application/pdf",
}


async def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pymupdf."""
    import fitz
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts).strip()
    except Exception as e:
        return f"Hiba a PDF feldolgozás során: {e}"


async def extract_all_attachments(
    mailbox: str,
    message_id: str,
    max_items: int = 10,
) -> list[dict]:
    """Extract text from all attachments (PDF → pymupdf, image → Vision).

    Returns list of {name, type, extracted_text}.
    """
    atts = await list_attachments(mailbox, message_id)
    atts = atts[:max_items]

    results = []
    for att in atts:
        content = await get_attachment_content(mailbox, message_id, att.id)
        if content is None:
            results.append({
                "name": att.name,
                "type": att.content_type,
                "extracted_text": "Nem sikerült letölteni a csatolmányt",
            })
            continue

        if att.content_type in PDF_TYPES:
            text = await extract_pdf_text(content)
            results.append({
                "name": att.name,
                "type": "pdf",
                "extracted_text": text,
            })
        elif att.is_image:
            desc = await analyze_image(content, att.name, att.content_type)
            results.append({
                "name": att.name,
                "type": "image",
                "extracted_text": desc,
            })
        else:
            results.append({
                "name": att.name,
                "type": att.content_type,
                "extracted_text": "(nem támogatott formátum)",
            })

    return results


async def analyze_email_attachments(
    mailbox: str,
    message_id: str,
    images_only: bool = True,
    max_images: int = 5,
) -> list[ImageAnalysis]:
    """Fetch and analyze image attachments from an email.
    
    Args:
        mailbox: Email address of the shared mailbox
        message_id: Graph API message ID
        images_only: If True, only process image attachments
        max_images: Maximum number of images to analyze (cost control)
        
    Returns:
        List of image analysis results
    """
    # 1. List attachments
    attachments = await list_attachments(mailbox, message_id)
    
    if images_only:
        attachments = [a for a in attachments if a.is_image]
    
    # Limit for cost control
    attachments = attachments[:max_images]
    
    if not attachments:
        return []
    
    # 2. Analyze each image
    results = []
    for att in attachments:
        if not att.is_image:
            continue
            
        # Download content
        content = await get_attachment_content(mailbox, message_id, att.id)
        
        if content is None:
            results.append(ImageAnalysis(
                attachment_id=att.id,
                filename=att.name,
                description="",
                error="Nem sikerült letölteni a csatolmányt",
            ))
            continue
        
        # Analyze with vision
        description = await analyze_image(content, att.name, att.content_type)
        
        results.append(ImageAnalysis(
            attachment_id=att.id,
            filename=att.name,
            description=description,
            error=None,
        ))
    
    return results
