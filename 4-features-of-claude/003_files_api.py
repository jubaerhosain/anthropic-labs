"""
Files API — upload an image once, reference it across multiple requests.

Contrast with 002_images.ipynb, which base64-encodes the image and embeds the
full encoded bytes in every message's content block:

    with open(path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}}

That means the entire image payload is re-sent on every single API call that
references it. The Files API instead uploads the bytes once, hands back a
`file_id`, and every subsequent message just points at that ID:

    {"type": "image", "source": {"type": "file", "file_id": file_id}}

No re-encoding, no re-uploading, smaller request bodies.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
model = "claude-opus-5"

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images", "prop1.png")
FILES_API_BETA = "files-api-2025-04-14"


def upload_image(path: str) -> str:
    """Upload an image once and return its file_id."""
    with open(path, "rb") as f:
        uploaded = client.beta.files.upload(
            file=(os.path.basename(path), f, "image/png"),
        )
    print(f"Uploaded file_id={uploaded.id} ({uploaded.size_bytes} bytes)")
    return uploaded.id


def ask_about_image(file_id: str, question: str) -> str:
    """Send a question that references the already-uploaded image by file_id.

    No image bytes are included in this request — only a pointer to the
    upload from upload_image().
    """
    response = client.beta.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image",
                        "source": {"type": "file", "file_id": file_id},
                    },
                ],
            }
        ],
        betas=[FILES_API_BETA],
    )
    return next(block.text for block in response.content if block.type == "text")


def main() -> None:
    file_id = upload_image(IMAGE_PATH)

    questions = [
        "What's in this image? Give a one-sentence summary.",
        "Are there any visible signs of fire risk or hazards in this image?",
        "Describe the roofing material and overall condition of the structure.",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        answer = ask_about_image(file_id, question)
        print(f"A: {answer}")

    # List files to show the upload is a persisted resource, not per-request state
    print("\nFiles in account:")
    for f in client.beta.files.list():
        print(f"  {f.id}: {f.filename} ({f.size_bytes} bytes)")

    # Clean up
    client.beta.files.delete(file_id)
    print(f"\nDeleted file_id={file_id}")


if __name__ == "__main__":
    main()
