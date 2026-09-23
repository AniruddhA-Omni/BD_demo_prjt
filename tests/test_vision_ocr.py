from PIL import Image, ImageDraw

from app.ingestion.router import IngestionRouter


def test_ingestion_router_extracts_ocr_text_from_images(tmp_path):
    image_path = tmp_path / "chart.png"
    image = Image.new("RGB", (220, 80), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 20), "Revenue 120", fill="black")
    image.save(image_path)

    evidence = IngestionRouter().ingest_file(image_path)

    assert evidence
    assert evidence[0].source_type == "image"
    assert evidence[0].file_type == "png"
    assert "Image file" in evidence[0].content or "Revenue" in evidence[0].content or "120" in evidence[0].content
