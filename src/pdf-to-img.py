import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF
import questionary
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

console = Console()


def save_page_image(page_index, page, output_dir, out_type, scale):
    """Convert a single PDF page to an image and save it."""
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    out_file = output_dir / f"{page_index + 1}{out_type}"

    if out_type.lower() == ".webp":
        Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(  # pyright: ignore[reportArgumentType]
            out_file, "WEBP"
        )
    else:
        pix.save(str(out_file))

    return out_file


def convert_pdf(pdf_file, out_type, scale):
    """Convert a single PDF to images with multi-threading and progress bar."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path("output") / f"{pdf_file.stem}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold green]→ Converting:[/bold green] {pdf_file.name}")
    console.print(f"[bold cyan]→ Output folder:[/bold cyan] {output_dir}\n")

    doc = fitz.open(pdf_file)
    total_pages = doc.page_count

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Rendering pages...", total=total_pages)

        def process_page(i):
            page = doc.load_page(i)
            save_page_image(i, page, output_dir, out_type, scale)
            progress.advance(task)

        # Multi-threaded conversion per PDF
        with ThreadPoolExecutor() as executor:
            list(executor.map(process_page, range(total_pages)))

    console.print(
        f"\n🎉 [bold green]Done![/bold green] Exported {total_pages} pages to [cyan]{output_dir}[/cyan]."
    )


def main():
    batch_mode = "--batch" in sys.argv

    # Get PDFs
    if batch_mode:
        folder_path = questionary.path(
            "Select folder containing PDFs:", default="input/pdfs"
        ).ask()
        folder = Path(folder_path)
        if not folder.is_dir():
            console.print("[bold red]❌ Invalid folder path![/bold red]")
            return
        pdf_files = sorted(folder.glob("*.pdf"))
        if not pdf_files:
            console.print("[bold red]❌ No PDF files found in the folder![/bold red]")
            return
    else:
        pdf_path = questionary.path("Path to PDF file:").ask()
        if not pdf_path or not Path(pdf_path).is_file():
            console.print("[bold red]❌ Invalid PDF path![/bold red]")
            return
        pdf_files = [Path(pdf_path)]

    # Output type & scale
    out_type = questionary.select(
        "Choose output image type:", choices=[".png", ".jpg", ".webp"]
    ).ask()
    try:
        scale = float(
            questionary.text(
                "Enter scaling factor (1.0 = original, 2.0 = double size):",
                default="1.0",
            ).ask()
        )
        if scale <= 0:
            raise ValueError
    except ValueError:
        console.print("[bold red]Invalid scaling factor. Must be positive.[/bold red]")
        return

    # Convert all PDFs
    for pdf_file in pdf_files:
        convert_pdf(pdf_file, out_type, scale)


if __name__ == "__main__":
    main()
