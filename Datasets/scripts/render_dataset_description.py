"""Создать HTML-версии описания датасета и короткого отчёта Task1."""

from pathlib import Path
from markdown_it import MarkdownIt
from dataset_paths import DATASET_DIR


def main():
    directory = DATASET_DIR
    style = """
    body { font: 16px/1.6 'Segoe UI', Arial, sans-serif; color: #202b36;
           max-width: 1050px; margin: 40px auto; padding: 0 24px; }
    h1 { font-size: 30px; line-height: 1.2; } h2 { margin-top: 32px; }
    table { border-collapse: collapse; width: 100%; font-size: 14px; margin: 20px 0; }
    td, th { padding: 8px 12px; border: 1px solid #ccd4dc; vertical-align: top; }
    th { background: #edf2f7; text-align: left; }
    tr:nth-child(even) { background: #fafbfc; }
    code { font-size: 0.9em; overflow-wrap: anywhere; }
    pre { padding: 16px; background: #edf2f7; white-space: pre-wrap; }
    img { max-width: 100%; height: auto; }
    @media print { body { margin: 0; padding: 0; font-size: 11pt; }
      h1 { font-size: 20pt; } h2 { font-size: 15pt; } h3 { font-size: 12pt; }
      table { font-size: 9pt; } tr, img { break-inside: avoid; }
      h1, h2, h3 { break-after: avoid; } }
    @page { size: A4; margin: 18mm; }
    """
    reports = [
        ("dataset_description", "Описание датасета и результаты классификации — Task1"),
        ("task1_report", "Отчёт по Task1 — Великолепная четверка"),
    ]
    for name, title in reports:
        text = (directory / f"{name}.md").read_text(encoding="utf-8")
        body = MarkdownIt("commonmark").enable("table").render(text)
        document = ("<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
                    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                    "<title>" + title + "</title><style>" + style +
                    "</style></head><body>" + body + "</body></html>")
        output = directory / f"{name}.html"
        output.write_text(document, encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
