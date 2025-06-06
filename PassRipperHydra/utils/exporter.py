# utils/exporter.py
import json
import csv
import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def export_results(results, format_type, output_file):
    """
    Export results in specified format (JSON, CSV, PDF).
    Args:
        results (list): List of result dictionaries.
        format_type (str): Export format (json, csv, pdf).
        output_file (str): Path to output file (without extension).
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    format_type = format_type.lower()
    
    if format_type == "json":
        with open(f"{output_file}.json", "w") as f:
            json.dump(results, f, indent=4)
    elif format_type == "csv":
        with open(f"{output_file}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["target", "mode", "username", "password", "status", "timestamp"])
            writer.writeheader()
            for result in results:
                result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow(result)
    elif format_type == "pdf":
        pdf = SimpleDocTemplate(f"{output_file}.pdf", pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph("PassRipperHydra Attack Report", styles['Title'])
        elements.append(title)
        
        # Metadata
        meta = Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(meta)
        elements.append(Paragraph("<br/>", styles['Normal']))
        
        # Table data
        data = [["Target", "Mode", "Username", "Password", "Status", "Timestamp"]]
        for result in results:
            result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data.append([
                result.get("target", "N/A"),
                result.get("mode", "N/A"),
                result.get("username", "N/A"),
                result.get("password", "N/A"),
                result.get("status", "N/A"),
                result.get("timestamp", "N/A")
            ])
        
        # Table styling
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        
        pdf.build(elements)