"""
Export utilities for PDF, JSON, and CSV generation.
"""

import json
import csv
import io
from typing import Dict, List, Any
from dataclasses import asdict
from datetime import datetime


def export_zone_config_json(zones: List, recommendations: Any) -> str:
    """
    Export zone configuration as JSON.
    
    Args:
        zones: List of Zone objects
        recommendations: Recommendations object
        
    Returns:
        JSON string
    """
    config = {
        'generated_at': datetime.now().isoformat(),
        'parameters': {
            'approach_zone_radius_m': recommendations.approach_zone_radius,
            'decision_zone_radius_m': recommendations.decision_zone_radius,
            'hysteresis_m': recommendations.hysteresis,
            'persistence_moving_sec': recommendations.persistence_moving_sec,
            'persistence_stopped_sec': recommendations.persistence_stopped_sec,
            'based_on': recommendations.based_on,
            'noise_used_m': recommendations.noise_used,
        },
        'zones': [
            {
                'id': z.id,
                'name': z.name,
                'center_lat': z.lat,
                'center_lon': z.lon,
                'radius_m': z.radius,
                'hysteresis_m': z.hysteresis,
            }
            for z in zones
        ]
    }
    
    return json.dumps(config, indent=2)


def export_metrics_csv(
    summary: Dict,
    stops: List,
    gps_metrics: Any = None
) -> str:
    """
    Export computed metrics as CSV.
    
    Returns:
        CSV string
    """
    output = io.StringIO()
    
    # Summary section
    writer = csv.writer(output)
    writer.writerow(['# Dataset Summary'])
    writer.writerow(['Metric', 'Value', 'Unit'])
    writer.writerow(['Duration', summary.get('duration_sec', 0), 'seconds'])
    writer.writerow(['Sample Rate', summary.get('sample_rate', 0), 'Hz'])
    writer.writerow(['Total Distance', summary.get('total_distance_m', 0), 'meters'])
    writer.writerow(['Stop Count', summary.get('stop_count', 0), ''])
    writer.writerow(['% Time Stopped', summary.get('stop_percent', 0), '%'])
    writer.writerow(['Verdict', summary.get('verdict', 'UNKNOWN'), ''])
    writer.writerow([])
    
    # GPS metrics
    if gps_metrics:
        writer.writerow(['# GPS Quality Metrics'])
        writer.writerow(['Metric', 'Value', 'Unit'])
        if gps_metrics.reported_std_available:
            writer.writerow(['Reported Std Mean', gps_metrics.reported_std_mean, 'meters'])
            writer.writerow(['Reported Std Max', gps_metrics.reported_std_max, 'meters'])
        if gps_metrics.measured_available:
            writer.writerow(['Measured Std Mean', gps_metrics.measured_std_mean, 'meters'])
            writer.writerow(['Measured Radius Max', gps_metrics.measured_radius_max, 'meters'])
        writer.writerow(['Max Jump', gps_metrics.jump_max, 'meters'])
        writer.writerow(['Large Jump Count', gps_metrics.large_jump_count, ''])
        writer.writerow([])
    
    # Stops table
    if stops:
        writer.writerow(['# Detected Stops'])
        writer.writerow(['Stop #', 'Duration (s)', 'Samples', 'Position Std (m)', 
                        'Max Radius (m)', 'Latitude', 'Longitude'])
        for s in stops:
            writer.writerow([
                s.index, round(s.duration, 2), s.sample_count,
                round(s.pos_std, 4), round(s.pos_max_radius, 4),
                s.lat, s.lon
            ])
    
    return output.getvalue()


def generate_pdf_summary(
    summary: Dict,
    stops: List,
    gps_metrics: Any,
    recommendations: Any,
    verdict: Any
) -> bytes:
    """
    Generate PDF summary report.
    
    Returns:
        PDF content as bytes
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        elements.append(Paragraph("GPS/IMU Analysis Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", styles['Heading2']))
        
        # Verdict box
        verdict_color = colors.green if verdict.verdict == "SUITABLE" else (
            colors.orange if verdict.verdict == "MARGINAL" else colors.red
        )
        
        summary_data = [
            ['Duration', f"{summary.get('duration_min', 0):.1f} minutes"],
            ['Sample Rate', f"{summary.get('sample_rate', 0):.1f} Hz"],
            ['Total Distance', f"{summary.get('total_distance_m', 0):.1f} m"],
            ['Stop Count', str(summary.get('stop_count', 0))],
            ['% Stopped', f"{summary.get('stop_percent', 0):.1f}%"],
            ['Verdict', verdict.verdict],
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Verdict explanation
        elements.append(Paragraph(verdict.summary, styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        elements.append(Paragraph("Recommended Parameters", styles['Heading2']))
        
        rec_data = [
            ['Parameter', 'Value', 'Description'],
            ['Decision Zone Radius', f"{recommendations.decision_zone_radius:.1f} m", 
             'Trigger zone for decisions'],
            ['Approach Zone Radius', f"{recommendations.approach_zone_radius:.1f} m",
             'Early preparation zone'],
            ['Hysteresis', f"{recommendations.hysteresis:.1f} m", 
             'Exit margin'],
            ['Persistence (Moving)', f"{recommendations.persistence_moving_sec:.2f} s",
             'Entry confirmation time'],
        ]
        
        rec_table = Table(rec_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(rec_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Stops table
        if stops:
            elements.append(Paragraph("Detected Stops", styles['Heading2']))
            
            stop_data = [['#', 'Duration', 'GPS Noise (std)', 'Max Radius']]
            for s in stops[:10]:  # Limit to 10 stops
                stop_data.append([
                    str(s.index),
                    f"{s.duration:.1f} s",
                    f"{s.pos_std*100:.1f} cm",
                    f"{s.pos_max_radius*100:.1f} cm"
                ])
            
            if len(stops) > 10:
                stop_data.append(['...', f'({len(stops)-10} more)', '', ''])
            
            stop_table = Table(stop_data, colWidths=[0.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            stop_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(stop_table)
        
        # Footer
        elements.append(Spacer(1, 0.5*inch))
        footer_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        elements.append(Paragraph(footer_text, styles['Normal']))
        
        doc.build(elements)
        return buffer.getvalue()
        
    except ImportError:
        # Fallback if reportlab not available
        return b"PDF generation requires reportlab library"
