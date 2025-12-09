"""
SLP Clinical Report Generator

Generates structured clinical reports for Speech-Language Pathologists.
"""

from .base import ReportGenerator


class SLPClinicalReport(ReportGenerator):
    """Clinical report for Speech-Language Pathologists."""
    
    report_type = "slp_clinical"
    
    SYSTEM_PROMPT = """You are generating a clinical report for a Speech-Language Pathologist reviewing a classroom observation session.

Based on the event data provided, create a structured clinical report that includes:

1. **Session Overview**: Duration, setting, participants
2. **Communication Profile** (for each focus child):
   - Expressive language: complexity, frequency, targets
   - Receptive indicators: response to verbal input
   - Pragmatic skills: turn-taking, topic maintenance, social communication
3. **Social Interaction Patterns**:
   - Peer interactions: initiation vs response, quality, reciprocity
   - Adult interactions: help-seeking, compliance, joint attention
4. **Behavioral Observations**:
   - Self-regulation strategies observed and their effectiveness
   - Sensory responses and triggers
   - Emotional patterns
5. **Strengths Identified**: What's working well
6. **Areas for Support**: Specific intervention targets
7. **Recommendations**: 2-3 actionable next steps

Use clinical terminology appropriate for SLP documentation. Be specific with examples from the events.
Write in Hebrew."""
