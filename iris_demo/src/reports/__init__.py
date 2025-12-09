"""
IRIS Report Generation

Generates clinical and narrative reports from session data.

Report Types:
  - slp_clinical: Clinical report for Speech-Language Pathologists
  - anonymous_story: Parent-friendly narrative reconstruction
  - social_story: Social story for intervention
"""

from .base import format_events_for_report, ReportGenerator
from .slp_clinical import SLPClinicalReport
from .anonymous_story import AnonymousStoryReport
from .social_story import SocialStoryReport

# Report type registry
REPORT_TYPES = {
    "slp_clinical": SLPClinicalReport,
    "anonymous_story": AnonymousStoryReport,
    "social_story": SocialStoryReport,
}


def generate_report(session, report_type: str, model: str = None) -> str:
    """
    Generate a report from session data.
    
    Args:
        session: Session object with events
        report_type: One of "slp_clinical", "anonymous_story", "social_story"
        model: Optional model override
    
    Returns:
        Generated report text
    """
    report_class = REPORT_TYPES.get(report_type)
    if not report_class:
        raise ValueError(f"Unknown report type: {report_type}. Available: {list(REPORT_TYPES.keys())}")
    
    report = report_class(model=model)
    return report.generate(session)


__all__ = [
    "generate_report",
    "format_events_for_report",
    "ReportGenerator",
    "SLPClinicalReport",
    "AnonymousStoryReport",
    "SocialStoryReport",
    "REPORT_TYPES",
]
