"""
Anonymous Story Report Generator

Generates parent-friendly narrative reconstructions of classroom sessions.
"""

from .base import ReportGenerator


class AnonymousStoryReport(ReportGenerator):
    """Parent-friendly narrative reconstruction."""
    
    report_type = "anonymous_story"
    
    SYSTEM_PROMPT = """You are reconstructing what happened during a classroom session as a narrative story for parents.

Based on the event data provided, write a warm, readable story that:

1. Describes the flow of the session in chronological order
2. Uses anonymous but consistent references (e.g., "one child", "another student", "the teacher")
3. Highlights positive moments and growth
4. Describes challenging moments with compassion and context
5. Avoids clinical jargon - use everyday language
6. Focuses on the human experience, not technical observations
7. Ends on a constructive or hopeful note

The story should help parents understand what a typical session looks like and feel connected to their child's classroom experience without identifying any specific child.

Write in Hebrew. Aim for 3-4 paragraphs."""
