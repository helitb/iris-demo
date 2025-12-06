"""
Social Story Report Generator

Generates social stories for children with autism based on observed situations.
"""

from .base import ReportGenerator


class SocialStoryReport(ReportGenerator):
    """Social story for intervention."""
    
    report_type = "social_story"
    
    SYSTEM_PROMPT = """You are creating a social story for a child with autism based on a challenging situation observed in the classroom.

Based on the event data, identify the most significant challenging moment and create a social story that:

1. **Title**: Simple, descriptive title
2. **Situation**: Describe what happened in simple, concrete terms (2-3 sentences)
3. **Feeling**: Acknowledge the emotion the child might have felt (1-2 sentences)
4. **Strategy**: Provide 1-2 simple coping strategies (2-3 sentences)
5. **Positive Outcome**: Describe what happens when the strategy is used (1-2 sentences)
6. **Practice Phrase**: A simple phrase the child can remember

Use:
- First person perspective ("When I...")
- Simple, concrete language
- Present tense
- Short sentences
- Positive framing

Write in Hebrew. Keep the total length to about 150-200 words.
Format with clear section headers."""
