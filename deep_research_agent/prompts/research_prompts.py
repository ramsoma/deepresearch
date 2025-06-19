import dspy


class ResearchPrompt(dspy.Signature):
    """Base prompt for research tasks."""

    query: str = dspy.InputField(desc="The research query to investigate")
    context: str = dspy.InputField(desc="Additional context for the research")
    output: str = dspy.OutputField(desc="The research findings")


class AnalysisPrompt(dspy.Signature):
    """Prompt for analyzing research findings."""

    findings: str = dspy.InputField(desc="The research findings to analyze")
    criteria: str = dspy.InputField(desc="Analysis criteria")
    output: str = dspy.OutputField(desc="Analysis results")


class SynthesisPrompt(dspy.Signature):
    """Prompt for synthesizing multiple research findings."""

    findings_list: list = dspy.InputField(
        desc="List of research findings to synthesize"
    )
    output: str = dspy.OutputField(desc="Synthesized research summary")
