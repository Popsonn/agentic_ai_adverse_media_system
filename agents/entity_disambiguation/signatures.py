import dspy 

class SmartContextExtractor(dspy.Signature):
    """Extract key context for entity search and disambiguation"""
    text = dspy.InputField(desc="User input text about the entity")
    entity_name = dspy.InputField(desc="Target entity name")
    
    # Core search context
    roles = dspy.OutputField(desc="Professional roles as JSON array")
    organizations = dspy.OutputField(desc="Associated organizations as JSON array") 
    locations = dspy.OutputField(desc="Geographic locations as JSON array")
    aliases = dspy.OutputField(desc="Known aliases or alternative names as JSON array")
    
    # Optional contextual info
    additional_context = dspy.OutputField(desc="Other relevant context as JSON object")

class EntityValidator(dspy.Signature):
    """Validate if search result matches the target entity"""
    entity_name = dspy.InputField(desc="Target entity name")
    user_context = dspy.InputField(desc="User provided context")
    search_result = dspy.InputField(desc="Search result description")
    is_match = dspy.OutputField(desc="true/false if this is the same entity")
    confidence = dspy.OutputField(desc="Confidence score 0-1")
    reasoning = dspy.OutputField(desc="Brief explanation of decision")
